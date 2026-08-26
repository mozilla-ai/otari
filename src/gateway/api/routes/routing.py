"""Runtime routing-policy management.

A policy is a model name callers use, which decides which real model serves the
request, what is tried after a retryable failure, and which guardrails always run.
``config.yml`` policies are read-only here (they are validated at startup and live
in a file this process does not own); these routes manage the ``routing_policies``
table, which means the same thing to a request but can change without a restart.

Scoping matches ``/v1/aliases``: a stored policy belongs to one workspace and,
within it, is either workspace-wide (``user_id`` omitted) or scoped to one user,
who is then the only caller that resolves it. Omitting ``workspace_id`` means the
deployment's default workspace, so a single-workspace deployment never names one.
See ``services/policy_store`` for precedence between the layers.

Master-key gated on every verb, like alias management. That is what makes a policy
safe as a unit of access: only an operator can decide which models a name reaches,
so a caller cannot widen their own access by writing a policy.
"""

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, require_deployment_operator
from gateway.api.routes._helpers import resolve_managed_workspace_id
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import RoutingPolicy
from gateway.models.routing import PolicySpec
from gateway.repositories.users_repository import get_active_user
from gateway.services.alias_service import all_alias_names
from gateway.services.policy_store import (
    all_policy_names,
    refresh_policy_cache,
    resolve_effective_policy,
)
from gateway.services.routing import (
    BudgetState,
    NoEligibleCandidatesError,
    backend_requires_pricing,
    compile_policy,
)
from gateway.services.routing.decide import explain_router_ordering
from gateway.services.routing.knn import unpriced_router_candidates

router = APIRouter(prefix="/v1/routing/policies", tags=["routing"])


class PolicyRequest(BaseModel):
    """Request to create or update a routing policy."""

    name: str = Field(description="Model name callers send, e.g. 'fast'.")
    spec: dict[str, Any] = Field(
        description=(
            "The policy body: select (with exactly one `default` entry, last), optional on_failure "
            "and guardrails. Same schema as a `routing.policies` entry in config.yml, and closed to "
            "unknown keys, so a typo is a 400 rather than a silently ignored setting."
        )
    )
    user_id: str | None = Field(
        default=None,
        description=(
            "User this policy belongs to. Omit for a policy every caller in the workspace sees. "
            "A user-scoped policy resolves only for that user and shadows the workspace-wide one "
            "of the same name."
        ),
    )
    workspace_id: uuid.UUID | None = Field(
        default=None,
        description=(
            "Workspace this policy belongs to. Omit for the deployment's default workspace. "
            "The policy resolves only for requests in that workspace, so two workspaces can each "
            "define their own 'fast'."
        ),
    )
    rename_from: str | None = Field(
        default=None,
        description=(
            "Current name of the policy to rename, in the same scope, meaning the same workspace and "
            "the same user. A rename never moves a policy between them. The stored row keeps its id and "
            "created_at and takes `name` and `spec`. Sending it asserts that policy exists, so a name with "
            "no stored row is a 404 rather than a create, even when it equals `name`. Omit to create or "
            "update the policy named `name`. Renaming changes what callers must send as `model`; usage "
            "already recorded keeps the old name."
        ),
    )


class PolicyResponse(BaseModel):
    """A routing policy and where it is defined."""

    name: str
    spec: dict[str, Any]
    # "config" for a config.yml policy (read-only here) or "stored" for a row in
    # routing_policies. Only stored policies can be edited or deleted.
    source: str
    user_id: str | None = None
    # The workspace the stored row lives in. Null for a config.yml policy, which
    # is deployment-wide and in force in every workspace.
    workspace_id: uuid.UUID | None = None
    # True when the selected candidate depends on request state (a condition or a
    # router), so the policy has no single target or price. Surfaced because it
    # changes where the policy can be used, and the dashboard needs to say so.
    is_dynamic: bool = False
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_model(cls, policy: RoutingPolicy, *, is_dynamic: bool) -> "PolicyResponse":
        return cls(
            name=policy.name,
            spec=policy.spec,
            source="stored",
            user_id=policy.user_id,
            workspace_id=policy.workspace_id,
            is_dynamic=is_dynamic,
            created_at=policy.created_at.isoformat() if policy.created_at else None,
            updated_at=policy.updated_at.isoformat() if policy.updated_at else None,
        )


class CandidateResponse(BaseModel):
    """One candidate in a compiled plan."""

    position: int
    instance: str
    model: str
    selection_reason: str
    dispatch_model: str


class DroppedResponse(BaseModel):
    """A candidate that was filtered out, and why."""

    selector: str
    reason: str
    detail: str


class ExplainRequest(BaseModel):
    """Ask what a policy would do, without dispatching anything.

    Either name a stored/configured policy (``name``) or pass a draft ``spec`` that
    has not been saved. The draft form is what makes authoring-time validation
    possible: the compiler filters candidates, so a chain can compile down to one
    attempt, and an author needs to see that before saving rather than during an
    outage.
    """

    name: str | None = Field(default=None, description="An existing policy to explain.")
    spec: dict[str, Any] | None = Field(default=None, description="An unsaved policy body to explain.")
    user_id: str | None = Field(default=None, description="Evaluate conditions as this user.")
    workspace_id: uuid.UUID | None = Field(
        default=None,
        description=(
            "Resolve `name` and the policy's candidate selectors in this workspace. "
            "Omit for the deployment's default workspace."
        ),
    )
    key_id: str | None = Field(default=None, description="Evaluate conditions as this API key id.")
    allowed_models: list[str] | None = Field(
        default=None,
        description="Simulate an API key's allow-list. Omit for unrestricted.",
    )
    budget_used_pct: float | None = Field(default=None, description="Simulated budget usage percentage.")
    budget_remaining_usd: float | None = Field(default=None, description="Simulated budget remaining, USD.")


class ExplainResponse(BaseModel):
    """The plan a policy compiles to for the given inputs."""

    name: str
    selection_reason: str
    is_dynamic: bool
    candidates: list[CandidateResponse]
    dropped: list[DroppedResponse]
    guardrails: list[dict[str, Any]]
    # Set when the policy hands its ordering to a router. For a router that needs
    # request state (kNN needs a prompt to embed and stored examples to compare it
    # against) the plan above is the *decline* path, because explain deliberately
    # dispatches nothing. Surfaced so the dashboard can say so rather than showing a
    # one-candidate plan that looks like the router was ignored. A weighted policy
    # is the exception: see `router_weights`.
    router_backend: str | None = None
    router_candidates: list[str] = Field(default_factory=list)
    router_weights: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "For a weighted policy, the percentage of traffic each candidate receives, normalized over "
            "the candidates this caller may use. Empty for every other policy, and for a weighted policy "
            "whose whole split this caller may not use: a split over no candidate is not a split, and each "
            "filtered candidate is named in `dropped` instead. A weighted split needs no request state, so "
            "unlike a learned router's ranking it is knowable here: the plan above is the real ordering by "
            "share, not the decline path."
        ),
    )


def _validated_spec(name: str, spec: dict[str, Any]) -> PolicySpec:
    """Parse a spec body into a ``PolicySpec``, or raise a 400 naming the field.

    The pydantic error is surfaced rather than flattened to a generic message: the
    schema's own messages explain the rules (one `default`, last; no `when` on a
    router entry; no threshold at 100), and a form can bind them to the field that
    is wrong.
    """
    try:
        return PolicySpec.model_validate(spec)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": f"routing policy '{name}' is not valid",
                # include_context=False is load-bearing: the context of a
                # value_error holds the original exception object, which is not
                # JSON-serializable and would turn a 400 into a 500.
                "errors": exc.errors(include_url=False, include_context=False),
            },
        ) from exc


def _validate_write(config: GatewayConfig, name: str, spec: PolicySpec, user_id: str | None) -> None:
    """Apply the startup policy rules to a runtime write, as a 400.

    A configured policy wins over a *workspace-wide* stored one during resolution,
    so storing a workspace-wide name that shadows one would be accepted and then
    never take effect. Refusing is the only answer that does not lie about what the gateway
    will do. A user-scoped policy is exempt: it outranks both other layers, so
    shadowing a configured name is a working override rather than dead data.
    """
    if user_id is None and name in config.routing.policies:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"'{name}' is already a routing policy in config.yml. Config policies take precedence over "
                "workspace-wide stored ones, so this one would never be used. Rename it, scope it to a user, or "
                "edit config.yml."
            ),
        )
    if name in all_alias_names(config):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"'{name}' is already an alias. An alias and a policy would claim the same model name for "
                "callers, leaving one of them dead. Rename the policy, or delete the alias and express it "
                "as this policy's default target."
            ),
        )
    # Chaining is refused across every scope, matching the alias rule: a policy
    # pointing at another policy or an alias is just as broken whichever side each
    # came from, and resolution is single-pass.
    indirections = all_policy_names(config) | all_alias_names(config) | {name}
    for selector in spec.static_selectors():
        try:
            config.validate_alias(name, selector, alias_names=indirections)
        except ValueError as exc:
            # `validate_alias` phrases both its target errors and its *name* errors
            # in alias terms, so both are rewritten; otherwise a policy named "a:b"
            # would be refused with a message about aliases.
            detail = (
                str(exc)
                .replace(f"aliases.{name}", f"routing policy '{name}'", 1)
                .replace("alias name", "routing policy name", 1)
            )
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail) from exc
    if spec.default_target in spec.on_failure:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"'{spec.default_target}' is both the default target and an on_failure entry. Retrying the "
                "candidate that just failed cannot help; remove it from on_failure."
            ),
        )


async def _validate_router_pricing(
    config: GatewayConfig, db: AsyncSession, spec: PolicySpec, workspace_id: uuid.UUID
) -> None:
    """Refuse a learned policy whose candidates are not all priced.

    The *learned* router scores by cost, so one unpriced candidate makes it decline
    every request and the policy silently serves its default target forever. That is
    indistinguishable from a broken router, and the fix (add pricing) is nothing
    the operator would think to look for. Startup only *warns* about the same
    problem in a config policy, because refusing there would take a running
    gateway down over an optimization; refusing a write costs one corrected
    request while the operator is looking at the policy.

    Scoped to the backends that actually read a price: the weighted router balances
    on operator-declared capacity, so requiring pricing there would refuse a policy
    that works. Candidates resolve in the workspace the policy is being stored
    into, so one naming an alias is validated as it will resolve for the requests
    this policy will actually serve.
    """
    if not backend_requires_pricing(spec.router_backend):
        return
    missing = await unpriced_router_candidates(
        config, db, spec.router_candidates, workspace_id=workspace_id
    )
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Router candidate(s) {', '.join(missing)} have no pricing. A router scores candidates by "
                "cost, so it would decline every request and this policy would always serve "
                f"'{spec.default_target}'. Add pricing for those models (POST /v1/pricing) first."
            ),
        )


async def _require_user(db: AsyncSession, user_id: str) -> None:
    """404 unless ``user_id`` names a live user.

    Unknown ids have to be caught here: the column is a foreign key, so one would
    otherwise surface as an opaque 500 from the commit. Soft-deleted users are
    rejected for the same reason every other user-scoped route uses
    ``get_active_user``: they cannot authenticate, so the policy would be dead on
    arrival.
    """
    if await get_active_user(db, user_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User '{user_id}' not found")


def _missing_policy_detail(
    config: GatewayConfig, name: str, user_id: str | None, verb: str, workspace_id: uuid.UUID
) -> str:
    """Explain a stored policy that is not there, naming the scope that was searched.

    A config.yml policy is visible in the listing but has no row, so "not found" on
    its own reads as a bug. Saying which of the two it is turns the 404 into the
    answer: edit config.yml, or check the scope.
    """
    if user_id is None and name in config.routing.policies:
        return f"Routing policy '{name}' is defined in config.yml and cannot be {verb} through the API."
    scope = "workspace-wide" if user_id is None else f"scoped to user '{user_id}'"
    return f"Routing policy '{name}' ({scope}) not found in workspace '{workspace_id}'"


async def _name_is_taken(db: AsyncSession, name: str, user_id: str | None, workspace_id: uuid.UUID) -> bool:
    """True when a stored policy already answers to ``name`` in this scope."""
    existing = (
        await db.execute(
            select(RoutingPolicy.id).where(
                RoutingPolicy.workspace_id == workspace_id,
                RoutingPolicy.name == name,
                RoutingPolicy.user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    return existing is not None


def _name_taken_detail(name: str) -> str:
    return f"Routing policy '{name}' already exists in this scope. Delete it first, or pick another name."


async def _refresh_quietly(db: AsyncSession, name: str) -> None:
    """Refresh this worker's cache after a committed write.

    The write already succeeded, so a refresh failure must not turn it into a 500.
    This worker picks the change up on its next background refresh; others converge
    within the TTL.
    """
    try:
        await refresh_policy_cache(db)
    except SQLAlchemyError:
        logger.warning("Policy cache refresh failed after writing '%s'; converges within TTL", name)


@router.get("", dependencies=[Depends(require_deployment_operator)])
async def list_policies(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    workspace_id: Annotated[
        uuid.UUID | None,
        Query(
            description=(
                "Only stored policies in this workspace. Config-file policies are always included, "
                "being deployment-wide. Omit to list the stored policies of every workspace."
            )
        ),
    ] = None,
) -> list[PolicyResponse]:
    """List every routing policy in force, from config.yml and from storage.

    Every scope at once, workspace-wide and user-scoped alike: this is the
    master-key management view, not what any one caller resolves.
    """
    statement = select(RoutingPolicy)
    if workspace_id is not None:
        # Stored rows only. A config.yml entry has no workspace: it is
        # deployment-wide and in force in every one of them, so filtering
        # it out would misreport what actually resolves.
        statement = statement.where(RoutingPolicy.workspace_id == workspace_id)
    rows = (await db.execute(statement.order_by(RoutingPolicy.name))).scalars().all()
    merged: dict[tuple[uuid.UUID | None, str, str | None], PolicyResponse] = {}
    for row in rows:
        key = (row.workspace_id, row.name, row.user_id)
        try:
            parsed = PolicySpec.model_validate(row.spec)
        except ValidationError:
            # Listed rather than hidden: an operator has to be able to see and fix a
            # row this build cannot parse, and hiding it would make the dashboard
            # disagree with the database.
            logger.warning("Stored routing policy %r does not validate; listing it as-is", row.name)
            merged[key] = PolicyResponse.from_model(row, is_dynamic=False)
            continue
        merged[key] = PolicyResponse.from_model(row, is_dynamic=parsed.is_dynamic)
    # Config last, matching effective_policies: a configured name beats the stored
    # workspace-wide row in every workspace, so it is listed once, unscoped,
    # instead of shadowing each workspace's row in place.
    for name, spec in config.routing.policies.items():
        merged[(None, name, None)] = PolicyResponse(
            name=name,
            spec=spec.model_dump(mode="json", exclude_none=True),
            source="config",
            is_dynamic=spec.is_dynamic,
        )
    return sorted(
        merged.values(),
        key=lambda policy: (policy.name, str(policy.workspace_id or ""), policy.user_id or ""),
    )


@router.post("", dependencies=[Depends(require_deployment_operator)])
async def set_policy(
    request: PolicyRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> PolicyResponse:
    """Create or update a stored policy in one workspace, optionally for one user.

    The spec is validated here and stored as given, so a row can never contain a
    body this build would refuse at load. The cache is refreshed twice: once before
    validating (so the shadowing checks see other writers' policies) and once after
    committing (so this worker serves the new policy immediately).

    ``rename_from`` renames the row instead of keying on ``name``. It is part of
    this write rather than an endpoint of its own so that an edit which both renames
    a policy and re-targets it cannot land half-applied, leaving the old name serving
    the new spec. The new name is validated exactly as a fresh one is, because a
    rename can walk a policy into every collision a create can. Sending the field
    asserts the named policy is stored, so it never falls back to creating one.
    """
    if request.user_id is not None:
        await _require_user(db, request.user_id)
    workspace_id = await resolve_managed_workspace_id(db, request.workspace_id)
    spec = _validated_spec(request.name, request.spec)
    await refresh_policy_cache(db)
    _validate_write(config, request.name, spec, request.user_id)
    await _validate_router_pricing(config, db, spec, workspace_id)

    # Both scopes are part of the identity: the upsert must not turn one
    # workspace's policy into another's, nor a workspace-wide policy into a
    # user-scoped one (or vice versa), just because the names match. A rename
    # moves the name half of that key and leaves both scopes alone.
    renaming = request.rename_from is not None and request.rename_from != request.name
    lookup_name = request.rename_from if request.rename_from is not None else request.name
    policy = (
        await db.execute(
            select(RoutingPolicy).where(
                RoutingPolicy.workspace_id == workspace_id,
                RoutingPolicy.name == lookup_name,
                RoutingPolicy.user_id == request.user_id,
            )
        )
    ).scalar_one_or_none()
    # The 404 keys on ``rename_from`` being sent at all rather than on the name having
    # changed: the field asserts that row exists, so naming a policy that is not
    # stored is answered instead of being quietly turned into a create. That holds for
    # an unchanged name too, where the alternative is a create dressed up as an edit.
    if request.rename_from is not None:
        if policy is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=_missing_policy_detail(config, lookup_name, request.user_id, "renamed", workspace_id),
            )
        if renaming:
            # Without this the rename would be an upsert onto the target name, silently
            # destroying whatever policy already answered to it.
            if await _name_is_taken(db, request.name, request.user_id, workspace_id):
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=_name_taken_detail(request.name))
            policy.name = request.name
    # Round-tripped through the model so the stored document is normalized (defaults
    # filled, key order stable) rather than whatever shape the client happened to
    # send. Otherwise two equivalent writes would produce different rows.
    stored_spec = spec.model_dump(mode="json", exclude_none=True)
    if policy:
        policy.spec = stored_spec
    else:
        policy = RoutingPolicy(
            name=request.name,
            spec=stored_spec,
            user_id=request.user_id,
            workspace_id=workspace_id,
        )
        db.add(policy)

    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        # The name check above and this commit are not one atomic step, so another
        # writer can claim the name in between and the unique constraint is what
        # catches it. Re-read rather than assuming: the same constraint class also
        # covers the user foreign key, and reporting a deleted user as a name clash
        # would send the operator after the wrong thing.
        if await _name_is_taken(db, request.name, request.user_id, workspace_id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=_name_taken_detail(request.name)
            ) from None
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error"
        ) from None
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error"
        ) from None
    await db.refresh(policy)
    # An operator changing where traffic goes is worth a line in the log: this is the
    # object that decides which model spends money.
    logger.info(
        "Routing policy written name=%s renamed_from=%s workspace=%s scope=%s candidates=%d dynamic=%s router=%s",
        policy.name,
        request.rename_from if renaming else "-",
        policy.workspace_id,
        policy.user_id or "workspace-wide",
        # A router entry contributes its whole pool, since the walker cascades
        # through the ranking. Counting one head candidate here logged
        # "candidates=1" for a policy that can dispatch three.
        (len(spec.router_candidates) or 1) + len(spec.on_failure),
        spec.is_dynamic,
        spec.router_backend or "none",
    )
    await _refresh_quietly(db, policy.name)
    return PolicyResponse.from_model(policy, is_dynamic=spec.is_dynamic)


@router.delete(
    "/{name:path}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_deployment_operator)],
)
async def delete_policy(
    name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    user_id: Annotated[
        str | None,
        Query(description="Delete the policy scoped to this user. Omit to delete the workspace-wide one."),
    ] = None,
    workspace_id: Annotated[
        uuid.UUID | None,
        Query(description="Delete the policy in this workspace. Omit for the deployment's default workspace."),
    ] = None,
) -> None:
    """Delete a stored policy in one scope.

    Scoped by ``workspace_id`` and ``user_id`` for the same reason the upsert is:
    deleting one workspace's policy must not touch another's, deleting the
    workspace-wide policy must not take a user's override with it, and deleting an
    override must leave the workspace-wide one serving everyone else.
    """
    target_workspace_id = await resolve_managed_workspace_id(db, workspace_id)
    policy = (
        await db.execute(
            select(RoutingPolicy).where(
                RoutingPolicy.workspace_id == target_workspace_id,
                RoutingPolicy.name == name,
                RoutingPolicy.user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    if policy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=_missing_policy_detail(config, name, user_id, "deleted", target_workspace_id),
        )

    await db.delete(policy)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error"
        ) from None
    await _refresh_quietly(db, name)


@router.post("/explain", dependencies=[Depends(require_deployment_operator)])
async def explain_policy(
    request: ExplainRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> ExplainResponse:
    """Compile a policy and return the plan, without dispatching anything.

    Master-key gated, and deliberately so: the response enumerates the policy's
    targets, which is exactly the information a policy exists to keep off the wire.
    It is a management surface, not a caller-facing one.

    Accepts an unsaved ``spec`` as well as a saved ``name``, so a form can validate
    what the operator is about to save. The response includes dropped candidates
    with reasons, which is the part that catches a "failover" policy that has
    quietly compiled down to a single attempt.
    """
    if request.name is None and request.spec is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pass `spec` (a draft to check), `name` (an existing policy), or both.",
        )

    if request.spec is not None:
        # A draft wins over the stored version of the same name. Sending both is the
        # normal editing flow: the operator is looking at policy "fast" and wants to
        # know what their unsaved edit would do, with the name kept for the label.
        name = request.name or "(draft)"
        spec = _validated_spec(name, request.spec)
    else:
        assert request.name is not None
        name = request.name
        resolved = resolve_effective_policy(config, name, request.user_id, workspace_id=request.workspace_id)
        if resolved is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Routing policy '{name}' not found"
            )
        spec = resolved

    # A weighted policy's ordering is computable here: it needs no prompt, no
    # stored examples and no provider call, only the weights the policy declares.
    # Passing it in makes explain show the split rather than the decline path.
    weighted_ordering, weighted_shares = explain_router_ordering(
        config,
        spec,
        user_id=request.user_id,
        allowlist=request.allowed_models,
        workspace_id=request.workspace_id,
    )
    try:
        plan = compile_policy(
            config,
            name,
            spec,
            user_id=request.user_id,
            key_id=request.key_id,
            allowlist=request.allowed_models,
            budget=BudgetState(
                used_pct=request.budget_used_pct, remaining_usd=request.budget_remaining_usd
            ),
            router_ordering=weighted_ordering,
            workspace_id=request.workspace_id,
        )
    except NoEligibleCandidatesError as exc:
        return ExplainResponse(
            name=name,
            selection_reason="none",
            is_dynamic=spec.is_dynamic,
            router_backend=spec.router_backend,
            router_candidates=spec.router_candidates,
            candidates=[],
            dropped=[
                DroppedResponse(selector=item.selector, reason=item.reason, detail=item.detail)
                for item in exc.dropped
            ],
            guardrails=[],
        )

    return ExplainResponse(
        name=name,
        selection_reason=plan.selection_reason,
        is_dynamic=spec.is_dynamic,
        router_backend=spec.router_backend,
        router_candidates=spec.router_candidates,
        router_weights={item.selector: round(item.share_pct, 2) for item in weighted_shares},
        candidates=[
            CandidateResponse(
                position=attempt.position,
                instance=attempt.instance,
                model=attempt.model,
                selection_reason=attempt.selection_reason,
                dispatch_model=attempt.dispatch_model,
            )
            for attempt in plan.attempts
        ],
        dropped=[
            DroppedResponse(selector=item.selector, reason=item.reason, detail=item.detail)
            for item in plan.dropped
        ],
        guardrails=[guardrail.model_dump(mode="json", exclude_none=True) for guardrail in plan.guardrails],
    )
