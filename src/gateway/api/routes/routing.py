"""Runtime routing-policy management.

A policy is a model name callers use, which decides which real model serves the
request, what is tried after a retryable failure, and which guardrails always run.
``config.yml`` policies are read-only here (they are validated at startup and live
in a file this process does not own); these routes manage the ``routing_policies``
table, which means the same thing to a request but can change without a restart.

Scoping matches ``/v1/aliases``: a stored policy is either global (``user_id``
omitted) or scoped to one user, who is then the only caller that resolves it. See
``services/policy_store`` for precedence between the layers.

Master-key gated on every verb, like alias management. That is what makes a policy
safe as a unit of access: only an operator can decide which models a name reaches,
so a caller cannot widen their own access by writing a policy.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, verify_master_key
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
from gateway.services.routing import BudgetState, NoEligibleCandidatesError, compile_policy

router = APIRouter(prefix="/v1/routing/policies", tags=["routing"])


class PolicyRequest(BaseModel):
    """Request to create or update a routing policy."""

    name: str = Field(description="Model name callers send, e.g. 'fast'.")
    spec: dict[str, Any] = Field(
        description=(
            "The policy body: select (with exactly one `default` entry, last), optional on_failure, "
            "guardrails, and limits. Same schema as a `routing.policies` entry in config.yml."
        )
    )
    user_id: str | None = Field(
        default=None,
        description=(
            "User this policy belongs to. Omit for a global policy every caller sees. "
            "A user-scoped policy resolves only for that user and shadows a global one of the same name."
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

    A configured policy wins over a *global* stored one during resolution, so
    storing a global name that shadows one would be accepted and then never take
    effect. Refusing is the only answer that does not lie about what the gateway
    will do. A user-scoped policy is exempt: it outranks both other layers, so
    shadowing a configured name is a working override rather than dead data.
    """
    if user_id is None and name in config.routing.policies:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"'{name}' is already a routing policy in config.yml. Config policies take precedence over "
                "global stored ones, so this one would never be used. Rename it, scope it to a user, or "
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


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_policies(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> list[PolicyResponse]:
    """List every routing policy in force, from config.yml and from storage.

    Every scope at once, global and user-scoped alike: this is the master-key
    management view, not what any one caller resolves.
    """
    rows = (await db.execute(select(RoutingPolicy).order_by(RoutingPolicy.name))).scalars().all()
    merged: dict[tuple[str, str | None], PolicyResponse] = {}
    for row in rows:
        try:
            parsed = PolicySpec.model_validate(row.spec)
        except ValidationError:
            # Listed rather than hidden: an operator has to be able to see and fix a
            # row this build cannot parse, and hiding it would make the dashboard
            # disagree with the database.
            logger.warning("Stored routing policy %r does not validate; listing it as-is", row.name)
            merged[(row.name, row.user_id)] = PolicyResponse.from_model(row, is_dynamic=False)
            continue
        merged[(row.name, row.user_id)] = PolicyResponse.from_model(row, is_dynamic=parsed.is_dynamic)
    # Config last, matching effective_policies: if a global name somehow exists on
    # both sides, list the one that would actually resolve rather than both.
    for name, spec in config.routing.policies.items():
        merged[(name, None)] = PolicyResponse(
            name=name,
            spec=spec.model_dump(mode="json", exclude_none=True),
            source="config",
            is_dynamic=spec.is_dynamic,
        )
    return sorted(merged.values(), key=lambda policy: (policy.name, policy.user_id or ""))


@router.post("", dependencies=[Depends(verify_master_key)])
async def set_policy(
    request: PolicyRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> PolicyResponse:
    """Create or update a stored policy, global or scoped to one user.

    The spec is validated here and stored as given, so a row can never contain a
    body this build would refuse at load. The cache is refreshed twice: once before
    validating (so the shadowing checks see other writers' policies) and once after
    committing (so this worker serves the new policy immediately).
    """
    if request.user_id is not None:
        await _require_user(db, request.user_id)
    spec = _validated_spec(request.name, request.spec)
    await refresh_policy_cache(db)
    _validate_write(config, request.name, spec, request.user_id)

    # Scope is part of the identity: the upsert must not turn a global policy into
    # a user-scoped one (or vice versa) just because the names match.
    policy = (
        await db.execute(
            select(RoutingPolicy).where(
                RoutingPolicy.name == request.name, RoutingPolicy.user_id == request.user_id
            )
        )
    ).scalar_one_or_none()
    # Round-tripped through the model so the stored document is normalized (defaults
    # filled, key order stable) rather than whatever shape the client happened to
    # send. Otherwise two equivalent writes would produce different rows.
    stored_spec = spec.model_dump(mode="json", exclude_none=True)
    if policy:
        policy.spec = stored_spec
    else:
        policy = RoutingPolicy(name=request.name, spec=stored_spec, user_id=request.user_id)
        db.add(policy)

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error"
        ) from None
    await db.refresh(policy)
    # An operator changing where traffic goes is worth a line in the log: this is the
    # object that decides which model spends money.
    logger.info(
        "Routing policy written name=%s scope=%s candidates=%d dynamic=%s",
        policy.name,
        policy.user_id or "global",
        1 + len(spec.on_failure),
        spec.is_dynamic,
    )
    await _refresh_quietly(db, policy.name)
    return PolicyResponse.from_model(policy, is_dynamic=spec.is_dynamic)


@router.delete("/{name:path}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(verify_master_key)])
async def delete_policy(
    name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    user_id: Annotated[
        str | None,
        Query(description="Delete the policy scoped to this user. Omit to delete the global one."),
    ] = None,
) -> None:
    """Delete a stored policy in one scope.

    Scoped by ``user_id`` for the same reason the upsert is: deleting the global
    policy must not take a user's override with it, and deleting an override must
    leave the global one serving everyone else.
    """
    policy = (
        await db.execute(
            select(RoutingPolicy).where(RoutingPolicy.name == name, RoutingPolicy.user_id == user_id)
        )
    ).scalar_one_or_none()
    if policy is None:
        scope = "global" if user_id is None else f"scoped to user '{user_id}'"
        detail = f"Routing policy '{name}' ({scope}) not found"
        if user_id is None and name in config.routing.policies:
            detail = f"Routing policy '{name}' is defined in config.yml and cannot be deleted through the API."
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)

    await db.delete(policy)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error"
        ) from None
    await _refresh_quietly(db, name)


@router.post("/explain", dependencies=[Depends(verify_master_key)])
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
        resolved = resolve_effective_policy(config, name, request.user_id)
        if resolved is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Routing policy '{name}' not found"
            )
        spec = resolved

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
        )
    except NoEligibleCandidatesError as exc:
        return ExplainResponse(
            name=name,
            selection_reason="none",
            is_dynamic=spec.is_dynamic,
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
