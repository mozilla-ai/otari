"""A workspace's policy over the deployment-wide code-execution sandbox.

The sandbox is an operator concern and stays one: its URL and any credential
live in the deployment's tool settings, and nothing here can point a workspace
at a different backend. What a policy row decides is *who on this deployment
may ask for code execution, and within which limits*.

Composition follows the rule in ``src/gateway/AGENTS.md`` (#655, settled in
#678): a workspace row may veto and may refine, never grant. So

* ``enabled=False`` refuses ``otari_code_execution`` for the workspace;
* ``max_iterations`` and ``exec_timeout_s`` are floored against what the
  request would otherwise get, so a value above the deployment's own ceiling
  narrows nothing rather than raising it;
* ``default_purpose_hint`` fills in only when the request named none, the same
  precedence the hybrid path applies to the policy it resolves from otari.ai;
* ``tools`` intersects the tool kinds the deployment's sandbox backend already
  serves, so it can only take one away, and a list that leaves nothing runnable
  refuses the request rather than serving an empty tool set;
* ``image`` names the sandbox image the workspace's code runs in, and may only
  name one the operator has already curated into ``sandbox_allowed_images``
  (plus the deployment's own ``sandbox_image``). A workspace-settable image is a
  supply-chain surface rather than a string, so the allow-list is the whole
  point of the column: without one, a workspace pins nothing;
* and **no row means no narrowing**, which is what makes a deployment that
  configures nothing behave exactly as it did.

The CRUD half is master-key routed (``routes/workspace_code_execution_policy.py``)
and role-gated per workspace: an organization owner/admin, or an owner/admin of
the workspace itself, may read *and* write it. Reads are gated too, which is a
departure from ``workspace_budget_default_service`` next door and a port of the
hosted service's own rule: code execution is a security and billing posture for
the whole workspace, not a per-member allowance. It is looser than the hosted
version in one way, admitting a workspace owner/admin and not only an
organization one, because that is the management gate every other
per-workspace surface in this repository uses
(``authorization.require_workspace_management_access``).

The request-path half is :func:`resolve_workspace_code_execution_policy`, a
plain read with no identity: the caller has already authenticated, and the
workspace comes off the key, never off a header (``services/workspace_scope``).
It is called from ``prepare_gateway_tools`` at admission, where the request's
session is live, and its values land on ``ToolContext``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import WorkspaceCodeExecutionPolicy
from gateway.models.tenancy import User, Workspace
from gateway.services.mcp_loop import MAX_TOOL_ITERATIONS_CAP
from gateway.services.sandbox_backend import CODE_EXECUTION_TOOL_NAMES, DEFAULT_EXEC_TIMEOUT_S
from gateway.services.tenancy import authorization
from gateway.services.tenancy.errors import SandboxImageNotAllowedError
from gateway.services.tenancy.organization_service import OrganizationService

# The two ceilings a workspace value is floored against, which are also the
# largest values worth storing: a policy may only narrow, so a number above the
# deployment's own ceiling would read as a configured limit and do nothing. The
# hosted service instead accepts any positive value and clamps it at resolve
# time; refusing it at the write is the better answer for a deployment whose
# operator is the same person, because a 422 says the invariant out loud where a
# silent clamp leaves a stored value nobody's request will ever see.
_MAX_ITERATIONS = MAX_TOOL_ITERATIONS_CAP
_MAX_EXEC_TIMEOUT_S = int(DEFAULT_EXEC_TIMEOUT_S)
# Matches the hosted column's own bound. An image reference longer than this is
# already pathological, and the column is ``String(255)``.
_MAX_IMAGE_LENGTH = 255


class WorkspaceCodeExecutionPolicyUpdate(BaseModel):
    """The policy to store for a workspace, as a whole.

    ``PUT`` semantics, ported from the hosted ``CodeExecutionConfigUpsert``:
    what is sent is what the workspace has afterwards, so an omitted limit is
    cleared rather than left as it was.
    """

    # Required rather than defaulted, which is where this parts company with the
    # hosted ``CodeExecutionConfigUpsert`` (``enabled: bool = False``). There, no
    # row means disabled, so an omitted flag and the stored default agree. Here no
    # row means *unnarrowed*, so either default would surprise somebody: an
    # omitted flag would silently turn the workspace off, or silently on.
    enabled: bool = Field(description="False refuses code execution for this workspace")
    default_purpose_hint: str | None = Field(
        default=None,
        max_length=2048,
        description="Hint used when a request declares otari_code_execution without one of its own",
    )
    max_iterations: int | None = Field(
        default=None,
        gt=0,
        le=_MAX_ITERATIONS,
        description=(
            f"Ceiling on tool-loop iterations; only ever lowers the effective limit, so at most {_MAX_ITERATIONS}"
        ),
    )
    exec_timeout_s: int | None = Field(
        default=None,
        gt=0,
        le=_MAX_EXEC_TIMEOUT_S,
        description=(
            "Ceiling on one execution's runtime in seconds; only ever lowers the effective limit, "
            f"so at most {_MAX_EXEC_TIMEOUT_S}"
        ),
    )
    image: str | None = Field(
        default=None,
        max_length=_MAX_IMAGE_LENGTH,
        description=(
            "Sandbox image this workspace's code runs in. Must be one the operator curated into "
            "sandbox_allowed_images (or the deployment's own sandbox_image); null uses the "
            "deployment's"
        ),
    )
    tools: list[str] | None = Field(
        default=None,
        description=(
            "Code-execution tool kinds this workspace may use, from "
            f"{', '.join(CODE_EXECUTION_TOOL_NAMES)}. Only ever removes one the backend serves; "
            "null exposes whatever it serves"
        ),
    )

    @field_validator("tools")
    @classmethod
    def _validate_tools(cls, value: list[str] | None) -> list[str] | None:
        """Refuse an unknown or empty tool list rather than storing one.

        Empty is refused in both stances, not only when ``enabled`` is true as
        the hosted ``_validate`` does: here a stored ``[]`` would be a third way
        of saying "refuse this workspace", and two spellings of one decision is
        how a surface ends up showing one and enforcing the other. ``null`` is
        the way to narrow nothing and ``enabled=False`` is the way to refuse.
        """
        if value is None:
            return None
        unknown = sorted({name for name in value if name not in CODE_EXECUTION_TOOL_NAMES})
        if unknown:
            msg = f"unknown tool(s): {', '.join(unknown)}; allowed: {', '.join(CODE_EXECUTION_TOOL_NAMES)}"
            raise ValueError(msg)
        # Order-preserving dedupe: the list is an unordered set semantically, and
        # storing a duplicate would show up twice in the dashboard's own controls.
        deduped = list(dict.fromkeys(value))
        if not deduped:
            msg = "tools must name at least one tool; use null to narrow nothing, or enabled=false to refuse"
            raise ValueError(msg)
        return deduped


class WorkspaceCodeExecutionPolicyPublic(BaseModel):
    """A workspace's policy, or the unconfigured policy it has without one."""

    workspace_id: uuid.UUID
    # False when the workspace has no row: everything below is then the
    # deployment's own behavior rather than a stored decision, which is what a
    # dashboard needs to say "not configured" instead of showing a policy
    # nobody set.
    configured: bool
    # Whether this deployment can run code execution at all, i.e. whether an
    # operator has pointed it at a sandbox. The OSS half of the hosted
    # ``CapabilityStatusPublic`` (otari-ai#1597): there the other half is a
    # licensing question this edition does not have. It is the ceiling this page
    # sits under, so a workspace toggled on where the deployment has no sandbox
    # reads as unavailable rather than as working. A boolean rather than the
    # hosted status enum plus reason string, because with one axis left there is
    # one thing to say and the dashboard says it in its own words.
    sandbox_configured: bool
    # The images this deployment's operator has curated, which is the whole set
    # ``image`` may be set to. Reported alongside the policy rather than from a
    # second endpoint because a form that offers a free-text image would be
    # offering something the write refuses; empty means the operator curated
    # none, and the dashboard says so instead of showing an empty picker.
    allowed_images: list[str]
    # The tool kinds ``tools`` may name. Fixed today, and reported rather than
    # hard-coded in the dashboard so the two cannot drift when a backend grows
    # one.
    available_tools: list[str]
    enabled: bool
    default_purpose_hint: str | None
    max_iterations: int | None
    exec_timeout_s: int | None
    image: str | None
    tools: list[str] | None
    created_at: str | None
    updated_at: str | None

    @classmethod
    def unconfigured(
        cls,
        workspace_id: uuid.UUID,
        *,
        sandbox_configured: bool,
        allowed_images: tuple[str, ...],
    ) -> WorkspaceCodeExecutionPolicyPublic:
        return cls(
            workspace_id=workspace_id,
            configured=False,
            sandbox_configured=sandbox_configured,
            allowed_images=list(allowed_images),
            available_tools=list(CODE_EXECUTION_TOOL_NAMES),
            enabled=True,
            default_purpose_hint=None,
            max_iterations=None,
            exec_timeout_s=None,
            image=None,
            tools=None,
            created_at=None,
            updated_at=None,
        )

    @classmethod
    def from_model(
        cls,
        policy: WorkspaceCodeExecutionPolicy,
        *,
        sandbox_configured: bool,
        allowed_images: tuple[str, ...],
    ) -> WorkspaceCodeExecutionPolicyPublic:
        return cls(
            workspace_id=policy.workspace_id,
            configured=True,
            sandbox_configured=sandbox_configured,
            allowed_images=list(allowed_images),
            available_tools=list(CODE_EXECUTION_TOOL_NAMES),
            enabled=policy.enabled,
            default_purpose_hint=policy.default_purpose_hint,
            max_iterations=policy.max_iterations,
            exec_timeout_s=policy.exec_timeout_s,
            image=policy.image,
            tools=list(policy.tools) if policy.tools is not None else None,
            created_at=policy.created_at.isoformat(),
            updated_at=policy.updated_at.isoformat(),
        )


@dataclass(frozen=True)
class ResolvedCodeExecutionPolicy:
    """What the request path reads off a stored policy.

    A value type rather than the ORM row, so the admission check cannot lazily
    touch the session after it has moved on, and so the tool context carries no
    ORM identity into a streaming response that outlives the request handler.
    """

    enabled: bool
    default_purpose_hint: str | None
    max_iterations: int | None
    exec_timeout_s: int | None
    image: str | None
    # ``frozenset`` rather than the stored list, because the request path only
    # ever asks whether a tool kind is in it, and an immutable one cannot be
    # edited by a backend it is handed to.
    tools: frozenset[str] | None


async def resolve_workspace_code_execution_policy(
    db: AsyncSession,
    workspace_id: uuid.UUID,
) -> ResolvedCodeExecutionPolicy | None:
    """The workspace's stored policy, or ``None`` when it has none.

    ``None`` and "a row that narrows nothing" are deliberately the same outcome
    for the caller; the distinction only matters to the management surface,
    which reports it as ``configured``.
    """
    policy = await db.get(WorkspaceCodeExecutionPolicy, workspace_id)
    if policy is None:
        return None
    return ResolvedCodeExecutionPolicy(
        enabled=policy.enabled,
        default_purpose_hint=policy.default_purpose_hint,
        max_iterations=policy.max_iterations,
        exec_timeout_s=policy.exec_timeout_s,
        image=policy.image,
        tools=frozenset(policy.tools) if policy.tools is not None else None,
    )


class WorkspaceCodeExecutionPolicyService:
    """Read and upsert one workspace's code-execution policy."""

    def __init__(self, db: AsyncSession, *, sandbox_configured: bool, allowed_images: tuple[str, ...] = ()):
        self.db = db
        self.organizations = OrganizationService(db)
        # Passed in rather than read here: whether a sandbox is configured, and
        # which images an operator curated, are questions about the running
        # deployment's config, which the route layer already holds and a service
        # has no business reaching for.
        self.sandbox_configured = sandbox_configured
        self.allowed_images = allowed_images

    async def get_policy(self, *, user: User, workspace_id: uuid.UUID) -> WorkspaceCodeExecutionPolicyPublic:
        """The workspace's policy. Reading it takes the same role as setting it."""
        workspace = await self._resolve_manageable(user=user, workspace_id=workspace_id)
        policy = await self.db.get(WorkspaceCodeExecutionPolicy, workspace.id)
        if policy is None:
            return self._unconfigured(workspace.id)
        return WorkspaceCodeExecutionPolicyPublic.from_model(
            policy, sandbox_configured=self.sandbox_configured, allowed_images=self.allowed_images
        )

    async def set_policy(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        request: WorkspaceCodeExecutionPolicyUpdate,
    ) -> WorkspaceCodeExecutionPolicyPublic:
        """Store the workspace's policy, replacing any existing one.

        Read-then-insert with nothing locking the gap, so two writers can both
        find no row and both insert; the primary key refuses the second. That
        refusal is not a conflict anybody needs to hear about, because a ``PUT``
        of the whole policy is idempotent: the loser re-reads the row the winner
        created and applies its own values over it, which is the same outcome it
        would have reached had it arrived a moment later. Narrower than
        ``organization_pricing_service``'s handling of the same race, which has
        to distinguish *which* period collided; here there is one row per
        workspace and nothing to disambiguate.
        """
        workspace = await self._resolve_manageable(user=user, workspace_id=workspace_id)
        self._require_allowed_image(request.image)
        # Read off the row once, here: a rollback below expires every instance in
        # the session, so `workspace.id` after one is a lazy load in a place that
        # cannot await it.
        resolved_id = workspace.id

        policy = await self.db.get(WorkspaceCodeExecutionPolicy, resolved_id)
        if policy is None:
            policy = WorkspaceCodeExecutionPolicy(workspace_id=resolved_id)
            self.db.add(policy)
        self._apply(policy, request)

        try:
            await self._commit()
        except IntegrityError:
            # `_commit` has already rolled back, which is what makes the reads
            # below usable: a failed flush leaves the session unusable, so
            # anything attempted before a rollback raises `PendingRollbackError`
            # and masks this. The retry commits through the same helper, so a
            # second failure rolls back too rather than leaving the caller a
            # session it cannot use.
            policy = await self.db.get(WorkspaceCodeExecutionPolicy, resolved_id)
            if policy is None:
                raise  # not the race: nothing is there to have collided with
            self._apply(policy, request)
            await self._commit()
        await self.db.refresh(policy)
        return WorkspaceCodeExecutionPolicyPublic.from_model(
            policy, sandbox_configured=self.sandbox_configured, allowed_images=self.allowed_images
        )

    def _require_allowed_image(self, image: str | None) -> None:
        """Refuse an image the operator has not curated.

        The whole reason ``image`` is a column and not a free string. A workspace
        owner is a lower privilege tier than the operator who runs this gateway,
        and an image is code that will execute here, so the set they may choose
        from is the operator's and not theirs. An operator who curated nothing
        has vetted nothing, and the refusal says that rather than pretending the
        value was malformed.

        Enforced again at admission (``prepare_gateway_tools``), because an
        operator may shrink the list after a workspace pinned from it.
        """
        candidate = _blank_to_none(image)
        if candidate is None or candidate in self.allowed_images:
            return
        if not self.allowed_images:
            raise SandboxImageNotAllowedError(
                "This deployment has curated no sandbox images, so a workspace cannot pin one. "
                "Set sandbox_allowed_images (or sandbox_image) on the gateway first."
            )
        raise SandboxImageNotAllowedError(
            f"Sandbox image {candidate!r} is not one this deployment allows. "
            f"Allowed: {', '.join(self.allowed_images)}."
        )

    async def _commit(self) -> None:
        """Commit, rolling back before any failure escapes.

        Every write path here goes through this rather than repeating the
        pattern, because the rollback is required and not tidy: SQLAlchemy
        leaves a session with a failed flush unusable, so a caller that skips it
        gets `PendingRollbackError` from the next statement instead of the error
        that actually happened.
        """
        try:
            await self.db.commit()
        except SQLAlchemyError:
            await self.db.rollback()
            raise

    @staticmethod
    def _apply(
        policy: WorkspaceCodeExecutionPolicy,
        request: WorkspaceCodeExecutionPolicyUpdate,
    ) -> None:
        """Write the whole request onto the row. Every field, since this is a ``PUT``."""
        policy.enabled = request.enabled
        policy.default_purpose_hint = _blank_to_none(request.default_purpose_hint)
        policy.max_iterations = request.max_iterations
        policy.exec_timeout_s = request.exec_timeout_s
        policy.image = _blank_to_none(request.image)
        policy.tools = request.tools

    async def clear_policy(self, *, user: User, workspace_id: uuid.UUID) -> WorkspaceCodeExecutionPolicyPublic:
        """Drop the workspace's policy, returning it to the deployment's behavior.

        Idempotent: a workspace that has no policy is already in the state this
        asks for, so it answers with the unconfigured policy rather than a 404.
        """
        workspace = await self._resolve_manageable(user=user, workspace_id=workspace_id)

        policy = await self.db.get(WorkspaceCodeExecutionPolicy, workspace.id)
        if policy is not None:
            await self.db.delete(policy)
            await self._commit()
        return self._unconfigured(workspace.id)

    def _unconfigured(self, workspace_id: uuid.UUID) -> WorkspaceCodeExecutionPolicyPublic:
        return WorkspaceCodeExecutionPolicyPublic.unconfigured(
            workspace_id, sandbox_configured=self.sandbox_configured, allowed_images=self.allowed_images
        )

    async def _resolve_manageable(self, *, user: User, workspace_id: uuid.UUID) -> Workspace:
        """Resolve a workspace the caller may see *and* manage.

        Visibility first, so a workspace the caller may not see answers 404
        rather than 403 and stays indistinguishable from one that does not
        exist; the role check then answers 403 for a member who may read the
        policy but not set it.
        """
        workspace = await authorization.resolve_visible_workspace(
            self.db, user=user, workspace_id=workspace_id, organizations=self.organizations
        )
        await authorization.require_workspace_management_access(
            self.db, user=user, workspace=workspace, organizations=self.organizations
        )
        return workspace


def _blank_to_none(value: str | None) -> str | None:
    """Treat a whitespace-only hint as absent.

    A cleared text input arrives as ``""``, and storing that would set the
    workspace's default hint to an empty string, which reads as "configured"
    while injecting nothing.
    """
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
