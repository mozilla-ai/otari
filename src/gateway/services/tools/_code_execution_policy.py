"""A workspace's policy over the deployment-wide code-execution sandbox.

The sandbox stays an operator concern, and a policy decides who may ask for code execution and within which limits.
A workspace row may veto and may refine, and it never grants.
``enabled=False`` refuses ``otari_code_execution`` for the workspace.
``max_iterations`` and ``exec_timeout_s`` can only lower what the request would otherwise get.
``default_purpose_hint`` applies only when the request names none.
``tools`` intersects the tool kinds the sandbox backend serves, and an empty result refuses the request.
``image`` may name only an image on the operator's allow-list, because a settable image is a supply-chain surface.
``executor`` pins who runs a provider-named code-execution declaration (``auto``, ``otari`` or ``provider``)
over the deployment default and the request's header. It is a choice rather than a narrowing, and it grants
no sandbox the deployment has not configured.
No row means no narrowing.
Reads and writes both require an owner or admin of the organization or of the workspace.
The request path's read, with no identity and the workspace from the key, is
``services/tenancy/workspace_code_execution_policy_service.resolve_workspace_code_execution_policy``.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field, field_validator

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.tools_exceptions import SandboxImageNotAllowedError, SandboxToolsUnrunnableError
from gateway.models.tenancy import User, Workspace
from gateway.models.tools import CodeExecutor, ResolvedCodeExecutionPolicy, WorkspaceCodeExecutionPolicy
from gateway.repositories.tools import WorkspaceCodeExecutionPolicyRepository
from gateway.services.sandbox_backend import (
    CODE_EXECUTION_TOOL_NAMES,
    DEFAULT_EXEC_TIMEOUT_S,
    SERVED_TOOL_NAMES,
)
from gateway.services.tenancy.authorization import WorkspaceAccess
from gateway.services.tools._loop_limits import MAX_TOOL_ITERATIONS_CAP

# A policy may only narrow, so a value above either ceiling would read as a
# configured limit and change nothing. The write refuses it rather than clamping
# it, so every stored limit is one a request can actually reach.
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
            "sandbox_allowed_session_images (or the deployment's own sandbox_session_image); null uses the "
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
    executor: CodeExecutor | None = Field(
        default=None,
        description=(
            "Who runs a provider-native code-execution declaration for this workspace: 'auto' (the "
            "provider when it runs the tool natively for the model, else this gateway's sandbox), "
            "'otari' or 'provider'. Pins over the deployment default and over the request's "
            "Otari-Code-Execution header; null leaves both in charge"
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

    @field_validator("executor", mode="before")
    @classmethod
    def _parse_executor(cls, value: object) -> object:
        """Accept the vocabulary in any case, and a blank string as no pin.

        An unparseable value passes through for the enum to refuse, because ``None`` would store no pin at all.
        """
        if isinstance(value, str) and not value.strip():
            return None
        parsed = CodeExecutor.parse(value)
        return value if parsed is None else parsed


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
    # The tool kinds this deployment's sandbox actually serves, which is what a
    # picker should offer: a control listing the whole *vocabulary* would offer
    # two options that narrow nothing and one that empties the set. Reported
    # rather than hard-coded in the dashboard so the two cannot drift when a
    # backend grows one.
    available_tools: list[str]
    enabled: bool
    default_purpose_hint: str | None
    max_iterations: int | None
    exec_timeout_s: int | None
    image: str | None
    tools: list[str] | None
    executor: CodeExecutor | None
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
            available_tools=list(SERVED_TOOL_NAMES),
            enabled=True,
            default_purpose_hint=None,
            max_iterations=None,
            exec_timeout_s=None,
            image=None,
            tools=None,
            executor=None,
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
            available_tools=list(SERVED_TOOL_NAMES),
            enabled=policy.enabled,
            default_purpose_hint=policy.default_purpose_hint,
            max_iterations=policy.max_iterations,
            exec_timeout_s=policy.exec_timeout_s,
            image=policy.image,
            tools=list(policy.tools) if policy.tools is not None else None,
            executor=CodeExecutor.parse(policy.executor),
            created_at=policy.created_at.isoformat(),
            updated_at=policy.updated_at.isoformat(),
        )


def read_code_execution_policy(answer: Mapping[str, Any]) -> ResolvedCodeExecutionPolicy:
    """Read the control plane's answer for one workspace's code execution policy.

    Raises ``ValueError`` when a field is malformed, so a policy that cannot be read fails closed.
    A ceiling above this gateway's own is read as sent, because it can only lower a limit.
    """
    enabled = answer.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")
    hint = answer.get("default_purpose_hint")
    if hint is not None and not isinstance(hint, str):
        raise ValueError("default_purpose_hint must be a string")
    tools = answer.get("tools")
    if tools is not None and (not isinstance(tools, list) or any(not isinstance(tool, str) for tool in tools)):
        raise ValueError("tools must be a list of strings")
    executor = answer.get("executor")
    if executor is not None and CodeExecutor.parse(executor) is None:
        raise ValueError("executor must be one of auto, otari or provider")
    return ResolvedCodeExecutionPolicy(
        enabled=enabled,
        default_purpose_hint=_blank_to_none(hint),
        max_iterations=_answer_ceiling(answer, "max_iterations"),
        exec_timeout_s=_answer_ceiling(answer, "exec_timeout_s"),
        image=None,
        tools=frozenset(tools) if tools is not None else None,
        executor=CodeExecutor.parse(executor),
    )


class WorkspaceCodeExecutionPolicyService:
    """Read and upsert one workspace's code-execution policy.

    Reads and writes run in a block of ``uow`` through ``policies``, so the block's end is the only commit.
    """

    def __init__(
        self,
        uow: UnitOfWork,
        policies: WorkspaceCodeExecutionPolicyRepository,
        access: WorkspaceAccess,
        *,
        sandbox_configured: bool,
        allowed_images: tuple[str, ...] = (),
    ):
        self._uow = uow
        self._policies = policies
        self._access = access
        # Passed in rather than read here: whether a sandbox is configured, and
        # which images an operator curated, are questions about the running
        # deployment's config, which the route layer already holds and a service
        # has no business reaching for.
        self.sandbox_configured = sandbox_configured
        self.allowed_images = allowed_images

    async def get_policy(self, *, user: User, workspace_id: uuid.UUID) -> WorkspaceCodeExecutionPolicyPublic:
        """The workspace's policy. Reading it takes the same role as setting it."""
        workspace = await self._resolve_manageable(user=user, workspace_id=workspace_id)
        async with self._uow:
            policy = await self._policies.get(workspace.id)
            if policy is None:
                return self._unconfigured(workspace.id)
            return self._public(policy)

    async def set_policy(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        request: WorkspaceCodeExecutionPolicyUpdate,
    ) -> WorkspaceCodeExecutionPolicyPublic:
        """Store the workspace's policy, replacing any existing one.

        Two writers may both find no row; the repository settles that race, so a
        ``PUT`` of the whole policy lands whichever arrives first.
        """
        workspace = await self._resolve_manageable(user=user, workspace_id=workspace_id)
        self._require_allowed_image(request.image)
        _require_runnable_tools(request.tools)
        async with self._uow:
            return self._public(await self._policies.put(workspace.id, **_stored_values(request)))

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
                "Set sandbox_allowed_session_images (or sandbox_session_image) on the gateway first."
            )
        raise SandboxImageNotAllowedError(
            f"Sandbox image {candidate!r} is not one this deployment allows. Allowed: {', '.join(self.allowed_images)}."
        )

    async def clear_policy(self, *, user: User, workspace_id: uuid.UUID) -> WorkspaceCodeExecutionPolicyPublic:
        """Drop the workspace's policy, returning it to the deployment's behavior.

        Idempotent: a workspace that has no policy is already in the state this
        asks for, so it answers with the unconfigured policy rather than a 404.
        """
        workspace = await self._resolve_manageable(user=user, workspace_id=workspace_id)
        async with self._uow:
            await self._policies.delete_for(workspace.id)
        return self._unconfigured(workspace.id)

    def _public(self, policy: WorkspaceCodeExecutionPolicy) -> WorkspaceCodeExecutionPolicyPublic:
        return WorkspaceCodeExecutionPolicyPublic.from_model(
            policy, sandbox_configured=self.sandbox_configured, allowed_images=self.allowed_images
        )

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
        workspace = await self._access.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        await self._access.require_workspace_management_access(user=user, workspace=workspace)
        return workspace


def _require_runnable_tools(tools: list[str] | None) -> None:
    """Refuse a tool list that leaves this deployment nothing to run.

    ``tools`` intersects what the backend serves, so a list naming only kinds it
    does not serve resolves to an empty set, and the request path answers 403.
    Storing it would be a second spelling of ``enabled=False`` that reads like a
    refinement, reachable from the dashboard by unticking one box, and the
    operator's only signal would be users reporting 403s later.

    This is the ``tools`` half of the guard ``_require_allowed_image`` gives
    ``image``. The request path still re-checks, for the same reason it re-checks
    the image: this asserts what is *storable*, and what a deployment serves can
    change under a row that was valid when it was written.
    """
    if tools is None or set(tools) & set(SERVED_TOOL_NAMES):
        return
    raise SandboxToolsUnrunnableError(SERVED_TOOL_NAMES)


def _stored_values(request: WorkspaceCodeExecutionPolicyUpdate) -> dict[str, Any]:
    """The whole request as the row stores it. Every field, since this is a ``PUT``."""
    return {
        "enabled": request.enabled,
        "default_purpose_hint": _blank_to_none(request.default_purpose_hint),
        "max_iterations": request.max_iterations,
        "exec_timeout_s": request.exec_timeout_s,
        "image": _blank_to_none(request.image),
        "tools": request.tools,
        "executor": request.executor.value if request.executor is not None else None,
    }


def _answer_ceiling(answer: Mapping[str, Any], field: str) -> int | None:
    """A ceiling from the control plane's answer, or ``None`` when it sends none."""
    value = answer.get(field)
    if value is None:
        return None
    # ``bool`` is an ``int`` subclass, so a JSON ``true`` would otherwise read as 1.
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


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
