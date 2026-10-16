"""Admit the code execution one request declared: who runs it, and the sandbox it runs in."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gateway.exceptions import TenancyError
from gateway.exceptions.tools_exceptions import (
    CodeExecutionContainerBusyError,
    CodeExecutionDeclarationError,
    CodeExecutionRefusedError,
)
from gateway.models.tools import CodeExecutor, ResolvedCodeExecutionPolicy
from gateway.services.code_execution import (
    CONTAINER_AUTO,
    CONTAINER_ID_PREFIX,
    ContainerBusyError,
    ContainerLease,
    ContainerNotFoundError,
    SandboxContainerRegistry,
    gateway_container_value,
    requested_container,
)
from gateway.services.sandbox_backend import SERVED_TOOL_NAMES
from gateway.services.tools import _code_execution_tool
from gateway.services.tools._candidate_tools import ProviderToolDeclaration
from gateway.services.tools._code_execution_declarations import (
    CODE_EXECUTION_HEADER,
    decide_code_executor,
    extract_code_execution_tool,
    first_provider_code_execution_tool,
    parse_code_execution_header,
    resolve_code_executor_preference,
)

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig

SANDBOX_NOT_CONFIGURED_DETAIL = (
    "otari_code_execution tool requested but no sandbox is configured on this gateway. "
    "Set OTARI_SANDBOX_URL on the gateway, or remove otari_code_execution from `tools`."
)
CODE_EXECUTOR_NOT_CONFIGURED_DETAIL = (
    "code execution was asked to run on this gateway but no sandbox is configured. "
    "Set OTARI_SANDBOX_URL on the gateway, or let the provider run it."
)
CODE_EXECUTION_HEADER_INVALID_DETAIL = f"{CODE_EXECUTION_HEADER} must be one of auto, otari, provider"
CODE_EXECUTOR_PINNED_DETAIL = (
    f"this workspace's code-execution policy decides who runs code; the {CODE_EXECUTION_HEADER} "
    "header cannot choose otherwise"
)
SANDBOX_PROVIDER_TOOL_CONFLICT_DETAIL = (
    "otari_code_execution cannot be combined with a provider-native code-execution tool "
    "(code_execution, code_interpreter, code_execution_<date>) in the same request; pick one. "
    "The gateway sandbox and the provider's own are separate environments, and a request "
    "addressing both has no single place its files and state live."
)
SANDBOX_NOT_ENABLED_DETAIL = "code execution is not enabled for this workspace"
SANDBOX_TOOLS_EXCLUDED_DETAIL = (
    "code execution is not available to this workspace: its policy's tool list excludes "
    "every tool kind this gateway's sandbox serves."
)
# Says what happened and nothing an API caller cannot act on. The setting to
# change is named on the management surface, which the operator reaches; naming
# it here would send an operator instruction to a data-plane caller, which is the
# boundary ``SANDBOX_NOT_ENABLED_DETAIL`` next door already respects.
SANDBOX_IMAGE_NOT_ALLOWED_DETAIL = "this workspace's code-execution policy pins a sandbox image that is not allowed"
# One detail for an unknown, expired, foreign or other-provider container, so an
# id never reveals which. The phrasing is the one clients of Anthropic's and
# OpenAI's containers already recognize as "drop the id and start over".
CONTAINER_GONE_DETAIL_TEMPLATE = "Container '{container_id}' has expired or does not exist."
CONTAINER_NOT_GATEWAY_RUN_DETAIL = (
    "container names a sandbox this gateway holds, and the code execution for this request runs on the "
    "provider, which cannot reach it. Drop the field, or send Otari-Code-Execution: otari to run the "
    "code here."
)
CONTAINER_BUSY_DETAIL = (
    "Container is in use by another request. A sandbox runs one request at a time; retry when it finishes."
)
# The id is echoed back so a client can tell which of several it should drop,
# and it arrives from the request body as an unbounded string, so what is echoed
# is clipped and stripped of anything that is not a plain printable character.
# A real one is ``otari_cntr_`` and 32 hex digits.
_CONTAINER_ID_ECHO_LIMIT = 64


def _echoable_container_id(value: str) -> str:
    """The caller's container id, safe to put in an error body.

    Printable ASCII only and bounded: the field is a bare string on the wire, so
    without this a megabyte of anything the caller likes comes back in the 400.
    """
    cleaned = "".join(char for char in value if char.isascii() and char.isprintable())
    if len(cleaned) > _CONTAINER_ID_ECHO_LIMIT:
        return cleaned[:_CONTAINER_ID_ECHO_LIMIT] + "…"
    return cleaned


@dataclass(frozen=True)
class AdmittedCodeExecution:
    """Who runs a request's code on one candidate's attempt, and the sandbox it runs in when the gateway does."""

    allowed_tools: frozenset[str] | None
    container_lease: ContainerLease | None
    containers: SandboxContainerRegistry | None
    exec_timeout_s: int | None
    executor: CodeExecutor | None
    max_iterations: int | None
    session_image: str | None
    tool_entry: dict[str, Any] | None
    tools_after_sandbox: list[dict[str, Any]] | None
    use_sandbox: bool


async def declare_code_execution(
    tools: list[dict[str, Any]] | None,
    *,
    container_id: object,
    code_execution_header: str | None,
    config: GatewayConfig,
    resolve_policy: Callable[[], Awaitable[ResolvedCodeExecutionPolicy | None]],
    container_registry: SandboxContainerRegistry | None,
) -> DeclaredCodeExecution:
    """Read the code execution one request declared, refusing what no candidate could serve.

    ``resolve_policy`` reads the workspace's code execution policy, and is awaited only when the
    request declares code execution and this deployment has a sandbox to run it.
    ``container_registry`` is where held sandboxes are leased, ``None`` where this deployment holds none.

    Raises:
        CodeExecutionDeclarationError: the declaration cannot be served as written.
        CodeExecutionRefusedError: the header asks for an executor the workspace's policy pins away.
    """
    # The deployment decides whether code can run here, because a hosted provider has no URL.
    sandbox_available = config.sandbox_configured()
    try:
        requested_executor = parse_code_execution_header(code_execution_header)
    except ValueError:
        raise CodeExecutionDeclarationError(CODE_EXECUTION_HEADER_INVALID_DETAIL) from None

    # The explicit gateway type always runs here.
    # A provider's keyword is only found here, and the executor decides it for each candidate.
    sandbox_tool_entry, tools_after_sandbox = extract_code_execution_tool(tools)
    provider_code_entry = first_provider_code_execution_tool(tools_after_sandbox)
    if sandbox_tool_entry is not None and not sandbox_available:
        raise CodeExecutionDeclarationError(SANDBOX_NOT_CONFIGURED_DETAIL)

    # A sandbox ID this gateway minted pins the code here, so a model change between turns keeps the sandbox.
    # An explicit header or a workspace pin still wins over it.
    names_held_sandbox = any(
        (gateway_container_value(raw) or "").startswith(CONTAINER_ID_PREFIX)
        for raw in (
            container_id,
            (sandbox_tool_entry or {}).get("container"),
            (provider_code_entry or {}).get("container"),
        )
    )
    # Only with a sandbox, so a stale ID gets the container refusal rather than the executor one.
    if names_held_sandbox and sandbox_available and requested_executor in (None, CodeExecutor.AUTO):
        requested_executor = CodeExecutor.OTARI

    preference: CodeExecutor | None = None
    code_execution_policy: ResolvedCodeExecutionPolicy | None = None
    # Without a sandbox, a provider's keyword is forwarded and no policy is read.
    if sandbox_available and (sandbox_tool_entry is not None or provider_code_entry is not None):
        code_execution_policy = await resolve_policy()
        preference, executor_conflict = resolve_code_executor_preference(
            requested=requested_executor,
            workspace=code_execution_policy.executor if code_execution_policy is not None else None,
            deployment=config.effective_code_executor(),
        )
        # A pin decides only a provider's keyword. The explicit type runs here whatever the pin says.
        if executor_conflict and provider_code_entry is not None:
            raise CodeExecutionRefusedError(CODE_EXECUTOR_PINNED_DETAIL)
    elif provider_code_entry is not None:
        if requested_executor is CodeExecutor.OTARI:
            raise CodeExecutionDeclarationError(CODE_EXECUTOR_NOT_CONFIGURED_DETAIL)
        preference = requested_executor or config.effective_code_executor()

    return DeclaredCodeExecution(
        config=config,
        container_id=container_id,
        container_registry=container_registry,
        keyword_entry=provider_code_entry,
        policy=code_execution_policy,
        preference=preference,
        sandbox_available=sandbox_available,
        sandbox_tool_entry=sandbox_tool_entry,
        tools_after_sandbox=tools_after_sandbox,
    )


class DeclaredCodeExecution:
    """The code execution one request declared, read once and admitted for each way its candidates run it.

    ``preference`` is who the request asked to run a provider's keyword, ``None``
    where it declared no code execution. Without a sandbox it is read without the
    workspace's pin, which only chooses between the sandbox and the provider.
    """

    def __init__(
        self,
        *,
        config: GatewayConfig,
        container_id: object,
        container_registry: SandboxContainerRegistry | None,
        keyword_entry: dict[str, Any] | None,
        policy: ResolvedCodeExecutionPolicy | None,
        preference: CodeExecutor | None,
        sandbox_available: bool,
        sandbox_tool_entry: dict[str, Any] | None,
        tools_after_sandbox: list[dict[str, Any]] | None,
    ) -> None:
        self.config = config
        self.container_id = container_id
        self.container_registry = container_registry
        self.keyword_entry = keyword_entry
        self.policy = policy
        self.preference = preference
        self.sandbox_available = sandbox_available
        self.sandbox_tool_entry = sandbox_tool_entry
        self.tools_after_sandbox = tools_after_sandbox
        # One claim on a held sandbox per request, however many of its candidates run the code there.
        self._leases: dict[str, ContainerLease | TenancyError] = {}

    def provider_declaration(self, *, follow_head: bool = False) -> ProviderToolDeclaration | None:
        """The provider's keyword this request declared, to decide for each candidate; ``None`` where it made none.

        Beside the explicit type, which always runs on the sandbox, every candidate is decided alike.
        """
        if self.keyword_entry is None:
            return None
        uniform = self.sandbox_tool_entry is not None
        return ProviderToolDeclaration(
            tool=_code_execution_tool.TOOL,
            entry=self.keyword_entry,
            preference=self.preference,
            backend_configured=self.sandbox_available,
            uniform=uniform,
            follow_head=follow_head and not uniform,
        )

    async def admit(self, *, claimed: bool, beside_sandbox: bool) -> AdmittedCodeExecution:
        """Admit the request's code for the candidates where Otari runs the keyword (``claimed``), or not.

        ``beside_sandbox`` is set where other candidates run the code on the sandbox,
        whose ``auto`` container a provider's attempt then leaves out rather than refuses.

        Raises:
            CodeExecutionDeclarationError: the declaration or its container cannot be served this way.
            CodeExecutionRefusedError: the workspace's policy refuses it.
            CodeExecutionContainerBusyError: the named container is serving another request.
        """
        executor: CodeExecutor | None = None
        if self.preference is not None and self.sandbox_available:
            if claimed:
                executor = CodeExecutor.OTARI
            elif self.keyword_entry is not None:
                executor = CodeExecutor.PROVIDER
            else:
                executor = decide_code_executor(self.preference, sandbox_configured=True, native_available=False)
        return await self._admit_executor(executor, beside_sandbox=beside_sandbox)

    async def _admit_executor(self, executor: CodeExecutor | None, *, beside_sandbox: bool) -> AdmittedCodeExecution:
        config = self.config
        sandbox_tool_entry = self.sandbox_tool_entry
        tools_after_sandbox = self.tools_after_sandbox
        provider_code_entry = self.keyword_entry
        code_execution_policy = self.policy
        sandbox_max_iterations: int | None = None
        sandbox_exec_timeout_s: int | None = None
        # A workspace policy below may only narrow the image and the tool kinds from the deployment's defaults.
        sandbox_session_image: str | None = config.effective_sandbox_image()
        sandbox_allowed_tools: frozenset[str] | None = None
        sandbox_container_lease: ContainerLease | None = None

        if executor is not None:
            if provider_code_entry is not None and executor is CodeExecutor.OTARI:
                # The claimed keyword keeps the caller's declaration shape, and so the result blocks it expects.
                # An explicit type beside it is folded in and adds only its hint.
                claimed, tools_after_sandbox = extract_code_execution_tool(tools_after_sandbox, intercept=True)
                assert claimed is not None  # ``provider_code_entry`` was found in the same list
                # A copy, since a provider's attempt of the same request forwards the caller's own entry.
                claimed = dict(claimed)
                if sandbox_tool_entry is not None and not claimed.get("purpose_hint"):
                    if sandbox_tool_entry.get("purpose_hint"):
                        claimed["purpose_hint"] = sandbox_tool_entry["purpose_hint"]
                sandbox_tool_entry = claimed
            elif provider_code_entry is not None and sandbox_tool_entry is not None:
                # Two sandboxes would split the caller's state across two places, so the request is refused.
                raise CodeExecutionDeclarationError(SANDBOX_PROVIDER_TOOL_CONFLICT_DETAIL)
        use_sandbox = executor is not None and sandbox_tool_entry is not None

        if use_sandbox:
            assert sandbox_tool_entry is not None
            # The policy only narrows what the deployment allows. No policy means no narrowing.
            if code_execution_policy is not None:
                if not code_execution_policy.enabled:
                    raise CodeExecutionRefusedError(SANDBOX_NOT_ENABLED_DETAIL)
                if not sandbox_tool_entry.get("purpose_hint") and code_execution_policy.default_purpose_hint:
                    sandbox_tool_entry["purpose_hint"] = code_execution_policy.default_purpose_hint
                sandbox_max_iterations = code_execution_policy.max_iterations
                sandbox_exec_timeout_s = code_execution_policy.exec_timeout_s
                if code_execution_policy.tools is not None:
                    # An empty intersection is refused, because a tool with nothing to run fails silently.
                    # It uses the same served set as the check on a policy write.
                    if not code_execution_policy.tools & set(SERVED_TOOL_NAMES):
                        raise CodeExecutionRefusedError(SANDBOX_TOOLS_EXCLUDED_DETAIL)
                    sandbox_allowed_tools = code_execution_policy.tools
                if code_execution_policy.image is not None:
                    # Re-checked because an operator may remove an image after a workspace pinned it.
                    if code_execution_policy.image not in config.pinnable_sandbox_images():
                        raise CodeExecutionRefusedError(SANDBOX_IMAGE_NOT_ALLOWED_DETAIL)
                    sandbox_session_image = code_execution_policy.image

        # A request that names no container has its sandbox released with it.
        # ``auto`` asks to hold one, and an ID asks for that one back.
        sandbox_containers = self.container_registry if use_sandbox else None
        if use_sandbox:
            container_request = requested_container(self.container_id) or requested_container(
                (sandbox_tool_entry or {}).get("container")
            )
            if container_request is None:
                # Nothing asked for, so nothing is held.
                sandbox_containers = None
            elif container_request != CONTAINER_AUTO:
                sandbox_container_lease = await self._lease(container_request)
            # ``auto`` without container reuse is not an error, and the response names no container.
        else:
            # The provider runs this code, so a container ID that only this gateway mints is refused.
            # ``auto`` can change the executor between turns, so a client can send one unchanged.
            stray = (None if beside_sandbox else gateway_container_value(self.container_id)) or (
                gateway_container_value((provider_code_entry or {}).get("container"))
            )
            if stray is not None:
                raise CodeExecutionDeclarationError(CONTAINER_NOT_GATEWAY_RUN_DETAIL)
        return AdmittedCodeExecution(
            allowed_tools=sandbox_allowed_tools,
            container_lease=sandbox_container_lease,
            containers=sandbox_containers,
            exec_timeout_s=sandbox_exec_timeout_s,
            executor=executor,
            max_iterations=sandbox_max_iterations,
            session_image=sandbox_session_image,
            tool_entry=sandbox_tool_entry,
            tools_after_sandbox=tools_after_sandbox,
            use_sandbox=use_sandbox,
        )

    async def _lease(self, container_request: str) -> ContainerLease:
        """The held sandbox ``container_request`` names, claimed once for every candidate that runs there.

        An ID names one sandbox and its files, so a deployment that cannot honor it refuses.
        It resolves against this caller's own leases before any new lease.
        """
        if container_request not in self._leases:
            gone = CodeExecutionDeclarationError(
                CONTAINER_GONE_DETAIL_TEMPLATE.format(container_id=_echoable_container_id(container_request))
            )
            outcome: ContainerLease | TenancyError = gone
            if self.container_registry is not None:
                try:
                    outcome = await self.container_registry.resolve(container_request)
                except ContainerNotFoundError:
                    outcome = gone
                except ContainerBusyError:
                    # The caller owns this ID, so a retry can succeed.
                    outcome = CodeExecutionContainerBusyError(CONTAINER_BUSY_DETAIL)
            self._leases[container_request] = outcome
        outcome = self._leases[container_request]
        if isinstance(outcome, TenancyError):
            raise outcome
        return outcome
