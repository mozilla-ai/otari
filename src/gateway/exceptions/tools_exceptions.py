"""Errors that the tools domain may raise, each carrying the status it renders as."""

from enum import StrEnum

from fastapi import status

from gateway.exceptions import (
    TenancyConflictError,
    TenancyError,
    TenancyForbiddenError,
    TenancyNotFoundError,
    TenancyValidationError,
)


class WorkspaceMcpServerNotFoundError(TenancyNotFoundError):
    def __init__(self, mcp_server_id: object):
        super().__init__(f"MCP server {mcp_server_id} not found")


class WorkspaceMcpServerAlreadyExistsError(TenancyConflictError):
    """A workspace already has an MCP server under this name.

    Refused rather than collapsed onto the existing row: the name is what the
    tool loop labels a server's tools with, so silently reusing it would point
    a caller's request at a different endpoint than the one they just
    configured.
    """

    def __init__(self, workspace_id: object, name: object):
        super().__init__(f"Workspace {workspace_id} already has an MCP server named '{name}'")


class WorkspaceMcpServerUnsafeUrlError(TenancyValidationError):
    """The URL failed the same SSRF and TLS checks a request-body MCP server faces.

    Carries the reason from `services.url_safety.UnsafeURLError` verbatim: it
    names the host and the range it resolved into, which is what an operator
    needs to fix the entry, and it is the operator's own URL either way (this
    surface is management-gated, not a caller-supplied endpoint).
    """

    def __init__(self, reason: str):
        super().__init__(reason)


class WorkspaceMcpServerLimitReachedError(TenancyValidationError):
    """The workspace already holds as many MCP servers as it may.

    A resolved request opens a session to every server it names, so the cap
    bounds the fan-out one workspace can ask a gateway process for.
    """

    def __init__(self, workspace_id: object, limit: int):
        super().__init__(f"Workspace {workspace_id} already has the maximum of {limit} MCP servers")


class WorkspaceWebSearchDomainsExcludedError(TenancyForbiddenError):
    """A request's search allow-list shares no domain with its workspace's.

    The two lists are intersected rather than overridden, so this is the empty
    intersection: every domain the request asked for is one the workspace does
    not permit. Refused rather than run, because an empty effective allow-list
    is read by ``_build_web_retrieval_backend`` as *no* allow-list (an empty list
    is falsy), which would turn the narrowest possible policy into no policy at
    all.
    """

    def __init__(self) -> None:
        super().__init__("The requested search domains are not permitted for this workspace")


class SandboxToolsUnrunnableError(TenancyValidationError):
    """A code-execution policy's tool list names nothing this deployment serves.

    The list intersects what the sandbox backend offers, so this one resolves to
    an empty set and every request would answer 403. Refused at the write rather
    than stored, because a policy that reads as a refinement and behaves as a
    refusal is the failure the surface exists to prevent, and the operator's only
    signal would be users reporting 403s days later.
    """

    def __init__(self, served: tuple[str, ...]):
        super().__init__(
            "A code-execution tool list must name at least one tool this deployment serves "
            f"({', '.join(served)}). Use enabled=false to refuse the workspace instead."
        )


class SandboxImageNotAllowedError(TenancyValidationError):
    """A workspace code-execution policy named a sandbox image the operator has not curated.

    A 400 rather than a 403: the caller has the role to set the policy, and the
    value they sent is the thing being refused. The message names the allowed
    set, which is not a disclosure worth withholding, because that set is
    already reported on the policy itself so the dashboard can offer it.
    """


class McpResolutionFailure(StrEnum):
    """Why an MCP server could not be resolved.

    The message a caller sees is fixed, so a member is the only thing that tells one cause from another.
    No member names any part of the peer's answer.
    """

    ANSWER_NOT_AN_OBJECT = "the answer is not an object"
    ENTRY_UNREADABLE = "an entry could not be read"
    ID_MISMATCH = "an entry names a different server"
    NO_CALLER_CREDENTIAL = "the request carried no caller credential"
    NO_SERVER_LIST = "the answer carries no list of servers"
    NO_WORKSPACE = "the request named no workspace"
    SEVERAL_ENTRIES = "several entries answered a request for one"


class McpServerResolutionFailedError(TenancyError):
    """An MCP server could not be resolved.

    The message is fixed because an error boundary may return it.
    The underlying detail quotes the stored server URL or its credential.
    The cause travels as ``reason``, which a boundary logs and never returns.
    """

    status_code = status.HTTP_502_BAD_GATEWAY

    def __init__(self, reason: McpResolutionFailure) -> None:
        super().__init__("MCP server resolution failed")
        self.reason = reason


__all__ = [
    "McpResolutionFailure",
    "McpServerResolutionFailedError",
    "SandboxImageNotAllowedError",
    "SandboxToolsUnrunnableError",
    "WorkspaceMcpServerAlreadyExistsError",
    "WorkspaceMcpServerLimitReachedError",
    "WorkspaceMcpServerNotFoundError",
    "WorkspaceMcpServerUnsafeUrlError",
    "WorkspaceWebSearchDomainsExcludedError",
]
