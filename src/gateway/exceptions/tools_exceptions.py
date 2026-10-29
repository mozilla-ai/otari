"""Errors that the tools domain may raise, each carrying the status it renders as."""

from enum import StrEnum

from fastapi import status

from gateway.core.error_codes import PRICING_REQUIRED
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


class WorkspaceMcpServerCheckBusyError(TenancyError):
    """No connection check could start, because every check slot on this process was taken.

    A 429 rather than a 503: it clears on its own within seconds, and a 5xx
    renders as a generic internal error, hiding the one thing the caller needs
    to know, which is to try again shortly.
    """

    status_code = status.HTTP_429_TOO_MANY_REQUESTS

    def __init__(self) -> None:
        super().__init__("Too many MCP server checks are running on this gateway; try again shortly")


class ContainerOnManagedCredentialError(TenancyValidationError):
    """A request named a container on a provider account this gateway shares across workspaces."""

    def __init__(self) -> None:
        super().__init__(
            "container cannot be used on this route: it resolves to a provider account this gateway "
            "manages on behalf of many workspaces, and a container id addresses state on that account "
            "rather than on your workspace. Use a model served by your own provider key."
        )


class McpServerDeclarationError(TenancyValidationError):
    """A request's MCP servers cannot be reached as it declared them."""


class McpServerConfigurationError(TenancyError):
    """A workspace's stored MCP servers cannot be used, which is the operator's to fix and not the caller's."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class CodeExecutionDeclarationError(TenancyValidationError):
    """A request's code execution declaration or container cannot be served as written."""


class CodeExecutionRefusedError(TenancyForbiddenError):
    """A workspace's code execution policy refuses the code execution a request declared."""


class CodeExecutionContainerBusyError(TenancyConflictError):
    """The container a request resumes is serving another request, so a retry can succeed."""


class WebToolDeclarationError(TenancyValidationError):
    """A request's managed web tool declaration cannot be served as written."""


class WebSearchInterceptedError(TenancyForbiddenError):
    """A request asked the provider to run a web search this deployment runs on its own backend."""


class WebAccessRefusedError(Exception):
    """A workspace's web search policy refuses the web access a request declared."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class WebAccessNotEnabledError(WebAccessRefusedError):
    """A workspace with web access turned off refuses a request that declares Fetch."""

    def __init__(self) -> None:
        super().__init__("web access is not enabled for this workspace")


class WebAccessToolNotAuthorizedError(WebAccessRefusedError):
    """A request declared a web tool its workspace's policy does not authorize."""

    def __init__(self) -> None:
        super().__init__("A requested managed web tool is not authorized for this workspace")


class WebSearchNotEnabledError(WebAccessRefusedError):
    """A workspace with web access turned off refuses a request that declares Search alone."""

    def __init__(self) -> None:
        super().__init__("web search is not enabled for this workspace")


class WebSearchPolicyResolutionFailure(StrEnum):
    """Why a workspace's web search policy could not be resolved.

    A member carries the message a caller sees.
    Its value names the cause, which a caller never sees.
    """

    def __new__(cls, cause: str, message: str) -> "WebSearchPolicyResolutionFailure":
        member = str.__new__(cls, cause)
        member._value_ = cause
        member.message = message
        return member

    message: str

    STORED_POLICY_INVALID = (
        "a stored domain rule is invalid",
        "Web search configuration contains an invalid domain rule",
    )


class WebSearchPolicyResolutionFailedError(Exception):
    """A workspace's web search policy could not be resolved."""

    def __init__(self, reason: WebSearchPolicyResolutionFailure) -> None:
        super().__init__(reason.message)
        self.message = reason.message
        self.reason = reason


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


class SandboxCredentialRefusedError(TenancyValidationError):
    """A sandbox key write this deployment cannot take: a blank key, or a provider that leases with none."""


class McpSessionsInterruptedError(Exception):
    """A request's MCP sessions were canceled while they opened, by something other than the request."""

    def __init__(self) -> None:
        super().__init__("MCP sessions were canceled before they opened")


class WebSearchKeyNotFoundError(TenancyNotFoundError):
    """No web search or fetch key with this id in the caller's organization."""

    def __init__(self, key_id: object):
        super().__init__(f"Web search key {key_id} not found")


class WebSearchKeyUnknownProviderError(TenancyValidationError):
    """A key named a provider its kind's tool cannot call."""

    def __init__(self, provider: str, known: tuple[str, ...], *, kind: str = "search"):
        super().__init__(f"Unknown web {kind} provider '{provider}'. Use one of: {', '.join(known)}")


class WebSearchKeyIncompleteError(TenancyValidationError):
    """A key lacks what its provider needs, which a call would otherwise take from the server's own settings."""

    def __init__(self, provider: str, missing: str):
        super().__init__(f"A key on {provider} needs {missing}")


class WebSearchKeySettingRefusedError(TenancyValidationError):
    """A key's ``api_base`` or ``options`` its provider, or this deployment, does not take."""

    def __init__(self, field: str, problem: str):
        super().__init__(f"{field}: {problem}")


class WebSearchKeyCredentialRequiredError(TenancyValidationError):
    """A new ``api_base`` on a key that holds a credential, sent without the credential.

    The next call would send the stored credential to the new host, and an
    organization admin only ever reads a key's ``last4``.
    """

    def __init__(self) -> None:
        super().__init__(
            "Changing a key's api_base sends its credentials to the new host: "
            "send api_key, and any hidden options credential, again in the same request"
        )


class WebSearchKeyNameRequiredError(TenancyValidationError):
    def __init__(self) -> None:
        super().__init__("A web search key needs a name")


class WebSearchKeyMalformedError(TenancyValidationError):
    def __init__(self) -> None:
        super().__init__("A web search key cannot contain whitespace or control characters")


class WebSearchKeyAlreadyExistsError(TenancyConflictError):
    def __init__(self, provider: str, name: str, *, kind: str = "search"):
        super().__init__(f"The organization already has a {provider} web {kind} key named '{name}'")


class WebSearchKeyArchivedError(TenancyValidationError):
    """An archived key cannot be changed, made a default or pinned until it is restored."""

    def __init__(self, key_id: object):
        super().__init__(f"Web search key {key_id} is archived; restore it first")


class WebSearchKeyNotArchivedError(TenancyValidationError):
    """A key in use is archived before it is deleted, so a deletion is never a surprise."""

    def __init__(self, key_id: object):
        super().__init__(f"Web search key {key_id} is not archived; archive it before deleting it")


class WebSearchKeyDefaultConflictError(TenancyConflictError):
    """Another key of the same kind became the organization's default at the same time."""

    def __init__(self, kind: str):
        super().__init__(f"Another web {kind} key became the default at the same time; retry")


class WorkspaceWebSearchKeyOverrideConflictError(TenancyValidationError):
    def __init__(self) -> None:
        super().__init__("A web search key cannot be both pinned and turned off for a workspace")


class SearchToolRefusedError(TenancyValidationError):
    """A search or fetch instance the instance rules refuse to write or test.

    A 422, the status the same route's request validation answers with.
    """

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT


class SearchToolNotFoundError(TenancyNotFoundError):
    """No configured or stored search or fetch instance has this name."""

    def __init__(self, name: str):
        super().__init__(f"No search or fetch instance '{name}'.")


class StoredSearchToolNotFoundError(TenancyNotFoundError):
    """No stored search or fetch instance has this name.

    The message says when the name is a config-file instance or the built-in
    fetcher, which the API cannot change.
    """


class SearchToolAlreadyStoredError(TenancyConflictError):
    """A create named a stored instance, or collided with a write another create committed first."""


class SearchToolChangedError(TenancyError):
    """An update expected a version of the stored instance other than the one stored."""

    status_code = status.HTTP_412_PRECONDITION_FAILED

    def __init__(self) -> None:
        super().__init__("This search tool was modified since you loaded it; reload and retry.")


class SearchToolKeyUnstorableError(TenancyValidationError):
    """An API key cannot be stored or re-encrypted, because ``OTARI_SECRET_KEY`` is missing or invalid.

    Its message is the secret box's own, which carries no key material.
    """


class SearchToolMisconfiguredError(TenancyValidationError):
    """A search instance nothing can run as it stands, such as a ``searxng`` one with no address to search."""

    def __init__(self, name: str, problem: str | None = None):
        super().__init__(problem or f"Search tool '{name}' is not configured correctly.")


class SearchToolSelectionError(TenancyValidationError):
    """A direct search that names no search instance it can run on (contract C6).

    No instance exists, a name matches none, or no name is given while several
    exist and no default names one.
    """


class DirectSearchFieldRefusedError(TenancyValidationError):
    """A direct search asking ``include_raw`` or ``provider_options`` of a call that does not run on any-search.

    A ``searxng`` instance until its adapter, and a Tavily or Brave organization
    key until theirs.
    """

    def __init__(self, provider: str) -> None:
        super().__init__(
            f"include_raw and provider_options are not available yet for a {provider} search: leave them unset."
        )


class DirectSearchFailedError(TenancyError):
    """The provider failed a direct search, or could not be reached.

    ``message`` is the provider's own account, for the usage row and the log
    only: an old client's can quote the upstream body. ``headers`` carry the
    failure's safe fields for the caller (decision 67), and ``upstream_status``
    is the provider's HTTP status when it gave one.
    """

    status_code = status.HTTP_502_BAD_GATEWAY

    def __init__(self, message: str, *, headers: dict[str, str], upstream_status: int | None) -> None:
        super().__init__(message)
        self.headers = headers
        self.upstream_status = upstream_status


class DirectSearchPricingRequiredError(TenancyError):
    """A direct search refused because ``require_pricing`` is on and no pricing row prices it.

    ``message`` names the row to add; the route returns it to the caller and
    writes it on the usage row.
    """

    status_code = status.HTTP_402_PAYMENT_REQUIRED
    error_code: str = PRICING_REQUIRED


class GatewayToolPricingRequiredError(TenancyError):
    """A request refused because ``require_pricing`` is on and no pricing row prices a gateway-run tool it uses.

    ``message`` names the tool and the row to add.
    """

    status_code = status.HTTP_402_PAYMENT_REQUIRED
    error_code: str = PRICING_REQUIRED


class WebSearchOptionRefusedError(TenancyValidationError):
    """A request's ``provider_options`` key the provider serving it does not take from a request.

    The provider's schema does not know the key or refuses its value, or marks
    the key operator-only. A 400 on either door (design G).
    """

    def __init__(self, key: str, problem: str):
        super().__init__(f"provider_options: {problem}.")
        self.key = key


class WebSearchCredentialRefusedError(TenancyError):
    """A workspace's search or fetch credential its provider cannot be called with as it is.

    One without a key or base URL its provider needs, which the library or the
    old code would fill from the server's own settings, so the deployment's key
    or endpoint would serve the tenant; or one naming a base URL the code that
    serves its provider cannot use, which would send the tenant's key to the
    provider's public endpoint instead. A 502: the fault is in a stored
    credential, not in the request.
    """

    status_code = status.HTTP_502_BAD_GATEWAY

    def __init__(self, provider: str, problem: str) -> None:
        super().__init__(f"The workspace's {provider} credential {problem}.")
        self.provider = provider


__all__ = [
    "DirectSearchFailedError",
    "DirectSearchFieldRefusedError",
    "McpSessionsInterruptedError",
    "SandboxCredentialRefusedError",
    "SandboxImageNotAllowedError",
    "SandboxToolsUnrunnableError",
    "SearchToolAlreadyStoredError",
    "SearchToolChangedError",
    "SearchToolKeyUnstorableError",
    "SearchToolMisconfiguredError",
    "SearchToolNotFoundError",
    "SearchToolRefusedError",
    "SearchToolSelectionError",
    "StoredSearchToolNotFoundError",
    "WebAccessNotEnabledError",
    "WebAccessRefusedError",
    "WebAccessToolNotAuthorizedError",
    "WebSearchCredentialRefusedError",
    "WebSearchNotEnabledError",
    "WebSearchOptionRefusedError",
    "WebSearchPolicyResolutionFailedError",
    "WebSearchPolicyResolutionFailure",
    "WorkspaceMcpServerAlreadyExistsError",
    "WorkspaceMcpServerLimitReachedError",
    "WorkspaceMcpServerNotFoundError",
    "WorkspaceMcpServerUnsafeUrlError",
    "WorkspaceWebSearchDomainsExcludedError",
]
