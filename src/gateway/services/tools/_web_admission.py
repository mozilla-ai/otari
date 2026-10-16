"""Admit the managed web tools one request declared, before any search runs.

:func:`read_web_declarations` reads who the request asked to run a provider's own
search or fetch keyword, which ``decide_candidate_tools`` then answers per candidate.
For the candidates that share one answer, the pipeline calls the rest in order:
:func:`extract_web_tools` takes the managed web tools out of the request, refusing an
ambiguous declaration, and :func:`admit_web_access` applies the workspace's web search
policy, read once for the whole request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gateway.exceptions.tools_exceptions import WebSearchInterceptedError, WebToolDeclarationError
from gateway.models.tools import CodeExecutor, WebTool
from gateway.services.tools import _web_fetch_tool, _web_search_tool
from gateway.services.tools._candidate_tools import ProviderToolDeclaration
from gateway.services.tools._web_access import WebAccessGrant, apply_web_access_policy
from gateway.services.tools._web_declarations import (
    WEB_FETCH_HEADER,
    WEB_SEARCH_HEADER,
    extract_web_fetch_tool,
    extract_web_search_tool,
    first_provider_web_fetch_tool,
    first_provider_web_search_tool,
    is_provider_web_fetch_tool_type,
    is_provider_web_search_tool_type,
    parse_web_fetch_header,
    parse_web_search_header,
    web_search_header_conflicts,
    web_search_intercept_enabled,
)
from gateway.services.tools._web_names import WEB_FETCH_TOOL_NAME, WEB_SEARCH_TOOL_NAME
from gateway.services.web_retrieval_policy import MAX_WEB_SEARCH_DOMAINS, read_domain_list

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig
    from gateway.models.tools import ResolvedWebSearchConfig

WEB_SEARCH_HEADER_INVALID_DETAIL = f"{WEB_SEARCH_HEADER} must be one of auto, otari, provider"
WEB_FETCH_HEADER_INVALID_DETAIL = f"{WEB_FETCH_HEADER} must be one of auto, otari, provider"
WEB_SEARCH_INTERCEPTED_DETAIL = (
    f"this deployment runs every web search on its own backend; the {WEB_SEARCH_HEADER} "
    "header cannot hand it to the provider"
)
WEB_SEARCH_NOT_CONFIGURED_DETAIL = (
    "otari_web_search tool requested but no search backend is configured on this gateway. "
    "Set OTARI_WEB_SEARCH_URL on the gateway, or remove otari_web_search from `tools`."
)
WEB_SEARCH_MAX_USES_INVALID_DETAIL = "web_search max_uses must be a non-negative integer"
WEB_FETCH_MAX_USES_INVALID_DETAIL = "web_fetch max_uses must be a non-negative integer"
WEB_FETCH_NOT_ENABLED_DETAIL = (
    "otari_web_fetch tool requested but web fetch is disabled on this gateway. "
    "Set OTARI_WEB_FETCH_ENABLED=true on the gateway, or remove otari_web_fetch from `tools`."
)
WEB_FETCH_DECLARATION_INVALID_DETAIL = "otari_web_fetch declarations may contain only the type field"
WEB_SEARCH_DECLARATION_INVALID_DETAIL = "otari_web_search declarations contain an unsupported field"
WEB_TOOL_DUPLICATE_DETAIL = "A managed web tool may be declared at most once"
WEB_TOOL_RESERVED_NAME_DETAIL = "A caller-defined function uses a reserved managed web-tool name"
WEB_SEARCH_REQUEST_DOMAIN_INVALID_DETAIL = (
    "Web search allowed_domains and blocked_domains must each contain at most "
    f"{MAX_WEB_SEARCH_DOMAINS} bare valid hostnames"
)
WEB_FETCH_REQUEST_DOMAIN_INVALID_DETAIL = (
    "Web fetch allowed_domains and blocked_domains must each contain at most "
    f"{MAX_WEB_SEARCH_DOMAINS} bare valid hostnames"
)


def _read_max_uses(entry: dict[str, Any] | None, invalid_detail: str) -> int | None:
    if entry is None or entry.get("max_uses") is None:
        return None
    value = entry["max_uses"]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(invalid_detail)
    return value


def read_web_search_max_uses(entry: dict[str, Any] | None) -> int | None:
    """A Search declaration's ``max_uses``, or ``None`` where it names none.

    Raises ``ValueError`` for a value that is not a non-negative integer.
    """
    return _read_max_uses(entry, WEB_SEARCH_MAX_USES_INVALID_DETAIL)


def read_web_fetch_max_uses(entry: dict[str, Any] | None) -> int | None:
    """A Fetch declaration's ``max_uses``, read as :func:`read_web_search_max_uses` reads a Search one."""
    return _read_max_uses(entry, WEB_FETCH_MAX_USES_INVALID_DETAIL)


def _canonicalize_request_domains(tool_entry: dict[str, Any]) -> None:
    """Canonicalize a declaration's own domain rules in place, by the rules a workspace's policy is read with.

    Raises:
        ValueError: a list ``read_domain_list`` refuses.
    """
    for field in ("allowed_domains", "blocked_domains"):
        values = tool_entry.get(field)
        if values is not None:
            tool_entry[field] = list(read_domain_list(values) or ())


_WEB_SEARCH_DECLARATION_FIELDS = frozenset(
    {
        "type",
        "max_uses",
        "max_results",
        "allowed_domains",
        "blocked_domains",
        "purpose_hint",
        "provider_options",
    }
)


def _function_tool_name(entry: dict[str, Any]) -> str | None:
    function = entry.get("function")
    if entry.get("type") == "function":
        name = function.get("name") if isinstance(function, dict) else entry.get("name")
    elif isinstance(entry.get("input_schema"), dict):
        # Anthropic function tools are flat and carry no ``type=function``.
        name = entry.get("name")
    else:
        return None
    return name if isinstance(name, str) else None


@dataclass(frozen=True)
class WebClaims:
    """Which provider-named web keywords the gateway runs itself rather than forwarding."""

    search: bool = False
    fetch: bool = False


def validate_web_declarations(tools: list[dict[str, Any]] | None, *, claims: WebClaims) -> None:
    """Refuse a managed web declaration that is ambiguous once ``claims`` are taken, before any policy or network I/O.

    Raises:
        WebToolDeclarationError: a tool is declared twice, malformed, or named after a managed one.
    """
    entries = [entry for entry in tools or [] if isinstance(entry, dict)]
    search_count = sum(
        entry.get("type") == "otari_web_search"
        or (claims.search and is_provider_web_search_tool_type(entry.get("type")))
        for entry in entries
    )
    fetch_count = sum(
        entry.get("type") == "otari_web_fetch" or (claims.fetch and is_provider_web_fetch_tool_type(entry.get("type")))
        for entry in entries
    )
    if search_count > 1 or fetch_count > 1:
        raise WebToolDeclarationError(WEB_TOOL_DUPLICATE_DETAIL)
    for entry in entries:
        if entry.get("type") == "otari_web_fetch" and set(entry) != {"type"}:
            raise WebToolDeclarationError(WEB_FETCH_DECLARATION_INVALID_DETAIL)
        if entry.get("type") == "otari_web_search" and not set(entry) <= _WEB_SEARCH_DECLARATION_FIELDS:
            raise WebToolDeclarationError(WEB_SEARCH_DECLARATION_INVALID_DETAIL)
    managed_names = {
        name for name, count in ((WEB_SEARCH_TOOL_NAME, search_count), (WEB_FETCH_TOOL_NAME, fetch_count)) if count
    }
    if any(_function_tool_name(entry) in managed_names for entry in entries):
        raise WebToolDeclarationError(WEB_TOOL_RESERVED_NAME_DETAIL)


def read_web_declarations(
    tools: list[dict[str, Any]] | None,
    *,
    web_search_header: str | None,
    web_fetch_header: str | None = None,
    config: GatewayConfig,
    uniform: bool,
) -> list[ProviderToolDeclaration]:
    """The provider web keywords one request declared, and who it asked to run each.

    Only with a backend to run it on does Otari claim a keyword. Interception claims
    every search keyword, and the request's :data:`WEB_SEARCH_HEADER` cannot take that
    back; without it the header decides, and without either the keyword is forwarded.
    ``uniform`` decides every candidate alike, Otari's for all where any one needs it.

    Raises:
        WebToolDeclarationError: a header cannot be read.
        WebSearchInterceptedError: the header hands the provider a search this deployment runs itself.
    """
    try:
        requested_web_search = parse_web_search_header(web_search_header)
    except ValueError:
        raise WebToolDeclarationError(WEB_SEARCH_HEADER_INVALID_DETAIL) from None
    try:
        requested_web_fetch = parse_web_fetch_header(web_fetch_header)
    except ValueError:
        raise WebToolDeclarationError(WEB_FETCH_HEADER_INVALID_DETAIL) from None
    provider_search_entry = first_provider_web_search_tool(tools)
    intercept_web_search = web_search_intercept_enabled(config)
    backend_configured = config.web_search_configured()
    if (
        provider_search_entry is not None
        and backend_configured
        and web_search_header_conflicts(requested_web_search, intercept=intercept_web_search)
    ):
        raise WebSearchInterceptedError(WEB_SEARCH_INTERCEPTED_DETAIL)
    declared = (
        (
            _web_search_tool.TOOL,
            provider_search_entry,
            CodeExecutor.OTARI if intercept_web_search else requested_web_search,
            backend_configured,
        ),
        (_web_fetch_tool.TOOL, first_provider_web_fetch_tool(tools), requested_web_fetch, config.web_fetch_enabled),
    )
    return [
        ProviderToolDeclaration(
            tool=tool, entry=entry, preference=preference, backend_configured=configured, uniform=uniform
        )
        for tool, entry, preference, configured in declared
        if entry is not None
    ]


@dataclass(frozen=True)
class DeclaredWebTools:
    """The managed web tools one request declared, and the tools it declared besides them."""

    fetch_tool_entry: dict[str, Any] | None
    remaining_user_tools: list[dict[str, Any]] | None
    search_tool_entry: dict[str, Any] | None

    @property
    def declared_any(self) -> bool:
        return self.search_tool_entry is not None or self.fetch_tool_entry is not None

    @property
    def requested_tools(self) -> list[WebTool]:
        """The managed web tools declared, Search before Fetch."""
        return [
            tool
            for tool, entry in ((WebTool.SEARCH, self.search_tool_entry), (WebTool.FETCH, self.fetch_tool_entry))
            if entry is not None
        ]


def extract_web_tools(
    tools: list[dict[str, Any]] | None,
    *,
    config: GatewayConfig,
    claims: WebClaims,
) -> DeclaredWebTools:
    """Take the managed web tools out of ``tools``, refusing one this deployment cannot serve.

    ``claims`` names the provider keywords Otari runs. Each one taken is a copy, since
    another candidate of the same request may forward the caller's own entry.

    Raises:
        WebToolDeclarationError: a declared web tool is ambiguous, malformed or not served here.
    """
    validate_web_declarations(tools, claims=claims)
    search_tool_entry, tools_after_search = extract_web_search_tool(tools, intercept=claims.search)
    if search_tool_entry is not None:
        search_tool_entry = dict(search_tool_entry)
    try:
        read_web_search_max_uses(search_tool_entry)
    except ValueError as exc:
        raise WebToolDeclarationError(WEB_SEARCH_MAX_USES_INVALID_DETAIL) from exc
    fetch_tool_entry, remaining_user_tools = extract_web_fetch_tool(tools_after_search, claim=claims.fetch)
    if fetch_tool_entry is not None:
        fetch_tool_entry = dict(fetch_tool_entry)
    try:
        read_web_fetch_max_uses(fetch_tool_entry)
    except ValueError as exc:
        raise WebToolDeclarationError(WEB_FETCH_MAX_USES_INVALID_DETAIL) from exc
    if fetch_tool_entry is not None:
        if not config.web_fetch_enabled:
            raise WebToolDeclarationError(WEB_FETCH_NOT_ENABLED_DETAIL)
        try:
            _canonicalize_request_domains(fetch_tool_entry)
        except ValueError as exc:
            raise WebToolDeclarationError(WEB_FETCH_REQUEST_DOMAIN_INVALID_DETAIL) from exc
    if search_tool_entry is not None:
        if not config.web_search_configured():
            raise WebToolDeclarationError(WEB_SEARCH_NOT_CONFIGURED_DETAIL)
        try:
            _canonicalize_request_domains(search_tool_entry)
        except ValueError as exc:
            raise WebToolDeclarationError(WEB_SEARCH_REQUEST_DOMAIN_INVALID_DETAIL) from exc
    return DeclaredWebTools(
        fetch_tool_entry=fetch_tool_entry,
        remaining_user_tools=remaining_user_tools,
        search_tool_entry=search_tool_entry,
    )


def admit_web_access(
    web: DeclaredWebTools,
    *,
    policy: ResolvedWebSearchConfig | None,
    config: GatewayConfig,
) -> WebAccessGrant:
    """Narrow the declared web tools to what the workspace's web search ``policy`` permits.

    Raises:
        WebAccessRefusedError: the policy refuses the request, and the subclass says why.
        WorkspaceWebSearchDomainsExcludedError: the request's domains share nothing with the workspace's.
    """
    return apply_web_access_policy(
        policy,
        requested_tools=web.requested_tools,
        search_tool_entry=web.search_tool_entry,
        fetch_tool_entry=web.fetch_tool_entry,
        config=config,
    )
