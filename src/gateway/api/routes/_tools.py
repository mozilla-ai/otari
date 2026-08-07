"""Helpers for extracting gateway-managed tools from a request payload.

These helpers are format-agnostic — they only look at the `type` string on
each tool entry. The same predicates and extractors are used from the
Chat-Completions, Anthropic Messages, and OpenAI Responses endpoints so
``otari_code_execution`` / ``otari_web_search`` requests get identical
handling regardless of wire shape.

The explicit ``otari_*`` tool types always trigger gateway-side execution.
Every other tool type — the short forms (``code_execution`` / ``web_search``)
and the provider-native keywords (``code_interpreter`` /
``code_execution_<date>`` / ``web_search_<date>``) — is left untouched in
``tools[]`` and forwarded to the upstream provider, which runs it server-side.
For code execution the keyword alone says who runs it: no flag, no env toggle.

Web search has one opt-in exception. A client that cannot be told to say
``otari_web_search`` (Claude Code, the Anthropic SDK, anything speaking a
provider's native vocabulary) would otherwise never reach a configured gateway
backend. Setting ``web_search_intercept`` makes the gateway also claim the
provider-named web-search keywords, so those clients work unchanged. It is off
by default because turning it on silently takes a search away from a provider
that would have run it (see ``docs/tools.md``). An OpenAI ``function`` named
``web_search`` is deliberately *not* claimed even then: that is a caller's own
tool, and hijacking it means the caller's handler never fires and it never gets
back a ``tool_call`` it can dispatch.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Any

from gateway.api.routes._schema_derive import SENSITIVE_PARAM_FIELDS
from gateway.core.env import otari_env
from gateway.log_config import logger
from gateway.services.tool_usage import ToolUsageTally
from gateway.services.web_search_backend import WEB_SEARCH_TOOL_NAME, WebSearchBackend

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig


class Tool(StrEnum):
    """Gateway-managed tool types — the only ``type`` values the gateway runs
    itself (everything else is forwarded to the upstream provider).

    Values are derived as ``otari_<member>`` so every gateway tool carries the
    ``otari_`` prefix by construction; registering a new gateway-run tool is a
    one-line addition here.
    """

    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list[Any]) -> str:
        return f"otari_{name.lower()}"

    CODE_EXECUTION = auto()  # -> "otari_code_execution"
    WEB_SEARCH = auto()  # -> "otari_web_search"


# The provider-named web-search keywords the gateway claims when interception is
# on: the bare short form, and any dated/preview variant. The prefix match keeps
# future Anthropic versions (``web_search_20991231``) and OpenAI's Responses
# spellings (``web_search_preview``) working without a release here.
_BARE_WEB_SEARCH_TYPE = "web_search"
_VERSIONED_WEB_SEARCH_PREFIX = "web_search_"


def _is_web_search_tool_type(type_value: Any) -> bool:
    """Recognise the explicit gateway-managed web_search tool type.

    Matches only ``"otari_web_search"``. Provider-named keywords
    (``"web_search"``, ``"web_search_<date>"``) are *not* matched — they pass
    through unchanged to the upstream provider, which runs the search itself.
    """
    if not isinstance(type_value, str):
        return False
    return type_value == Tool.WEB_SEARCH


def _is_provider_web_search_tool_type(type_value: Any) -> bool:
    """Recognise a provider-named web-search keyword (interception only).

    ``"web_search"`` (Claude Code's short form, OpenAI Responses' native type)
    or any ``"web_search_<suffix>"`` variant. Does not match
    ``"otari_web_search"``, which :func:`_is_web_search_tool_type` owns.
    """
    if not isinstance(type_value, str):
        return False
    return type_value == _BARE_WEB_SEARCH_TYPE or type_value.startswith(_VERSIONED_WEB_SEARCH_PREFIX)


def _is_any_web_search_tool_type(type_value: Any) -> bool:
    """The gateway-managed type or a provider-named keyword."""
    return _is_web_search_tool_type(type_value) or _is_provider_web_search_tool_type(type_value)


def declares_native_web_search(tool_entry: dict[str, Any] | None) -> bool:
    """Whether the caller declared web search in a provider's *native* vocabulary.

    True for a dated/preview keyword (``web_search_20250305``), which is what the
    Anthropic SDK, Claude Code, and Claude Desktop send and what makes them expect
    ``server_tool_use`` / ``web_search_tool_result`` blocks back so a citations
    panel has something to render. False for ``otari_web_search`` and for the bare
    ``web_search`` short form: neither implies the native response shape, so those
    callers keep receiving the plain-text result they always have.
    """
    if not tool_entry:
        return False
    type_value = tool_entry.get("type")
    return isinstance(type_value, str) and type_value.startswith(_VERSIONED_WEB_SEARCH_PREFIX)


def _is_code_execution_tool_type(type_value: Any) -> bool:
    """Recognise the explicit gateway-managed code-execution tool type.

    Matches only ``"otari_code_execution"``. Provider-named keywords
    (``"code_execution"``, ``"code_interpreter"``, ``"code_execution_<date>"``)
    are *not* matched — they pass through unchanged to the upstream provider,
    which runs the code in its own native sandbox.
    """
    if not isinstance(type_value, str):
        return False
    return type_value == Tool.CODE_EXECUTION


# Gateway-internal fields the provider SDKs (any-llm, anthropic, openai, …)
# don't accept as ``acompletion`` kwargs. Strip these from the model_dump
# before forwarding to upstream — Anthropic in particular rejects unknown
# kwargs with a hard error.
_GATEWAY_INTERNAL_FIELDS = (
    "mcp_servers",
    "mcp_server_ids",
    "guardrails",
    "tools_header",
    "max_tool_iterations",
    "session_label",
    "user",
)


def _strip_gateway_fields(
    fields: dict[str, Any],
    *,
    tools_extracted: bool = False,
    remaining_user_tools: list[dict[str, Any]] | None = None,
    web_search_declared_name: str | None = None,
) -> dict[str, Any]:
    """Strip gateway-internal fields from a ``request.model_dump(...)`` payload.

    Mutates ``fields`` in place and returns it for chaining. When the caller
    extracted any gateway-managed tool entry from ``tools`` (sandbox /
    web_search / future), pass ``tools_extracted=True`` and the remaining
    user-supplied tools; the original ``tools`` list is replaced (or popped
    entirely if none remain).

    ``web_search_declared_name`` is the ``name`` on an extracted web-search entry.
    When the caller forced that name with ``tool_choice``, the choice is retargeted
    to the backend's canonical tool name (see :func:`_retargeted_tool_choice`).

    Sensitive provider-call fields (credentials, ``provider`` selection, ...) are
    also stripped: the request schemas never derive them (see
    ``_schema_derive.SENSITIVE_PARAM_FIELDS``), but the Responses request allows
    extra fields, so a client could still smuggle one in. The gateway resolves
    these itself, and the provider-call merge spreads request fields last, so a
    client value would otherwise override the operator-controlled one.
    """
    for k in _GATEWAY_INTERNAL_FIELDS:
        fields.pop(k, None)
    for k in SENSITIVE_PARAM_FIELDS:
        fields.pop(k, None)
    if tools_extracted:
        if remaining_user_tools:
            fields["tools"] = remaining_user_tools
        else:
            fields.pop("tools", None)
    if web_search_declared_name and "tool_choice" in fields:
        fields["tool_choice"] = _retargeted_tool_choice(fields["tool_choice"], web_search_declared_name)
    return fields


def _resolve_sandbox_purpose_hint(
    sandbox_tool_entry: dict[str, Any] | None,
    config: GatewayConfig | None = None,
) -> str | None:
    """Resolve the per-tool ``purpose_hint`` for the sandbox.

    Priority: tool entry's ``purpose_hint`` → the effective config value
    (dashboard override / ``OTARI_SANDBOX_PURPOSE_HINT`` env / YAML) → ``None``
    (SandboxBackend falls back to its built-in default).
    """
    return (
        (sandbox_tool_entry.get("purpose_hint") if sandbox_tool_entry else None)
        or (config.sandbox_purpose_hint if config is not None else None)
        or otari_env("SANDBOX_PURPOSE_HINT")
        or None
    )


def _extract_first_matching_tool(
    tools: list[dict[str, Any]] | None,
    predicate: Callable[[Any], bool],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    """Pull the first tool entry whose ``type`` matches ``predicate``.

    Returns ``(entry_or_None, remaining_tools_or_None)``. The extracted entry
    is thin (no function schema); the gateway-managed backend's
    ``openai_tools`` provides the full definition during tool-use-loop
    injection. Remaining user-supplied tools pass through unchanged.
    """
    if not tools:
        return None, tools
    entry: dict[str, Any] | None = None
    remaining: list[dict[str, Any]] = []
    for t in tools:
        if entry is None and isinstance(t, dict) and predicate(t.get("type")):
            entry = t
        else:
            remaining.append(t)
    return entry, (remaining or None)


def _extract_code_execution_tool(
    tools: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    """Pull the first ``{"type": "otari_code_execution"}`` entry out of ``tools``.

    Only the explicit gateway-managed type is extracted (and run in the
    gateway sandbox). Provider-named code-execution keywords stay in
    ``tools[]`` and reach the upstream provider unchanged.
    """
    return _extract_first_matching_tool(tools, _is_code_execution_tool_type)


def _extract_web_search_tool(
    tools: list[dict[str, Any]] | None,
    *,
    intercept: bool = False,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    """Pull the first gateway-run web-search entry out of ``tools``.

    With ``intercept`` off (the default) only the explicit
    ``{"type": "otari_web_search"}`` is extracted; provider-named web_search
    keywords stay in ``tools[]`` and reach the upstream provider unchanged.

    With ``intercept`` on, the provider-named keywords (``web_search``,
    ``web_search_<date>``) are claimed too, so a client that only speaks a
    provider's vocabulary reaches the gateway's backend. An OpenAI ``function``
    named ``web_search`` is still never claimed; see the module docstring.
    """
    predicate = _is_any_web_search_tool_type if intercept else _is_web_search_tool_type
    return _extract_first_matching_tool(tools, predicate)


def _retargeted_tool_choice(tool_choice: Any, declared_name: str) -> Any:
    """Point a forced ``tool_choice`` at the gateway's canonical web-search tool.

    A caller may declare web search under its own name
    (``{"type": "web_search_20250305", "name": "search_the_web"}``) and force it
    with a matching ``tool_choice``. The declaration is replaced by the backend's
    own tool, which is named :data:`WEB_SEARCH_TOOL_NAME`, so an unrewritten
    ``tool_choice`` would name a tool the provider never received and be rejected.

    Only a choice naming ``declared_name`` is rewritten; ``auto`` / ``any`` /
    ``none`` and choices naming a different tool pass through untouched. Returns a
    new object rather than mutating the caller's.
    """
    if not isinstance(tool_choice, dict) or declared_name == WEB_SEARCH_TOOL_NAME:
        return tool_choice
    # Anthropic: {"type": "tool", "name": ...}. Responses: {"type": "function", "name": ...}.
    if tool_choice.get("name") == declared_name:
        return {**tool_choice, "name": WEB_SEARCH_TOOL_NAME}
    # Chat Completions: {"type": "function", "function": {"name": ...}}.
    function = tool_choice.get("function")
    if isinstance(function, dict) and function.get("name") == declared_name:
        return {**tool_choice, "function": {**function, "name": WEB_SEARCH_TOOL_NAME}}
    return tool_choice


def _web_search_intercept_enabled(config: GatewayConfig | None = None) -> bool:
    """Whether provider-named web-search keywords are claimed by the gateway.

    Effective config value (dashboard override / ``OTARI_WEB_SEARCH_INTERCEPT`` env /
    YAML) first, falling back to the env var so pure-env deployments work without a
    config file. Off when unset, so an upgrade never changes who runs a search.
    """
    configured = config.web_search_intercept if config is not None else None
    if configured is not None:
        return configured
    raw = otari_env("WEB_SEARCH_INTERCEPT")
    if raw is None:
        return False
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def web_search_declaration_forms(config: GatewayConfig | None = None) -> list[str]:
    """Every ``tools[].type`` this deployment routes to the web-search backend.

    Advertised by ``GET /v1/tools``. The dated form is spelled with a placeholder
    (``web_search_<date>``) because the match is a prefix, not a fixed list: any
    suffix works, including future Anthropic versions.
    """
    forms = [str(Tool.WEB_SEARCH)]
    if _web_search_intercept_enabled(config):
        forms += [_BARE_WEB_SEARCH_TYPE, f"{_VERSIONED_WEB_SEARCH_PREFIX}<date>"]
    return forms


def _resolve_web_search_purpose_hint(
    tool_entry: dict[str, Any] | None,
    config: GatewayConfig | None = None,
) -> str | None:
    """Per-tool entry → effective config (override / env / YAML) → ``None`` (backend default)."""
    return (
        (tool_entry.get("purpose_hint") if tool_entry else None)
        or (config.web_search_purpose_hint if config is not None else None)
        or otari_env("WEB_SEARCH_PURPOSE_HINT")
        or None
    )


def _build_web_search_backend(
    *,
    base_url: str,
    tool_entry: dict[str, Any],
    auth_token: str | None = None,
    config: GatewayConfig | None = None,
    tally: ToolUsageTally | None = None,
) -> WebSearchBackend:
    """Construct a WebSearchBackend honouring env-level + per-tool config.

    Per-tool entry fields (``max_results``, ``allowed_domains``,
    ``blocked_domains``, ``purpose_hint``) override env-level defaults.
    Operator-level env knobs:

      * ``OTARI_WEB_SEARCH_ENGINES`` — comma-separated SearXNG engine list
      * ``OTARI_WEB_SEARCH_MAX_RESULTS`` — default cap on returned hits
      * ``OTARI_WEB_SEARCH_EXTRACT`` — "0"/"false" to disable in-process
        content extraction (snippet-only mode).
      * ``OTARI_WEB_SEARCH_PURPOSE_HINT`` — per-deployment hint override.
    """
    kwargs: dict[str, Any] = {"base_url": base_url, "tally": tally}

    # Operator knobs resolve from the effective config value (dashboard override /
    # env / YAML) first, falling back to the env var so pure-env deployments are
    # unchanged. A dashboard override mutates ``config``, so it hot-applies here.
    engines_str = (config.web_search_engines if config is not None else None) or otari_env("WEB_SEARCH_ENGINES")
    if engines_str:
        engines = tuple(e.strip() for e in engines_str.split(",") if e.strip())
        if engines:
            kwargs["engines"] = engines

    config_max = config.web_search_max_results if config is not None else None
    if config_max is not None:
        kwargs["max_results"] = config_max
    else:
        max_env = otari_env("WEB_SEARCH_MAX_RESULTS")
        if max_env:
            try:
                parsed_max = int(max_env)
            except ValueError:
                logger.warning("OTARI_WEB_SEARCH_MAX_RESULTS=%r is not an int; ignoring", max_env)
            else:
                if parsed_max >= 1:
                    kwargs["max_results"] = parsed_max
                else:
                    logger.warning("OTARI_WEB_SEARCH_MAX_RESULTS=%r is not >= 1; ignoring", max_env)
    req_max = tool_entry.get("max_results")
    if isinstance(req_max, int) and req_max > 0:
        kwargs["max_results"] = req_max

    config_extract = config.web_search_extract if config is not None else None
    if config_extract is not None:
        kwargs["extract_content"] = config_extract
    else:
        extract_env = otari_env("WEB_SEARCH_EXTRACT")
        if extract_env is not None:
            kwargs["extract_content"] = extract_env.lower() not in {"0", "false", "no", "off"}

    allowed = tool_entry.get("allowed_domains")
    if isinstance(allowed, list) and allowed:
        kwargs["allowed_domains"] = tuple(str(d) for d in allowed)
    blocked = tool_entry.get("blocked_domains")
    if isinstance(blocked, list) and blocked:
        kwargs["blocked_domains"] = tuple(str(d) for d in blocked)

    purpose_hint = _resolve_web_search_purpose_hint(tool_entry, config)
    if purpose_hint:
        kwargs["purpose_hint"] = purpose_hint

    # Provider-specific knobs (e.g. Tavily's search_depth / topic). The gateway
    # forwards these to the search backend as-is; the adapter interprets them.
    provider_options = tool_entry.get("provider_options")
    if isinstance(provider_options, dict) and provider_options:
        kwargs["provider_options"] = provider_options

    # Forwarded to the search backend as `X-Gateway-Token` so the platform-hosted
    # backend can authenticate the gateway. Unset (and so unsent) in standalone.
    if auth_token:
        kwargs["auth_token"] = auth_token

    return WebSearchBackend(**kwargs)
