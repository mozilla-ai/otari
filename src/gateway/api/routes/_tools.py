"""Helpers for extracting gateway-managed tools from a request payload.

These helpers are format-agnostic — they only look at the `type` string on
each tool entry. The same predicates and extractors are used from the
Chat-Completions, Anthropic Messages, and OpenAI Responses endpoints so
``otari_code_execution`` / ``otari_web_search`` requests get identical
handling regardless of wire shape.

Only the explicit ``otari_*`` tool types trigger gateway-side execution.
Every other tool type — the legacy gateway short forms (``code_execution`` /
``web_search``) and the provider-native keywords (``code_interpreter`` /
``code_execution_<date>`` / ``web_search_<date>``) — is left untouched in
``tools[]`` and forwarded to the upstream provider, which runs it server-side.
The keyword alone says who runs the code — no flag, no env toggle.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum, auto
from typing import Any

from gateway.api.routes._schema_derive import SENSITIVE_PARAM_FIELDS
from gateway.core.env import otari_env
from gateway.log_config import logger
from gateway.services.web_search_backend import WebSearchBackend


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


def _is_web_search_tool_type(type_value: Any) -> bool:
    """Recognise the explicit gateway-managed web_search tool type.

    Matches only ``"otari_web_search"``. Provider-named keywords
    (``"web_search"``, ``"web_search_<date>"``) are *not* matched — they pass
    through unchanged to the upstream provider, which runs the search itself.
    """
    if not isinstance(type_value, str):
        return False
    return type_value == Tool.WEB_SEARCH


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
    "user",
)


def _strip_gateway_fields(
    fields: dict[str, Any],
    *,
    tools_extracted: bool = False,
    remaining_user_tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Strip gateway-internal fields from a ``request.model_dump(...)`` payload.

    Mutates ``fields`` in place and returns it for chaining. When the caller
    extracted any gateway-managed tool entry from ``tools`` (sandbox /
    web_search / future), pass ``tools_extracted=True`` and the remaining
    user-supplied tools; the original ``tools`` list is replaced (or popped
    entirely if none remain).

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
    return fields


def _resolve_sandbox_purpose_hint(sandbox_tool_entry: dict[str, Any] | None) -> str | None:
    """Resolve the per-tool ``purpose_hint`` for the sandbox.

    Priority: tool entry's ``purpose_hint`` → ``OTARI_SANDBOX_PURPOSE_HINT``
    env → ``None`` (SandboxBackend falls back to its built-in default).
    """
    return (
        (sandbox_tool_entry.get("purpose_hint") if sandbox_tool_entry else None)
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
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    """Pull the first ``{"type": "otari_web_search"}`` entry out of ``tools``.

    Only the explicit gateway-managed type is extracted (and run against the
    gateway's web_search backend). Provider-named web_search keywords stay in
    ``tools[]`` and reach the upstream provider unchanged.
    """
    return _extract_first_matching_tool(tools, _is_web_search_tool_type)


def _resolve_web_search_purpose_hint(tool_entry: dict[str, Any] | None) -> str | None:
    """Per-tool entry → ``OTARI_WEB_SEARCH_PURPOSE_HINT`` → ``None`` (backend default)."""
    return (tool_entry.get("purpose_hint") if tool_entry else None) or otari_env("WEB_SEARCH_PURPOSE_HINT") or None


def _build_web_search_backend(
    *, base_url: str, tool_entry: dict[str, Any], auth_token: str | None = None
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
    kwargs: dict[str, Any] = {"base_url": base_url}

    engines_str = otari_env("WEB_SEARCH_ENGINES")
    if engines_str:
        engines = tuple(e.strip() for e in engines_str.split(",") if e.strip())
        if engines:
            kwargs["engines"] = engines

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

    extract_env = otari_env("WEB_SEARCH_EXTRACT")
    if extract_env is not None:
        kwargs["extract_content"] = extract_env.lower() not in {"0", "false", "no", "off"}

    allowed = tool_entry.get("allowed_domains")
    if isinstance(allowed, list) and allowed:
        kwargs["allowed_domains"] = tuple(str(d) for d in allowed)
    blocked = tool_entry.get("blocked_domains")
    if isinstance(blocked, list) and blocked:
        kwargs["blocked_domains"] = tuple(str(d) for d in blocked)

    purpose_hint = _resolve_web_search_purpose_hint(tool_entry)
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
