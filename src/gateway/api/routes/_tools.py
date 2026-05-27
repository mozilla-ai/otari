"""Helpers for extracting gateway-managed tools from a request payload.

These helpers are format-agnostic — they only look at the `type` string on
each tool entry. The same predicates and extractors are used from the
Chat-Completions, Anthropic Messages, and OpenAI Responses endpoints so
``code_execution`` / ``web_search`` requests get identical handling regardless
of wire shape.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from gateway.log_config import logger
from gateway.services.web_search_backend import WebSearchBackend


def _is_web_search_tool_type(type_value: Any) -> bool:
    """Recognise the tool-array shapes that map to the web_search backend.

    Accepts:
      * ``"web_search"`` — gateway-native short form (matches OpenAI's
        ``{"type": "web_search"}`` server-managed tool).
      * ``"web_search_*"`` — Anthropic versioned types (e.g.
        ``"web_search_20250305"``, future ``"web_search_20260209"``).

    Matching Anthropic by prefix means new versions keep working without a
    code change; the backend's search semantics are version-agnostic.
    """
    if not isinstance(type_value, str):
        return False
    return type_value == "web_search" or type_value.startswith("web_search_")


def _is_code_execution_tool_type(type_value: Any) -> bool:
    """Recognise the tool-array shapes Anthropic and OpenAI use for code execution.

    Accepts:
      * ``"code_execution"`` — gateway-native short form
      * ``"code_interpreter"`` — OpenAI Responses/Assistants API
      * ``"code_execution_*"`` — Anthropic versioned types
        (e.g. ``"code_execution_20250825"``)

    Matching Anthropic by prefix means new versions (``code_execution_20260101``,
    etc.) keep working without a code change. Our sandbox is a generic Python
    REPL, so we don't need to track per-version semantics.
    """
    if not isinstance(type_value, str):
        return False
    return (
        type_value == "code_execution" or type_value == "code_interpreter" or type_value.startswith("code_execution_")
    )


# Gateway-internal fields the provider SDKs (any-llm, anthropic, openai, …)
# don't accept as ``acompletion`` kwargs. Strip these from the model_dump
# before forwarding to upstream — Anthropic in particular rejects unknown
# kwargs with a hard error.
_GATEWAY_INTERNAL_FIELDS = (
    "mcp_servers",
    "mcp_server_ids",
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
    """
    for k in _GATEWAY_INTERNAL_FIELDS:
        fields.pop(k, None)
    if tools_extracted:
        if remaining_user_tools:
            fields["tools"] = remaining_user_tools
        else:
            fields.pop("tools", None)
    return fields


def _resolve_sandbox_purpose_hint(sandbox_tool_entry: dict[str, Any] | None) -> str | None:
    """Resolve the per-tool ``purpose_hint`` for the sandbox.

    Priority: tool entry's ``purpose_hint`` → ``GATEWAY_SANDBOX_PURPOSE_HINT``
    env → ``None`` (SandboxBackend falls back to its built-in default).
    """
    return (
        (sandbox_tool_entry.get("purpose_hint") if sandbox_tool_entry else None)
        or os.environ.get("GATEWAY_SANDBOX_PURPOSE_HINT")
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
    """Pull the first code-execution-style entry out of ``tools``.

    Detects the gateway-native ``{"type": "code_execution"}`` shape plus the
    provider-native equivalents from OpenAI (``code_interpreter``) and Anthropic
    (``code_execution_20250825`` and future versions). All three map to the same
    sandbox backend so swapping ``base_url`` to the gateway keeps existing
    SDK code working unchanged.
    """
    return _extract_first_matching_tool(tools, _is_code_execution_tool_type)


def _extract_web_search_tool(
    tools: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    """Pull the first web-search-style entry out of ``tools``.

    Accepts both the gateway-native ``{"type": "web_search"}`` shape (matching
    OpenAI's server-managed tool) and Anthropic's dated variants
    (``web_search_20250305`` etc.). All map to the same backend.
    """
    return _extract_first_matching_tool(tools, _is_web_search_tool_type)


def _resolve_web_search_purpose_hint(tool_entry: dict[str, Any] | None) -> str | None:
    """Per-tool entry → ``GATEWAY_WEB_SEARCH_PURPOSE_HINT`` → ``None`` (backend default)."""
    return (
        (tool_entry.get("purpose_hint") if tool_entry else None)
        or os.environ.get("GATEWAY_WEB_SEARCH_PURPOSE_HINT")
        or None
    )


def _build_web_search_backend(*, base_url: str, tool_entry: dict[str, Any]) -> WebSearchBackend:
    """Construct a WebSearchBackend honouring env-level + per-tool config.

    Per-tool entry fields (``max_results``, ``allowed_domains``,
    ``blocked_domains``, ``purpose_hint``) override env-level defaults.
    Operator-level env knobs:

      * ``GATEWAY_WEB_SEARCH_ENGINES`` — comma-separated SearXNG engine list
      * ``GATEWAY_WEB_SEARCH_MAX_RESULTS`` — default cap on returned hits
      * ``GATEWAY_WEB_SEARCH_EXTRACT`` — "0"/"false" to disable in-process
        content extraction (snippet-only mode).
      * ``GATEWAY_WEB_SEARCH_PURPOSE_HINT`` — per-deployment hint override.
    """
    kwargs: dict[str, Any] = {"base_url": base_url}

    engines_str = os.environ.get("GATEWAY_WEB_SEARCH_ENGINES")
    if engines_str:
        engines = tuple(e.strip() for e in engines_str.split(",") if e.strip())
        if engines:
            kwargs["engines"] = engines

    max_env = os.environ.get("GATEWAY_WEB_SEARCH_MAX_RESULTS")
    if max_env:
        try:
            parsed_max = int(max_env)
        except ValueError:
            logger.warning("GATEWAY_WEB_SEARCH_MAX_RESULTS=%r is not an int; ignoring", max_env)
        else:
            if parsed_max >= 1:
                kwargs["max_results"] = parsed_max
            else:
                logger.warning("GATEWAY_WEB_SEARCH_MAX_RESULTS=%r is not >= 1; ignoring", max_env)
    req_max = tool_entry.get("max_results")
    if isinstance(req_max, int) and req_max > 0:
        kwargs["max_results"] = req_max

    extract_env = os.environ.get("GATEWAY_WEB_SEARCH_EXTRACT")
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

    return WebSearchBackend(**kwargs)
