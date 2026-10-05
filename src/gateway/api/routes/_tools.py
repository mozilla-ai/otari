"""Route-side helpers for the gateway-managed tools of a request.

Stripping gateway-only fields, resolving purpose hints, retargeting ``tool_choice``
and building the web retrieval backend. They are shared by the Chat-Completions,
Anthropic Messages, and OpenAI Responses endpoints, so ``otari_code_execution`` /
``otari_web_search`` requests get identical handling regardless of wire shape.

The explicit ``otari_*`` tool types always trigger gateway-side execution.
Which web-search keywords the gateway claims is decided in ``services/tools/_web_declarations.py``,
and which code-execution keywords in ``services/tools/_code_execution_declarations.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gateway.api.routes._schema_derive import SENSITIVE_PARAM_FIELDS
from gateway.core.config import parse_bool_env
from gateway.core.env import otari_env
from gateway.services.tool_usage import ToolUsageTally
from gateway.services.tools import web_search_max_results_baseline
from gateway.services.web_retrieval_backend import (
    WEB_SEARCH_TOOL_NAME,
    WebRetrievalBackend,
    WebRetrievalCounter,
)
from gateway.services.web_retrieval_policy import DomainPolicy

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig

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


def _build_web_retrieval_backend(
    *,
    base_url: str | None,
    search_tool_entry: dict[str, Any] | None,
    fetch_tool_entry: dict[str, Any] | None = None,
    fetch_policy: DomainPolicy | None = None,
    counter: WebRetrievalCounter | None = None,
    auth_token: str | None = None,
    config: GatewayConfig | None = None,
    tally: ToolUsageTally | None = None,
) -> WebRetrievalBackend:
    """Construct a WebRetrievalBackend honoring env-level + per-tool config.

    Per-tool entry fields (``max_results``, ``allowed_domains``,
    ``blocked_domains``, ``purpose_hint``) override env-level defaults.
    Operator-level env knobs:

      * ``OTARI_WEB_SEARCH_ENGINES`` — comma-separated SearXNG engine list
      * ``OTARI_WEB_SEARCH_MAX_RESULTS`` — default cap on returned hits
      * ``OTARI_WEB_SEARCH_EXTRACT``: "0"/"false" disables local result-page
        extraction (snippet-only mode).
      * ``OTARI_WEB_SEARCH_PURPOSE_HINT`` — per-deployment hint override.

    ``base_url`` may be ``None`` when the deployment configured a licensed
    search provider instead, which the backend then calls directly.
    """
    kwargs: dict[str, Any] = {
        "base_url": base_url,
        "tally": tally,
        "trust_env_proxy": (
            config.web_retrieval_trust_env_proxy
            if config is not None
            else parse_bool_env(otari_env("WEB_RETRIEVAL_TRUST_ENV_PROXY", "false"))
        ),
    }

    # A licensed provider this deployment holds the key for wins over the URL,
    # and is how a deployment searches with no backend service in front of it.
    if config is not None and config.web_search_provider_configured():
        kwargs["provider"] = config.web_search_provider
        kwargs["provider_api_key"] = config.web_search_provider_api_key

    # Operator knobs resolve from the effective config value (dashboard override /
    # env / YAML) first, falling back to the env var so pure-env deployments are
    # unchanged. A dashboard override mutates ``config``, so it hot-applies here.
    engines_str = (config.web_search_engines if config is not None else None) or otari_env("WEB_SEARCH_ENGINES")
    if engines_str:
        engines = tuple(e.strip() for e in engines_str.split(",") if e.strip())
        if engines:
            kwargs["engines"] = engines

    kwargs["max_results"] = web_search_max_results_baseline(config)
    tool_entry = search_tool_entry or {}
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

    kwargs["enable_search"] = search_tool_entry is not None
    kwargs["enable_fetch"] = fetch_tool_entry is not None
    kwargs["fetch_policy"] = fetch_policy
    kwargs["counter"] = counter

    return WebRetrievalBackend(**kwargs)
