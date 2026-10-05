"""Helpers for extracting gateway-managed tools from a request payload.

These helpers are format-agnostic — they only look at the `type` string on
each tool entry. The same predicates and extractors are used from the
Chat-Completions, Anthropic Messages, and OpenAI Responses endpoints so
``otari_code_execution`` / ``otari_web_search`` requests get identical
handling regardless of wire shape.

The explicit ``otari_*`` tool types always trigger gateway-side execution.
Which web-search keywords the gateway claims is decided in ``services/tools/_web_declarations.py``.

A provider-native code-execution keyword (``code_execution``,
``code_interpreter``, ``code_execution_<date>``) is decided by the request's
**executor** instead (:class:`gateway.models.tools.CodeExecutor`): the
provider, Otari's sandbox, or ``auto``, which picks the provider only when it
runs that tool natively for the dispatched model. ``auto`` is the default, and
it is what lets a request written against a frontier model's own sandbox keep
working when the model is swapped for one that has none. The deployment sets
the default, a workspace policy may pin a value, and :data:`CODE_EXECUTION_HEADER`
chooses per request where the workspace has not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gateway.api.routes._schema_derive import SENSITIVE_PARAM_FIELDS
from gateway.core.config import parse_bool_env
from gateway.core.env import otari_env
from gateway.models.tools import CodeExecutor
from gateway.services.tool_usage import ToolUsageTally
from gateway.services.tools import Dialect, Tool, extract_first_matching_tool, web_search_max_results_baseline
from gateway.services.web_retrieval_backend import (
    WEB_SEARCH_TOOL_NAME,
    WebRetrievalBackend,
    WebRetrievalCounter,
)
from gateway.services.web_retrieval_policy import DomainPolicy

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig

# Per-request choice of who runs a provider-native code-execution declaration.
# A header rather than a body field so the body stays the untouched payload a
# provider's own SDK sends; every SDK can add a default header without a code
# change. One of ``CodeExecutor``'s values, case-insensitive.
CODE_EXECUTION_HEADER = "Otari-Code-Execution"


def _is_code_execution_tool_type(type_value: Any) -> bool:
    """Recognize the explicit gateway-managed code-execution tool type.

    Matches only ``"otari_code_execution"``. The provider-named keywords are
    :func:`_is_provider_code_execution_tool_type`'s, and whether the gateway
    claims one is the executor's decision, not the keyword's.
    """
    if not isinstance(type_value, str):
        return False
    return type_value == Tool.CODE_EXECUTION


# The provider-named code-execution keywords: OpenAI's ``code_interpreter``, the
# bare short form, and any dated/preview variant. The prefix match keeps future
# Anthropic versions (``code_execution_20991231``) working without a release
# here, mirroring the web-search keywords above.
_BARE_CODE_EXECUTION_TYPES = frozenset({"code_execution", "code_interpreter"})
_VERSIONED_CODE_EXECUTION_PREFIX = "code_execution_"
_OPENAI_CODE_INTERPRETER_TYPE = "code_interpreter"
# The one provider each native vocabulary belongs to, and the wire format it is
# native in. Anthropic's dated ``code_execution_<date>`` is a Messages server
# tool; OpenAI's ``code_interpreter`` is a Responses built-in tool. Neither has a
# native form on Chat Completions, and the bare ``code_execution`` short form is
# nobody's, so a request declaring it is never natively served and ``auto``
# always runs it here.
_NATIVE_CODE_EXECUTION: dict[str, tuple[str, Dialect]] = {
    _VERSIONED_CODE_EXECUTION_PREFIX: ("anthropic", Dialect.MESSAGES),
    _OPENAI_CODE_INTERPRETER_TYPE: ("openai", Dialect.RESPONSES),
}


def _is_provider_code_execution_tool_type(type_value: Any) -> bool:
    """Recognize a provider-named code-execution keyword.

    Matched on the tool ``type`` alone, never on a caller's ``function`` named
    ``code_execution``: that is the caller's own tool, the same carve-out
    the web-search predicates make for a function named ``web_search``.
    Does not match ``otari_code_execution``, which
    :func:`_is_code_execution_tool_type` owns.
    """
    if not isinstance(type_value, str):
        return False
    return type_value in _BARE_CODE_EXECUTION_TYPES or type_value.startswith(_VERSIONED_CODE_EXECUTION_PREFIX)


def _is_any_code_execution_tool_type(type_value: Any) -> bool:
    """The gateway-managed type or a provider-named keyword."""
    return _is_code_execution_tool_type(type_value) or _is_provider_code_execution_tool_type(type_value)


def declares_code_execution(tools: list[dict[str, Any]] | None) -> bool:
    """Whether ``tools`` asks for code execution in any vocabulary, the gateway's or a provider's."""
    return any(isinstance(entry, dict) and _is_any_code_execution_tool_type(entry.get("type")) for entry in tools or [])


def first_provider_code_execution_tool(tools: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    """The first provider-named code-execution entry in ``tools``, left in place."""
    for entry in tools or []:
        if isinstance(entry, dict) and _is_provider_code_execution_tool_type(entry.get("type")):
            return entry
    return None


def native_code_execution_dialect(tool_entry: dict[str, Any] | None) -> Dialect | None:
    """The wire format whose native result blocks the caller expects, or ``None``.

    ``"messages"`` for Anthropic's dated keyword, which is what the Anthropic SDK
    and Claude Code send and what makes them expect ``server_tool_use`` and
    ``code_execution_tool_result`` blocks back. ``"responses"`` for OpenAI's
    ``code_interpreter``, whose callers expect a ``code_interpreter_call`` item.
    ``None`` for ``otari_code_execution`` and for the bare ``code_execution``
    short form, neither of which implies a native response shape, so those
    callers keep receiving the plain tool-loop result they always have.
    """
    type_value = tool_entry.get("type") if tool_entry else None
    if not isinstance(type_value, str):
        return None
    if type_value.startswith(_VERSIONED_CODE_EXECUTION_PREFIX):
        return _NATIVE_CODE_EXECUTION[_VERSIONED_CODE_EXECUTION_PREFIX][1]
    if type_value == _OPENAI_CODE_INTERPRETER_TYPE:
        return _NATIVE_CODE_EXECUTION[_OPENAI_CODE_INTERPRETER_TYPE][1]
    return None


def provider_runs_code_natively(tool_entry: dict[str, Any] | None, *, provider: str | None, dialect: Dialect) -> bool:
    """Whether the dispatched provider would run this declaration in its own sandbox.

    True only when the keyword is the provider's own vocabulary *and* the request
    arrived in the wire format that vocabulary is native to: Anthropic's dated
    keyword on Messages against an Anthropic model, OpenAI's ``code_interpreter``
    on Responses against an OpenAI model. Everything else (a Mistral model asked
    in Anthropic's words, any keyword on Chat Completions, an unknown provider)
    is a declaration the provider cannot honor, which is exactly when ``auto``
    brings the code here.
    """
    if provider is None or tool_entry is None:
        return False
    type_value = tool_entry.get("type")
    if not isinstance(type_value, str):
        return False
    key = _VERSIONED_CODE_EXECUTION_PREFIX if type_value.startswith(_VERSIONED_CODE_EXECUTION_PREFIX) else type_value
    native = _NATIVE_CODE_EXECUTION.get(key)
    return native is not None and native == (provider.lower(), dialect)


def parse_code_execution_header(value: str | None) -> CodeExecutor | None:
    """The executor a request asked for, ``None`` when it asked for none.

    Raises ``ValueError`` for a value outside the vocabulary: a misspelled header
    is a caller mistake to report, not a default to fall back to.
    """
    if value is None or not value.strip():
        return None
    executor = CodeExecutor.parse(value)
    if executor is None:
        msg = f"{CODE_EXECUTION_HEADER} must be one of {', '.join(e.value for e in CodeExecutor)}"
        raise ValueError(msg)
    return executor


def resolve_code_executor_preference(
    *,
    requested: CodeExecutor | None,
    workspace: CodeExecutor | None,
    deployment: CodeExecutor,
) -> tuple[CodeExecutor, bool]:
    """Compose the three layers into one preference, and say whether they clashed.

    A workspace pin wins over the request, and the request wins over the
    deployment default: the workspace's owner set the pin for a billing or data
    reason a caller may not override, while the deployment default is only what
    applies when nobody closer to the request said otherwise. The second value is
    true when the request asked for something the workspace pinned away, so the
    caller can refuse out loud rather than silently run elsewhere.
    """
    if workspace is not None:
        return workspace, requested is not None and requested != workspace
    return requested or deployment, False


def decide_code_executor(
    preference: CodeExecutor,
    *,
    sandbox_configured: bool,
    native_available: bool,
) -> CodeExecutor:
    """Turn a preference into who runs the code: ``OTARI`` or ``PROVIDER``.

    ``AUTO`` prefers the provider when it serves the tool natively, and with no
    sandbox configured it also leaves the provider in charge, because there is
    nothing to bring the code to; an explicit ``OTARI`` is returned as asked so
    the caller can refuse it with the missing-sandbox detail instead.
    """
    if preference is not CodeExecutor.AUTO:
        return preference
    if native_available or not sandbox_configured:
        return CodeExecutor.PROVIDER
    return CodeExecutor.OTARI


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


def _extract_code_execution_tool(
    tools: list[dict[str, Any]] | None,
    *,
    intercept: bool = False,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    """Pull the first gateway-run code-execution entry out of ``tools``.

    With ``intercept`` off (the default) only the explicit
    ``{"type": "otari_code_execution"}`` is extracted; provider-named keywords
    stay in ``tools[]`` and reach the upstream provider unchanged. With it on,
    which is what an executor decision of ``OTARI`` means, the provider-named
    keywords are claimed too, so a client speaking a provider's vocabulary
    reaches the gateway's sandbox.
    """
    predicate = _is_any_code_execution_tool_type if intercept else _is_code_execution_tool_type
    return extract_first_matching_tool(tools, predicate)


def code_execution_declaration_forms(config: GatewayConfig | None = None) -> list[str]:
    """Every ``tools[].type`` this deployment may route to the sandbox.

    Advertised by ``GET /api/v1/tools``. The provider-named keywords appear
    unless the deployment's executor is ``provider``; under ``auto`` they are
    routed here only for a model whose provider does not run them natively,
    which the listing cannot say per model, so it lists the forms the gateway
    is prepared to claim.
    """
    forms = [str(Tool.CODE_EXECUTION)]
    executor = config.effective_code_executor() if config is not None else CodeExecutor.AUTO
    if executor is not CodeExecutor.PROVIDER:
        forms += sorted(_BARE_CODE_EXECUTION_TYPES) + [f"{_VERSIONED_CODE_EXECUTION_PREFIX}<date>"]
    return forms


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
