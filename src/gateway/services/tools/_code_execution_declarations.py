"""How a request declares code execution, and who runs a provider's own code execution keyword.

A provider-native code-execution keyword (``code_execution``,
``code_interpreter``, ``code_execution_<date>``) is decided by the request's
**executor** instead (:class:`gateway.models.tools.CodeExecutor`): the
provider, Otari's sandbox, or ``auto``, which picks the provider for each
attempt whose model runs that tool natively and the sandbox for the rest. ``auto``
is the default, and it is what lets a request written against a frontier
model's own sandbox keep working when the model is swapped, or falls back, to
one that has none. The deployment sets the default, a workspace policy may pin
a value, and :data:`CODE_EXECUTION_HEADER` chooses per request where the
workspace has not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from any_llm import AnyLLM

from gateway.models.tools import CodeExecutor
from gateway.services.tools._declarations import Tool, extract_first_matching_tool, first_matching_tool
from gateway.services.tools._native import Dialect

if TYPE_CHECKING:
    from collections.abc import Mapping

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
# here, mirroring the web-search keywords in ``_web_declarations``.
_BARE_CODE_EXECUTION_TYPES = frozenset({"code_execution", "code_interpreter"})
_VERSIONED_CODE_EXECUTION_PREFIX = "code_execution_"
_OPENAI_CODE_INTERPRETER_TYPE = "code_interpreter"
# The providers and wire formats each keyword is native to. Anthropic's dated
# ``code_execution_<date>`` is a Messages server tool; OpenAI's
# ``code_interpreter`` is a Responses built-in tool. Gemini runs code natively on
# both Chat Completions and Messages, and any-llm maps its results into the shape
# each format expects, so Gemini also serves the bare short form and, on
# Messages, Anthropic's dated keyword. Every other pairing is one ``auto`` brings
# to the gateway's sandbox. Vertex AI is left out: the question is asked per
# provider, and that provider serves Claude and Mistral models as well as Gemini.
_GOOGLE_PROVIDERS = frozenset({"gemini"})
_NATIVE_CODE_EXECUTION: dict[str, frozenset[tuple[str, Dialect]]] = {
    _VERSIONED_CODE_EXECUTION_PREFIX: frozenset(
        {("anthropic", Dialect.MESSAGES), *((provider, Dialect.MESSAGES) for provider in _GOOGLE_PROVIDERS)}
    ),
    _OPENAI_CODE_INTERPRETER_TYPE: frozenset({("openai", Dialect.RESPONSES)}),
    "code_execution": frozenset(
        (provider, dialect) for provider in _GOOGLE_PROVIDERS for dialect in (Dialect.CHAT, Dialect.MESSAGES)
    ),
}
# Gemini's own declaration: a tool object keyed by the tool name, with no ``type``.
_GEMINI_CODE_EXECUTION_KEY = "code_execution"


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
    return first_matching_tool(tools, _is_provider_code_execution_tool_type)


def native_code_execution_dialect(tool_entry: Mapping[str, Any] | None) -> Dialect | None:
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
        return Dialect.MESSAGES
    if type_value == _OPENAI_CODE_INTERPRETER_TYPE:
        return Dialect.RESPONSES
    return None


def provider_runs_code_natively(
    tool_entry: dict[str, Any] | None, *, provider: str | None, dialect: Dialect, model: str | None = None
) -> bool:
    """Whether the dispatched provider would run this declaration in its own sandbox.

    True only when the keyword is the provider's own vocabulary *and* the request
    arrived in the wire format that vocabulary is native to: Anthropic's dated
    keyword on Messages against an Anthropic or Gemini model, OpenAI's
    ``code_interpreter`` on Responses against an OpenAI model, the bare
    ``code_execution`` against Gemini on Chat Completions or Messages (see
    ``_NATIVE_CODE_EXECUTION``). Everything else (a Mistral model asked in
    Anthropic's words, an unknown provider) is a declaration the provider cannot
    honor, which is exactly when ``auto`` brings the code here.
    """
    if provider is None or tool_entry is None:
        return False
    type_value = tool_entry.get("type")
    if not isinstance(type_value, str):
        return False
    key = _VERSIONED_CODE_EXECUTION_PREFIX if type_value.startswith(_VERSIONED_CODE_EXECUTION_PREFIX) else type_value
    return (provider.lower(), dialect) in _NATIVE_CODE_EXECUTION.get(key, frozenset())


def _is_gemini_code_execution_tool(entry: Any) -> bool:
    """Gemini's own ``{"code_execution": {...}}`` declaration, which carries no ``type``."""
    return isinstance(entry, dict) and set(entry) == {_GEMINI_CODE_EXECUTION_KEY}


def with_typed_code_execution_tool(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """``tools`` with Gemini's own ``{"code_execution": {}}`` declared as the bare ``code_execution`` keyword.

    Admission reads a declaration's ``type``, so a request in Gemini's spelling is
    decided like any other code-execution keyword: a workspace pinned to ``otari``
    runs it on the sandbox, and a non-Gemini candidate under ``auto`` does too. A
    Gemini attempt gets Gemini's spelling back (:func:`with_native_code_execution_tool`).
    """
    if not tools or not any(_is_gemini_code_execution_tool(entry) for entry in tools):
        return tools
    return [{"type": "code_execution"} if _is_gemini_code_execution_tool(entry) else entry for entry in tools]


def with_native_code_execution_tool(
    tools: list[dict[str, Any]] | None, *, provider: str
) -> list[dict[str, Any]] | None:
    """``tools`` with a provider-named code-execution keyword in Gemini's own form, for a Gemini attempt.

    Gemini declares the tool as ``{"code_execution": {}}``, and rejects the
    ``type``-keyed spellings the other providers use. Rewritten per attempt, not
    per request, so a fallback to another provider still receives the keyword as
    the caller wrote it. A second keyword collapses into the first, since Gemini
    takes the tool once.
    """
    if not tools or provider.lower() not in _GOOGLE_PROVIDERS:
        return tools
    out: list[dict[str, Any]] = []
    declared = False
    for entry in tools:
        if isinstance(entry, dict) and (
            _is_provider_code_execution_tool_type(entry.get("type")) or _is_gemini_code_execution_tool(entry)
        ):
            if not declared:
                out.append({_GEMINI_CODE_EXECUTION_KEY: {}})
                declared = True
            continue
        out.append(entry)
    return out


def provider_attempt_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """``kwargs`` adjusted for the provider its ``provider:model`` selector dispatches to.

    Idempotent, so every attempt path can apply it without tracking whether another already has.
    """
    try:
        provider, _ = AnyLLM.split_model_provider(str(kwargs.get("model") or ""))
    except Exception:  # noqa: BLE001 - an unknown selector is left for the provider call to refuse
        return kwargs
    tools = kwargs.get("tools")
    rewritten = with_native_code_execution_tool(tools, provider=str(getattr(provider, "value", provider)))
    return kwargs if rewritten is tools else {**kwargs, "tools": rewritten}


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


def extract_code_execution_tool(
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
