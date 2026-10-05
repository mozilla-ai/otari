"""How a request declares the managed web tools, and whether the gateway claims a provider's own search.

The explicit ``otari_web_search`` and ``otari_web_fetch`` types always run on the gateway.
A provider-native web-search keyword (``web_search`` / ``web_search_<date>``)
is forwarded to the upstream provider unless ``web_search_intercept`` is on or
:data:`WEB_SEARCH_HEADER` asks otherwise. The header takes ``CodeExecutor``'s
vocabulary: ``auto`` claims the keyword only when a provider in the chain cannot
run it (Anthropic's dated keyword on Messages, OpenAI's on Responses are the
native pairings), ``otari`` always, ``provider`` never, and none of them can
undo interception. Interception is off by default because turning it on
silently takes a search away from a provider that would have run it (see
``docs/tools.md``). An OpenAI ``function`` named ``web_search`` is deliberately
*not* claimed even then: that is a caller's own tool, and hijacking it means the
caller's handler never fires and it never gets back a ``tool_call`` it can
dispatch.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from gateway.core.env import otari_env
from gateway.models.tools import CodeExecutor
from gateway.services.tools._declarations import Tool, extract_first_matching_tool
from gateway.services.tools._native import Dialect
from gateway.services.web_retrieval_backend import WEB_SEARCH_NATIVE_TYPE_PREFIX

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig

# Per-request choice of who runs a provider-named web-search declaration, in the
# same vocabulary and for the same reason as ``CODE_EXECUTION_HEADER``. It can
# add a claim but never remove one ``web_search_intercept`` makes.
WEB_SEARCH_HEADER = "Otari-Web-Search"


# The provider-named web-search keywords the gateway claims when interception is
# on: the bare short form, and any dated/preview variant. The prefix match keeps
# future Anthropic versions (``web_search_20991231``) and OpenAI's Responses
# spellings (``web_search_preview``) working without a release here.
_BARE_WEB_SEARCH_TYPE = "web_search"


def _is_web_search_tool_type(type_value: Any) -> bool:
    """Recognize the explicit gateway-managed web_search tool type.

    Matches only ``"otari_web_search"``. Provider-named keywords
    (``"web_search"``, ``"web_search_<date>"``) are *not* matched; they pass
    through unchanged to the upstream provider, which runs the search itself.
    """
    if not isinstance(type_value, str):
        return False
    return type_value == Tool.WEB_SEARCH


def is_provider_web_search_tool_type(type_value: Any) -> bool:
    """Recognize a provider-named web-search keyword (interception only).

    ``"web_search"`` (Claude Code's short form, OpenAI Responses' native type)
    or any ``"web_search_<suffix>"`` variant. Does not match
    ``"otari_web_search"``, which :func:`_is_web_search_tool_type` owns.
    """
    if not isinstance(type_value, str):
        return False
    return type_value == _BARE_WEB_SEARCH_TYPE or type_value.startswith(WEB_SEARCH_NATIVE_TYPE_PREFIX)


def _is_any_web_search_tool_type(type_value: Any) -> bool:
    """The gateway-managed type or a provider-named keyword."""
    return _is_web_search_tool_type(type_value) or is_provider_web_search_tool_type(type_value)


# Where a provider-named web-search keyword is the provider's own: Anthropic's
# dated keyword on Messages, OpenAI's bare and preview keywords on Responses.
# Every other pairing, a keyword in the other provider's words included, names a
# search the dispatched provider cannot run.
_ANTHROPIC_WEB_SEARCH_TYPE = re.compile(r"web_search_\d{8}")
_OPENAI_WEB_SEARCH_PREVIEW_PREFIX = "web_search_preview"


def _native_web_search_pairing(type_value: Any) -> tuple[str, Dialect] | None:
    """The ``(provider, dialect)`` a provider-named web-search keyword is native to."""
    if not isinstance(type_value, str):
        return None
    if type_value == _BARE_WEB_SEARCH_TYPE or type_value.startswith(_OPENAI_WEB_SEARCH_PREVIEW_PREFIX):
        return ("openai", Dialect.RESPONSES)
    if _ANTHROPIC_WEB_SEARCH_TYPE.fullmatch(type_value):
        return ("anthropic", Dialect.MESSAGES)
    return None


def first_provider_web_search_tool(tools: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    """The first provider-named web-search declaration in ``tools``, if any."""
    for entry in tools or []:
        if isinstance(entry, dict) and is_provider_web_search_tool_type(entry.get("type")):
            return entry
    return None


def provider_runs_web_search_natively(
    tool_entry: dict[str, Any] | None, *, provider: str | None, dialect: Dialect
) -> bool:
    """Whether the dispatched provider would run this web-search declaration itself.

    The web-search counterpart of :func:`provider_runs_code_natively`. ``None``
    for the provider reads as not native, as it does there.
    """
    if provider is None or tool_entry is None:
        return False
    native = _native_web_search_pairing(tool_entry.get("type"))
    return native is not None and native == (provider.lower(), dialect)


def parse_web_search_header(value: str | None) -> CodeExecutor | None:
    """Who a request asked to run its web search, ``None`` when it asked for no one.

    Raises ``ValueError`` for a value outside the vocabulary, as
    :func:`parse_code_execution_header` does.
    """
    if value is None or not value.strip():
        return None
    executor = CodeExecutor.parse(value)
    if executor is None:
        msg = f"{WEB_SEARCH_HEADER} must be one of {', '.join(e.value for e in CodeExecutor)}"
        raise ValueError(msg)
    return executor


def claims_provider_web_search(
    tool_entry: dict[str, Any] | None,
    *,
    requested: CodeExecutor | None,
    intercept: bool,
    backend_configured: bool,
    providers: Sequence[str | None],
    dialect: Dialect,
) -> bool:
    """Whether the gateway runs a provider-named web-search declaration itself.

    Only with a backend to run it on. Interception claims every keyword, and the
    request's :data:`WEB_SEARCH_HEADER` cannot take that back (see
    :func:`web_search_header_conflicts`); without interception the header decides,
    and without either nothing is claimed. ``auto`` claims a keyword unless every
    candidate in ``providers`` (the fallback chain, head first) runs it natively,
    so a chain that falls back to a model with no search of its own never
    forwards it a search nobody will run.
    """
    if tool_entry is None or not backend_configured:
        return False
    if intercept:
        return True
    if requested is CodeExecutor.AUTO:
        return not providers or not all(
            provider_runs_web_search_natively(tool_entry, provider=provider, dialect=dialect) for provider in providers
        )
    return requested is CodeExecutor.OTARI


def web_search_header_conflicts(requested: CodeExecutor | None, *, intercept: bool) -> bool:
    """Whether the request asked the provider to run a search the deployment claims.

    ``web_search_intercept`` is what puts every search under the workspace's
    web-search policy and tool pricing, so a caller's header may not opt out of it.
    """
    return intercept and requested is CodeExecutor.PROVIDER


def _is_web_fetch_tool_type(type_value: Any) -> bool:
    """Recognize only the canonical gateway-managed Fetch declaration."""
    return isinstance(type_value, str) and type_value == Tool.WEB_FETCH


def extract_web_search_tool(
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
    return extract_first_matching_tool(tools, predicate)


def extract_web_fetch_tool(
    tools: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    """Pull the first canonical Fetch declaration, leaving native types alone."""
    return extract_first_matching_tool(tools, _is_web_fetch_tool_type)


def web_search_intercept_enabled(config: GatewayConfig | None = None) -> bool:
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

    Advertised by ``GET /api/v1/tools``. The dated form is spelled with a placeholder
    (``web_search_<date>``) because the match is a prefix, not a fixed list: any
    suffix works, including future Anthropic versions.
    """
    forms = [str(Tool.WEB_SEARCH)]
    if web_search_intercept_enabled(config):
        forms += [_BARE_WEB_SEARCH_TYPE, f"{WEB_SEARCH_NATIVE_TYPE_PREFIX}<date>"]
    return forms
