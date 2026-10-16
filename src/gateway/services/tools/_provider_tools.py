"""Who runs the built-in tools on one provider, for a caller that leaves the choice to the deployment.

A request's answer depends on how it declares each tool and who it asks to run
it (``decide_candidate_tools``). This answers the question a page listing
providers asks before any request exists, under two assumptions: the caller
declares each tool in the provider's own words, in the dialect they are native
to, and sends no ``Otari-*`` executor header, so the deployment's settings
decide. Code execution then follows the deployment's executor, web search is
claimed where interception is on, and every other choice is ``auto``.
A workspace's code execution pin is not consulted.
"""

from typing import Any

from gateway.core.config import GatewayConfig
from gateway.models.tools import CodeExecutor
from gateway.services.tools import _code_execution_tool, _web_search_tool
from gateway.services.tools._builtin_tool import BuiltinTool
from gateway.services.tools._candidate_tools import CandidateTools, ProviderToolDeclaration, decide_candidate_tools
from gateway.services.tools._native import Dialect
from gateway.services.tools._registry import BUILTIN_TOOLS
from gateway.services.tools._web_declarations import web_search_intercept_enabled

# One declaration of a built-in tool in each provider vocabulary the gateway
# recognizes, with the dialect it is native to. Which provider runs which is
# each tool's own ``runs_natively`` answer, not this list's.
_PROVIDER_WORDS: tuple[tuple[Dialect, dict[str, Any]], ...] = (
    (Dialect.MESSAGES, {"type": "web_search_20250305"}),
    (Dialect.MESSAGES, {"type": "web_fetch_20250910"}),
    (Dialect.MESSAGES, {"type": "code_execution_20250825"}),
    (Dialect.RESPONSES, {"type": "web_search"}),
    (Dialect.RESPONSES, {"type": "code_interpreter"}),
)


# Providers whose callers declare tools in the provider's own format, which the
# gateway forwards untouched (Gemini's ``google_search``, ``url_context`` and
# ``code_execution``): the provider runs every one of them itself.
_OWN_TOOL_FORMATS = frozenset({"gemini", "vertexai"})


def _deployment_preference(config: GatewayConfig, tool: BuiltinTool) -> CodeExecutor:
    if tool is _code_execution_tool.TOOL:
        return config.effective_code_executor()
    if tool is _web_search_tool.TOOL and web_search_intercept_enabled(config):
        return CodeExecutor.OTARI
    return CodeExecutor.AUTO


def provider_tools(config: GatewayConfig, provider: str) -> CandidateTools:
    """Which built-in tools ``provider`` runs itself (``native``) and which Otari runs for it (``claimed``).

    ``provider`` is the any-llm provider an instance dispatches to. A tool it runs
    under none of the declarations above is decided in a dialect where nothing is
    native, so Otari runs it wherever its preference and a backend allow.
    """
    if provider in _OWN_TOOL_FORMATS:
        # Native everywhere, so only a deployment that always runs a tool here claims it.
        claimed = frozenset(
            tool.name
            for tool in BUILTIN_TOOLS
            if _deployment_preference(config, tool) is CodeExecutor.OTARI and tool.configured(config)
        )
        return CandidateTools(claimed=claimed, native=frozenset(tool.name for tool in BUILTIN_TOOLS) - claimed)
    answer = CandidateTools()
    for tool in BUILTIN_TOOLS:
        dialect, entry = next(
            (
                (dialect, entry)
                for dialect, entry in _PROVIDER_WORDS
                if tool.runs_natively(entry, provider=provider, dialect=dialect)
            ),
            (Dialect.CHAT, {"type": tool.name}),
        )
        declaration = ProviderToolDeclaration(
            tool=tool,
            entry=entry,
            preference=_deployment_preference(config, tool),
            backend_configured=tool.configured(config),
        )
        answer |= decide_candidate_tools([declaration], providers=[provider], dialect=dialect)[0]
    return answer
