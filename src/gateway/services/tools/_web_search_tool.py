"""Web search, as the tool registry lists it."""

from gateway.services.tools import _web_search_chat, _web_search_messages, _web_search_responses
from gateway.services.tools._builtin_tool import BuiltinTool
from gateway.services.tools._native import Dialect
from gateway.services.tools._web_declarations import declares_web_search, provider_runs_web_search_natively
from gateway.services.tools._web_names import WEB_SEARCH_TOOL_NAME, web_search_tool_definition
from gateway.services.tools._web_search import web_search_configured
from gateway.types.key_scope import KeyScope

TOOL = BuiltinTool(
    name=WEB_SEARCH_TOOL_NAME,
    definition=web_search_tool_definition,
    configured=web_search_configured,
    declares=declares_web_search,
    key_scope=KeyScope.WEB_SEARCH,
    runs_natively=provider_runs_web_search_natively,
    native={
        Dialect.CHAT: _web_search_chat.RENDERING,
        Dialect.MESSAGES: _web_search_messages.RENDERING,
        Dialect.RESPONSES: _web_search_responses.RENDERING,
    },
)
