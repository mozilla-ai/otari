"""Web search, as the tool registry lists it."""

from gateway.core.config import GatewayConfig
from gateway.services.tools import _web_search_messages, _web_search_responses
from gateway.services.tools._builtin_tool import BuiltinTool
from gateway.services.tools._native import Dialect
from gateway.services.web_retrieval_backend import WEB_SEARCH_TOOL_NAME, web_search_tool_definition

TOOL = BuiltinTool(
    name=WEB_SEARCH_TOOL_NAME,
    definition=web_search_tool_definition,
    configured=GatewayConfig.web_search_configured,
    native={
        Dialect.MESSAGES: _web_search_messages.RENDERING,
        Dialect.RESPONSES: _web_search_responses.RENDERING,
    },
)
