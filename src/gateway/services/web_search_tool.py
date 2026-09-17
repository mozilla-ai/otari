"""Web search, as the tool registry lists it."""

from gateway.core.config import GatewayConfig
from gateway.services.builtin_tool import BuiltinTool
from gateway.services.web_retrieval_backend import WEB_SEARCH_TOOL_NAME, web_search_tool_definition

TOOL = BuiltinTool(
    name=WEB_SEARCH_TOOL_NAME,
    definition=web_search_tool_definition,
    configured=GatewayConfig.web_search_configured,
)
