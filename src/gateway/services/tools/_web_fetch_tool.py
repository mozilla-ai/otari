"""Web fetch, as the tool registry lists it."""

from gateway.services.tools._builtin_tool import BuiltinTool
from gateway.services.tools._web_names import WEB_FETCH_TOOL_NAME, web_fetch_tool_definition

TOOL = BuiltinTool(
    name=WEB_FETCH_TOOL_NAME,
    definition=web_fetch_tool_definition,
    configured=lambda config: config.web_fetch_enabled,
)
