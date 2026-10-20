"""Web fetch, as the tool registry lists it."""

from gateway.services.tools import _web_fetch_chat, _web_fetch_messages
from gateway.services.tools._builtin_tool import BuiltinTool
from gateway.services.tools._native import Dialect
from gateway.services.tools._web_declarations import provider_runs_web_fetch_natively
from gateway.services.tools._web_names import WEB_FETCH_TOOL_NAME, web_fetch_tool_definition

TOOL = BuiltinTool(
    name=WEB_FETCH_TOOL_NAME,
    definition=web_fetch_tool_definition,
    configured=lambda config: config.web_fetch_enabled,
    runs_natively=provider_runs_web_fetch_natively,
    native={Dialect.CHAT: _web_fetch_chat.RENDERING, Dialect.MESSAGES: _web_fetch_messages.RENDERING},
)
