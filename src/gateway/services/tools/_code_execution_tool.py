"""Code execution, as the tool registry lists it."""

from gateway.core.config import GatewayConfig
from gateway.services.sandbox_backend import CODE_EXECUTION_TOOL_NAME, code_execution_tool_definition
from gateway.services.tools import _code_execution_messages, _code_execution_responses
from gateway.services.tools._builtin_tool import BuiltinTool
from gateway.services.tools._native import Dialect

TOOL = BuiltinTool(
    name=CODE_EXECUTION_TOOL_NAME,
    definition=code_execution_tool_definition,
    configured=GatewayConfig.sandbox_configured,
    native={
        Dialect.MESSAGES: _code_execution_messages.RENDERING,
        Dialect.RESPONSES: _code_execution_responses.RENDERING,
    },
)
