"""Code execution, as the tool registry lists it."""

from gateway.core.config import GatewayConfig
from gateway.services.builtin_tool import BuiltinTool
from gateway.services.sandbox_backend import CODE_EXECUTION_TOOL_NAME, code_execution_tool_definition

TOOL = BuiltinTool(
    name=CODE_EXECUTION_TOOL_NAME,
    definition=code_execution_tool_definition,
    configured=GatewayConfig.sandbox_configured,
)
