"""Code execution, as the tool registry lists it."""

from gateway.core.config import GatewayConfig
from gateway.core.env import otari_env
from gateway.services.builtin_tool import BuiltinTool
from gateway.services.sandbox_backend import CODE_EXECUTION_TOOL_NAME, code_execution_tool_definition


def _configured(config: GatewayConfig) -> bool:
    return bool(config.sandbox_url or otari_env("SANDBOX_URL"))


TOOL = BuiltinTool(
    name=CODE_EXECUTION_TOOL_NAME,
    definition=code_execution_tool_definition,
    configured=_configured,
)
