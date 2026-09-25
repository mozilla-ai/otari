"""The registry of tools the gateway runs itself.

The one list of tools a model can call that the gateway answers instead of the caller.
It is a literal tuple edited by hand: nothing is discovered and nothing registers itself on import.
Each entry names the ``BuiltinTool`` its own module declares, so nothing is built here.
"""

from gateway.services.tools import _code_execution_tool, _web_fetch_tool, _web_search_tool
from gateway.services.tools._builtin_tool import BuiltinTool
from gateway.services.tools._native import Dialect, NativeRendering

BUILTIN_TOOLS: tuple[BuiltinTool, ...] = (_web_search_tool.TOOL, _web_fetch_tool.TOOL, _code_execution_tool.TOOL)


def native_rendering(name: str, dialect: Dialect) -> NativeRendering | None:
    """How ``name``'s gateway-run calls are announced in ``dialect``, or ``None`` where they are not.

    ``None`` covers both a tool with nothing to announce in that dialect and a name
    the gateway does not run at all, such as one an MCP server supplied.
    """
    for tool in BUILTIN_TOOLS:
        if tool.name == name:
            return tool.native.get(dialect)
    return None
