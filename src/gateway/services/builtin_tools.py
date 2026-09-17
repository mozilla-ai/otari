"""The registry of tools the gateway runs itself.

The one list of tools a model can call that the gateway answers instead of the caller.
It is a literal tuple edited by hand: nothing is discovered and nothing registers itself on import.
Each entry names the ``BuiltinTool`` its own module declares, so nothing is built here.
"""

from gateway.services import code_execution_tool, web_search_tool
from gateway.services.builtin_tool import BuiltinTool

BUILTIN_TOOLS: tuple[BuiltinTool, ...] = (web_search_tool.TOOL, code_execution_tool.TOOL)
