"""The tools the gateway runs itself when a model calls them.

``BUILTIN_TOOLS`` lists every such tool, and ``BuiltinTool`` is the shape of one entry.
"""

from gateway.services.tools._builtin_tool import BuiltinTool
from gateway.services.tools._registry import BUILTIN_TOOLS

__all__ = ["BUILTIN_TOOLS", "BuiltinTool"]
