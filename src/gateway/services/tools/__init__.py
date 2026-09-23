"""The tools the gateway runs itself when a model calls them.

``BUILTIN_TOOLS`` lists every such tool, and ``BuiltinTool`` is the shape of one entry.
``native_rendering`` answers how one tool's calls are announced in a given wire dialect.
"""

from gateway.services.tools._builtin_tool import BuiltinTool
from gateway.services.tools._native import (
    SERVER_TOOL_USE_ID_PREFIX,
    Dialect,
    NativeCall,
    NativeRendering,
)
from gateway.services.tools._registry import BUILTIN_TOOLS, native_rendering

__all__ = [
    "BUILTIN_TOOLS",
    "SERVER_TOOL_USE_ID_PREFIX",
    "BuiltinTool",
    "Dialect",
    "NativeCall",
    "NativeRendering",
    "native_rendering",
]
