"""The tools the gateway runs itself when a model calls them.

``BUILTIN_TOOLS`` lists every such tool, and ``BuiltinTool`` is the shape of one entry.
``native_rendering`` answers how one tool's calls are announced in a given wire dialect,
and ``ToolUseBudget`` is the per-request cap on one tool's gateway-run calls.
"""

from gateway.services.tools._builtin_tool import BuiltinTool
from gateway.services.tools._native import (
    SERVER_TOOL_USE_ID_PREFIX,
    Dialect,
    NativeCall,
    NativeRendering,
)
from gateway.services.tools._registry import BUILTIN_TOOLS, native_rendering
from gateway.services.tools._use_budget import MAX_USES_EXCEEDED_ERROR, ToolUseBudget, is_capped_call

__all__ = [
    "BUILTIN_TOOLS",
    "MAX_USES_EXCEEDED_ERROR",
    "SERVER_TOOL_USE_ID_PREFIX",
    "BuiltinTool",
    "Dialect",
    "NativeCall",
    "NativeRendering",
    "ToolUseBudget",
    "is_capped_call",
    "native_rendering",
]
