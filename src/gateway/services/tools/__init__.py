"""The tools the gateway runs itself when a model calls them.

``BUILTIN_TOOLS`` lists every such tool, and ``BuiltinTool`` is the shape of one entry.
``native_rendering`` answers how one tool's calls are announced in a given wire dialect,
and ``ToolUseBudget`` is the per-request cap on one tool's gateway-run calls.
``Tool`` names the ``tools[].type`` values the gateway runs itself.
``claim_web_declarations``, ``extract_web_tools`` and ``admit_web_access`` admit a request's managed web tools,
and ``apply_web_access_policy`` narrows its web access to what its workspace permits.
``web_search_max_results_baseline`` is how many search results a request gets when it names none.
"""

from gateway.services.tools._builtin_tool import BuiltinTool
from gateway.services.tools._declarations import Tool, extract_first_matching_tool
from gateway.services.tools._native import (
    SERVER_TOOL_USE_ID_PREFIX,
    Dialect,
    NativeCall,
    NativeRendering,
)
from gateway.services.tools._registry import BUILTIN_TOOLS, native_rendering
from gateway.services.tools._use_budget import MAX_USES_EXCEEDED_ERROR, ToolUseBudget, is_capped_call
from gateway.services.tools._web_access import WebAccessGrant, apply_web_access_policy
from gateway.services.tools._web_admission import (
    DeclaredWebTools,
    admit_web_access,
    claim_web_declarations,
    extract_web_tools,
    read_web_search_max_uses,
)
from gateway.services.tools._web_declarations import (
    WEB_SEARCH_HEADER,
    web_search_declaration_forms,
    web_search_intercept_enabled,
)
from gateway.services.tools._web_search_results import web_search_max_results_baseline

__all__ = [
    "BUILTIN_TOOLS",
    "MAX_USES_EXCEEDED_ERROR",
    "SERVER_TOOL_USE_ID_PREFIX",
    "WEB_SEARCH_HEADER",
    "BuiltinTool",
    "DeclaredWebTools",
    "Dialect",
    "NativeCall",
    "NativeRendering",
    "Tool",
    "ToolUseBudget",
    "WebAccessGrant",
    "admit_web_access",
    "apply_web_access_policy",
    "claim_web_declarations",
    "extract_first_matching_tool",
    "extract_web_tools",
    "is_capped_call",
    "native_rendering",
    "read_web_search_max_uses",
    "web_search_declaration_forms",
    "web_search_intercept_enabled",
    "web_search_max_results_baseline",
]
