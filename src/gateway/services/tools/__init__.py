"""The tools the gateway runs itself when a model calls them.

``BUILTIN_TOOLS`` lists every such tool, and ``BuiltinTool`` is the shape of one entry.
``native_rendering`` answers how one tool's calls are announced in a given wire dialect,
and ``ToolUseBudget`` is the per-request cap on one tool's gateway-run calls.
``Tool`` names the ``tools[].type`` values the gateway runs itself.
``claim_web_declarations``, ``extract_web_tools`` and ``admit_web_access`` admit a request's managed web tools,
``admit_code_execution`` admits its code execution, ``admit_mcp_servers`` its MCP servers,
and ``apply_web_access_policy`` narrows its web access to what its workspace permits.
``web_search_max_results_baseline`` is how many search results a request gets when it names none.
"""

from gateway.services.tools._builtin_tool import BuiltinTool
from gateway.services.tools._code_execution_admission import (
    CONTAINER_GONE_DETAIL_TEMPLATE,
    AdmittedCodeExecution,
    admit_code_execution,
)
from gateway.services.tools._code_execution_declarations import (
    CODE_EXECUTION_HEADER,
    code_execution_declaration_forms,
    decide_code_executor,
    declares_code_execution,
    extract_code_execution_tool,
    first_provider_code_execution_tool,
    native_code_execution_dialect,
    parse_code_execution_header,
    provider_runs_code_natively,
    resolve_code_executor_preference,
)
from gateway.services.tools._code_execution_responses import CODE_INTERPRETER_CALL_ID_PREFIX
from gateway.services.tools._declarations import Tool, extract_first_matching_tool
from gateway.services.tools._mcp_admission import admit_mcp_servers
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
    check_web_tools_alone,
    claim_web_declarations,
    extract_web_tools,
    read_web_search_max_uses,
)
from gateway.services.tools._web_declarations import (
    WEB_SEARCH_HEADER,
    web_search_declaration_forms,
    web_search_intercept_enabled,
)
from gateway.services.tools._web_search_keys import WebSearchKeyService, workspace_search_credential
from gateway.services.tools._web_search_responses import WEB_SEARCH_CALL_ID_PREFIX
from gateway.services.tools._web_search_results import web_search_max_results_baseline

__all__ = [
    "BUILTIN_TOOLS",
    "CODE_EXECUTION_HEADER",
    "CODE_INTERPRETER_CALL_ID_PREFIX",
    "CONTAINER_GONE_DETAIL_TEMPLATE",
    "MAX_USES_EXCEEDED_ERROR",
    "SERVER_TOOL_USE_ID_PREFIX",
    "WEB_SEARCH_CALL_ID_PREFIX",
    "WEB_SEARCH_HEADER",
    "AdmittedCodeExecution",
    "BuiltinTool",
    "DeclaredWebTools",
    "Dialect",
    "NativeCall",
    "NativeRendering",
    "Tool",
    "ToolUseBudget",
    "WebAccessGrant",
    "WebSearchKeyService",
    "admit_code_execution",
    "admit_mcp_servers",
    "admit_web_access",
    "apply_web_access_policy",
    "check_web_tools_alone",
    "claim_web_declarations",
    "code_execution_declaration_forms",
    "decide_code_executor",
    "declares_code_execution",
    "extract_code_execution_tool",
    "extract_first_matching_tool",
    "extract_web_tools",
    "first_provider_code_execution_tool",
    "is_capped_call",
    "native_code_execution_dialect",
    "native_rendering",
    "parse_code_execution_header",
    "provider_runs_code_natively",
    "read_web_search_max_uses",
    "resolve_code_executor_preference",
    "web_search_declaration_forms",
    "web_search_intercept_enabled",
    "web_search_max_results_baseline",
    "workspace_search_credential",
]
