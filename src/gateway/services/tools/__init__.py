"""The tools the gateway runs itself when a model calls them.

``BUILTIN_TOOLS`` lists every such tool, and ``BuiltinTool`` is the shape of one entry.
``native_rendering`` answers how one tool's calls are announced in a given wire dialect,
``decide_candidate_tools`` which of a request's built-in tools Otari runs on each candidate it may be sent to,
and ``ToolUseBudget`` is the per-request cap on each tool's gateway-run calls.
``Tool`` names the ``tools[].type`` values the gateway runs itself.
``read_web_declarations``, ``extract_web_tools`` and ``admit_web_access`` admit a request's managed web tools,
``declare_code_execution`` its code execution, ``admit_mcp_servers`` its MCP servers,
``refuse_shadowed_builtin_tools`` refuses connected MCP servers whose tool names shadow a declared built-in,
and ``apply_web_access_policy`` narrows its web access to what its workspace permits.
``web_search_max_results_baseline`` is how many search results a request gets when it names none.
``search_provider_catalog`` lists the providers a search or fetch tool may name.
``check_tool_entry``, ``check_tool_write``, ``check_kind_unchanged``, ``check_tool_name_is_free`` and
``default_to_pin`` hold a write through ``/api/v1/search-tools`` to the instance rules, and
``run_connection_test`` tests an instance with one search or fetch.
``resolve_search`` and ``resolve_fetch`` say which provider, key and options a web search or fetch runs with.
``pick_direct_search_instance``, ``resolve_direct_search`` and ``run_direct_search`` hold the direct
search endpoint's rules and run its call, and ``SearchQuery`` with the helpers beside it map LiteLLM's
search fields onto a provider. ``build_web_provider_client`` builds the client the web tools' providers
are called through, one per app, and ``WebToolFailure`` is what a failed web tool call may show its caller.
``OTARI_TOOL_CALLS_FIELD`` is where a Chat Completions response reports the gateway's web tool calls, and
``without_otari_tool_calls`` takes it back off the messages a client sends.
``WEB_SEARCH_TOOL_NAME``, ``WEB_FETCH_TOOL_NAME`` and the two tool definitions name the web tools to the model.
"""

from gateway.services.tools._builtin_tool import BuiltinTool
from gateway.services.tools._candidate_tools import CandidateTools, ProviderToolDeclaration, decide_candidate_tools
from gateway.services.tools._chat_tool_calls import OTARI_TOOL_CALLS_FIELD, without_otari_tool_calls
from gateway.services.tools._code_execution_admission import (
    CONTAINER_GONE_DETAIL_TEMPLATE,
    AdmittedCodeExecution,
    DeclaredCodeExecution,
    declare_code_execution,
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
from gateway.services.tools._code_execution_policy import (
    WorkspaceCodeExecutionPolicies,
    WorkspaceCodeExecutionPolicyPublic,
    WorkspaceCodeExecutionPolicyService,
    WorkspaceCodeExecutionPolicyUpdate,
    read_code_execution_policy,
)
from gateway.services.tools._code_execution_responses import CODE_INTERPRETER_CALL_ID_PREFIX
from gateway.services.tools._code_execution_tool import TOOL as CODE_EXECUTION_TOOL
from gateway.services.tools._combined_backend import CombinedToolBackend, owning_backend
from gateway.services.tools._declarations import Tool, extract_first_matching_tool
from gateway.services.tools._loop_limits import MAX_TOOL_ITERATIONS_CAP
from gateway.services.tools._mcp_admission import admit_mcp_servers, refuse_shadowed_builtin_tools
from gateway.services.tools._native import (
    SERVER_TOOL_USE_ID_PREFIX,
    Dialect,
    NativeCall,
    NativeRendering,
)
from gateway.services.tools._registry import BUILTIN_TOOLS, native_rendering
from gateway.services.tools._search_provider_catalog import search_provider_catalog
from gateway.services.tools._search_tool_connection import (
    check_test_input,
    instance_to_test,
    run_connection_test,
    unsaved_instance_to_test,
)
from gateway.services.tools._search_tool_writes import (
    DefaultPin,
    check_kind_unchanged,
    check_tool_entry,
    check_tool_name_is_free,
    check_tool_write,
    default_to_pin,
)
from gateway.services.tools._use_budget import MAX_USES_EXCEEDED_ERROR, ToolUseBudget, is_capped_call
from gateway.services.tools._web_access import WebAccessGrant, apply_web_access_policy
from gateway.services.tools._web_admission import (
    DeclaredWebTools,
    WebClaims,
    admit_web_access,
    extract_web_tools,
    read_web_declarations,
    read_web_fetch_max_uses,
    read_web_search_max_uses,
    validate_web_declarations,
)
from gateway.services.tools._web_declarations import (
    WEB_FETCH_HEADER,
    WEB_SEARCH_HEADER,
    web_search_declaration_forms,
    web_search_intercept_enabled,
)
from gateway.services.tools._web_names import (
    WEB_FETCH_TOOL_NAME,
    WEB_RETRIEVAL_RESULT_MAX_BYTES,
    WEB_SEARCH_NATIVE_TYPE_PREFIX,
    WEB_SEARCH_TOOL_NAME,
    web_fetch_tool_definition,
    web_search_tool_definition,
)
from gateway.services.tools._web_provider_client import build_web_provider_client
from gateway.services.tools._web_search import (
    CHARS_PER_TOKEN,
    DEFAULT_MAX_TOKENS_PER_PAGE,
    DIRECT_DEFAULT_MAX_RESULTS,
    DirectSearch,
    DirectSearchResult,
    ResolvedFetch,
    ResolvedSearch,
    SearchQuery,
    filter_hits_by_domain,
    page_budget_chars,
    pick_direct_search_instance,
    resolve_direct_search,
    resolve_fetch,
    resolve_search,
    run_direct_search,
)
from gateway.services.tools._web_search_keys import WebSearchKeyService, WorkspaceSearchKeys
from gateway.services.tools._web_search_responses import WEB_SEARCH_CALL_ID_PREFIX
from gateway.services.tools._web_search_results import DEFAULT_MAX_RESULTS, web_search_max_results_baseline
from gateway.services.tools._web_tool_failure import WebToolFailure
from gateway.services.tools._workspace_defaults import CodeExecutionWorkspaceDefaults

__all__ = [
    "MAX_TOOL_ITERATIONS_CAP",
    "CodeExecutionWorkspaceDefaults",
    "WorkspaceCodeExecutionPolicies",
    "WorkspaceCodeExecutionPolicyPublic",
    "WorkspaceCodeExecutionPolicyService",
    "WorkspaceCodeExecutionPolicyUpdate",
    "read_code_execution_policy",
    "BUILTIN_TOOLS",
    "CODE_EXECUTION_HEADER",
    "CODE_EXECUTION_TOOL",
    "CODE_INTERPRETER_CALL_ID_PREFIX",
    "CombinedToolBackend",
    "CONTAINER_GONE_DETAIL_TEMPLATE",
    "MAX_USES_EXCEEDED_ERROR",
    "OTARI_TOOL_CALLS_FIELD",
    "SERVER_TOOL_USE_ID_PREFIX",
    "WEB_SEARCH_CALL_ID_PREFIX",
    "WEB_FETCH_HEADER",
    "WEB_SEARCH_HEADER",
    "AdmittedCodeExecution",
    "DeclaredCodeExecution",
    "BuiltinTool",
    "CandidateTools",
    "ProviderToolDeclaration",
    "DeclaredWebTools",
    "Dialect",
    "NativeCall",
    "NativeRendering",
    "Tool",
    "ToolUseBudget",
    "WebAccessGrant",
    "WebClaims",
    "WebSearchKeyService",
    "WorkspaceSearchKeys",
    "admit_mcp_servers",
    "admit_web_access",
    "apply_web_access_policy",
    "code_execution_declaration_forms",
    "decide_code_executor",
    "decide_candidate_tools",
    "declare_code_execution",
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
    "read_web_declarations",
    "read_web_fetch_max_uses",
    "read_web_search_max_uses",
    "refuse_shadowed_builtin_tools",
    "resolve_code_executor_preference",
    "validate_web_declarations",
    "web_search_declaration_forms",
    "web_search_intercept_enabled",
    "DEFAULT_MAX_RESULTS",
    "web_search_max_results_baseline",
    "search_provider_catalog",
    "DefaultPin",
    "check_kind_unchanged",
    "check_test_input",
    "check_tool_entry",
    "check_tool_name_is_free",
    "check_tool_write",
    "default_to_pin",
    "instance_to_test",
    "run_connection_test",
    "unsaved_instance_to_test",
    "WEB_FETCH_TOOL_NAME",
    "WEB_RETRIEVAL_RESULT_MAX_BYTES",
    "WEB_SEARCH_NATIVE_TYPE_PREFIX",
    "WEB_SEARCH_TOOL_NAME",
    "web_fetch_tool_definition",
    "web_search_tool_definition",
    "ResolvedFetch",
    "ResolvedSearch",
    "resolve_fetch",
    "resolve_search",
    "CHARS_PER_TOKEN",
    "DEFAULT_MAX_TOKENS_PER_PAGE",
    "DIRECT_DEFAULT_MAX_RESULTS",
    "DirectSearch",
    "DirectSearchResult",
    "SearchQuery",
    "WebToolFailure",
    "without_otari_tool_calls",
    "build_web_provider_client",
    "filter_hits_by_domain",
    "owning_backend",
    "page_budget_chars",
    "pick_direct_search_instance",
    "resolve_direct_search",
    "run_direct_search",
]
