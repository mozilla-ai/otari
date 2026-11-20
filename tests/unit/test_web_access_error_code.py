"""The workspace-switch refusals carry a stable error code and say where the switch is."""

from gateway.core.error_codes import ERROR_CODE_HEADER, WEB_ACCESS_DISABLED, error_headers
from gateway.exceptions.tools_exceptions import (
    WebAccessNotEnabledError,
    WebAccessToolNotAuthorizedError,
    WebSearchNotEnabledError,
)
from gateway.services.tools._code_execution_admission import SANDBOX_NOT_ENABLED_DETAIL


def test_web_access_not_enabled_carries_the_code() -> None:
    assert WebAccessNotEnabledError().error_code == WEB_ACCESS_DISABLED == "web_access_disabled"


def test_web_search_not_enabled_carries_the_code() -> None:
    assert WebSearchNotEnabledError().error_code == WEB_ACCESS_DISABLED


def test_not_authorized_is_a_distinct_refusal_without_the_disabled_code() -> None:
    assert WebAccessToolNotAuthorizedError().error_code is None


def test_error_headers_surface_the_code() -> None:
    assert error_headers(WEB_ACCESS_DISABLED)[ERROR_CODE_HEADER] == "web_access_disabled"


def test_the_refusals_name_where_the_workspace_allows_the_tool() -> None:
    assert WebSearchNotEnabledError().message == (
        "web search is not enabled for this workspace. "
        "An owner or admin of the workspace allows it under Tools > Web search in the dashboard."
    )
    assert WebAccessNotEnabledError().message == (
        "web access is not enabled for this workspace. "
        "An owner or admin of the workspace allows it under Tools > Web search in the dashboard."
    )
    assert SANDBOX_NOT_ENABLED_DETAIL == (
        "code execution is not enabled for this workspace. "
        "An owner or admin of the workspace allows it under Tools > Code execution in the dashboard."
    )
