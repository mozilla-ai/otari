"""The web-access refusals carry a stable error code (Otari#2990)."""

from gateway.core.error_codes import ERROR_CODE_HEADER, WEB_ACCESS_DISABLED, error_headers
from gateway.exceptions.tools_exceptions import (
    WebAccessNotEnabledError,
    WebAccessToolNotAuthorizedError,
    WebSearchNotEnabledError,
)


def test_web_access_not_enabled_carries_the_code() -> None:
    assert WebAccessNotEnabledError().error_code == WEB_ACCESS_DISABLED == "web_access_disabled"


def test_web_search_not_enabled_carries_the_code() -> None:
    assert WebSearchNotEnabledError().error_code == WEB_ACCESS_DISABLED


def test_not_authorized_is_a_distinct_refusal_without_the_disabled_code() -> None:
    assert WebAccessToolNotAuthorizedError().error_code is None


def test_error_headers_surface_the_code() -> None:
    assert error_headers(WEB_ACCESS_DISABLED)[ERROR_CODE_HEADER] == "web_access_disabled"
