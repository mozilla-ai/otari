"""Unit tests for the coarse error-class mapping in the usage summary."""

from __future__ import annotations

import pytest

from gateway.api.routes.usage import error_class_for


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (402, "pricing"),
        (429, "rate_limit"),
        (401, "auth"),
        (403, "auth"),
        (407, "auth"),
        (400, "client_error"),
        (404, "client_error"),
        (422, "client_error"),
        (500, "provider_error"),
        (502, "provider_error"),
        (504, "provider_error"),
        (529, "provider_error"),
    ],
)
def test_status_codes_map_to_their_display_class(status_code: int, expected: str) -> None:
    assert error_class_for(status_code) == expected


def test_missing_code_is_unknown() -> None:
    # Rows written before the column existed, plus failures no HTTP status
    # describes, must land in a bucket rather than crash the breakdown.
    assert error_class_for(None) == "unknown"


@pytest.mark.parametrize("status_code", [0, 200, 302, 600, 999])
def test_out_of_range_codes_are_unknown(status_code: int) -> None:
    # status_code comes from provider exceptions, so it is not guaranteed to be a
    # sane 4xx/5xx. A garbage value must classify as unknown, never as a fault
    # class an operator would act on.
    assert error_class_for(status_code) == "unknown"
