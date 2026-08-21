"""The setup guide's failure classification, which is pure and worth pinning.

The category is the only thing about a failure that crosses the wire, because
the dashboard writes its own sentence per category rather than showing the
provider's error text. So the mapping is the contract, and a status that quietly
changed category would change what an operator is told to go and fix.
"""

import pytest

from gateway.services.tenancy.workspace_activation_service import activation_error_category


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (400, "invalid_request"),
        (404, "invalid_request"),
        (422, "invalid_request"),
        # No pricing for the model is a configuration screen, while a budget, a
        # model allow-list or a rate limit is a policy one.
        (402, "configuration"),
        (403, "policy"),
        (429, "policy"),
        (502, "upstream"),
        (503, "upstream"),
        (504, "timeout"),
        (500, "internal"),
    ],
)
def test_each_recorded_status_maps_to_the_screen_that_fixes_it(status_code: int, expected: str) -> None:
    assert activation_error_category(status_code) == expected


def test_a_row_with_no_status_is_internal() -> None:
    """A stream that ended without usage data leaves no status behind."""
    assert activation_error_category(None) == "internal"


@pytest.mark.parametrize("status_code", [409, 413, 418])
def test_an_unmapped_client_error_is_the_caller_s_request(status_code: int) -> None:
    assert activation_error_category(status_code) == "invalid_request"


@pytest.mark.parametrize("status_code", [501, 507, 599])
def test_an_unmapped_server_error_is_internal(status_code: int) -> None:
    assert activation_error_category(status_code) == "internal"
