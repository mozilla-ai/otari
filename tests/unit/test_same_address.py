"""Whether two values are the same address."""

import pytest

from gateway.core.addresses import same_address


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("ada@example.com", "ada@example.com"),
        ("ada@example.com", "Ada@Example.com"),
        ("ada@example.com", "  ada@example.com  "),
    ],
)
def test_case_and_surrounding_space_are_the_same_address(first: str, second: str) -> None:
    assert same_address(first, second) is True


def test_two_different_addresses_are_not_the_same_address() -> None:
    assert same_address("ada@example.com", "bob@example.com") is False


@pytest.mark.parametrize(
    "value",
    [
        "not an address",
        "Ada <ada@example.com>",
        "ada@example.com, bob@example.com",
        "",
    ],
)
def test_a_value_that_is_not_an_address_matches_nothing_including_itself(value: str) -> None:
    """Both sides normalize to nothing, and nothing must not match nothing."""
    assert same_address(value, value) is False
    assert same_address(value, "ada@example.com") is False
    assert same_address("ada@example.com", value) is False
