"""The email blocklist's match rule: how an entry is normalized and which entries an address meets."""

import pytest

from gateway.core.email_domains import blocklist_entry, blocklist_keys


@pytest.mark.parametrize(
    ("typed", "stored"),
    [
        ("Spam.Example", "spam.example"),
        ("@spam.example", "spam.example"),
        ("*.spam.example", "spam.example"),
        ("spam.example.", "spam.example"),
        ("  Person@Spam.Example ", "person@spam.example"),
        ("person@münchen.de", "person@xn--mnchen-3ya.de"),
        ("münchen.de", "xn--mnchen-3ya.de"),
        ("example.рф", "example.xn--p1ai"),
        ("person@example.рф", "person@example.xn--p1ai"),
    ],
)
def test_an_entry_is_normalized_to_the_form_it_is_stored_in(typed: str, stored: str) -> None:
    assert blocklist_entry(typed) == stored


@pytest.mark.parametrize(
    "typed",
    [
        "",
        "com",
        "not a domain",
        "a@@b.com",
        "@",
        "pers on@example.com",
        "http://x.com",
        "*.person@example.com",
        "@person@example.com",
        "example.xn--a-",
    ],
)
def test_a_value_that_is_neither_an_address_nor_a_domain_is_refused(typed: str) -> None:
    assert blocklist_entry(typed) is None


def test_an_address_meets_itself_and_every_domain_above_it_but_not_the_top_level_domain() -> None:
    assert blocklist_keys("Person@Mail.Spam.Example") == (
        "person@mail.spam.example",
        "mail.spam.example",
        "spam.example",
    )


def test_an_address_meets_an_entry_typed_in_either_spelling_of_its_domain() -> None:
    entry = blocklist_entry("münchen.de")
    assert entry in blocklist_keys("person@MÜNCHEN.de")
    assert entry in blocklist_keys("person@xn--mnchen-3ya.de")


def test_a_lookalike_domain_is_not_met() -> None:
    assert "spam.example" not in blocklist_keys("person@notspam.example")


def test_an_address_with_no_domain_meets_nothing() -> None:
    assert blocklist_keys("nobody") == ()
