"""The domain normalizer a claim is stored through and matched by.

Pure string work with no database. It gets its own unit test rather than being
covered only through the API because the two callers have to agree exactly:
`organization_domain_service` stores what ``normalized_domain`` returns, and
every later sign-in matches it against what ``email_domain`` returns. A
disagreement between them stores a claim that silently matches nobody, which no
error surfaces and no operator can see.
"""

from gateway.core.email_domains import (
    PUBLIC_EMAIL_DOMAINS,
    email_domain,
    is_public_email_domain,
    is_registrable_domain,
    normalized_domain,
)


def test_the_domain_of_an_address_is_lowercased_and_unqualified() -> None:
    assert email_domain("Person@Example.COM") == "example.com"
    # An RFC-legal absolute form. An admin claims "example.com" without the dot,
    # so keeping it here would make the two forms fail to match.
    assert email_domain("person@example.com.") == "example.com"
    assert email_domain("  person@example.com  ") == "example.com"


def test_only_the_last_at_separates_the_domain() -> None:
    """Plus-addressing and quoted local parts must not shift the split."""
    assert email_domain("first+tag@example.com") == "example.com"
    assert email_domain('"odd@local"@example.com') == "example.com"


def test_an_address_with_no_usable_domain_is_no_match_rather_than_an_error() -> None:
    assert email_domain("not-an-address") is None
    assert email_domain("trailing@") is None
    assert email_domain("trailing@.") is None


def test_a_claim_may_be_typed_as_a_domain_or_as_a_whole_address() -> None:
    """Pasting one's own address into the field claims the right thing."""
    assert normalized_domain("Example.com") == "example.com"
    assert normalized_domain("me@Example.com") == "example.com"
    assert normalized_domain("  ME@EXAMPLE.COM.  ") == "example.com"


def test_the_two_normalizers_agree_on_the_forms_an_admin_might_type() -> None:
    """The invariant the whole feature rests on, stated once."""
    for typed, signed_in_as in [
        ("example.com", "person@example.com"),
        ("Example.COM", "Person@Example.com"),
        ("me@example.com.", "colleague@example.com"),
        ("  sub.example.com  ", "person@SUB.example.com"),
    ]:
        assert normalized_domain(typed) == email_domain(signed_in_as)


def test_an_internationalized_domain_lands_on_one_spelling_from_either_side() -> None:
    """The failure this module exists to prevent, in its sharpest form.

    Without punycoding, an admin typing ``münchen.de`` is refused (the shape
    check is ASCII-only), retypes the punycode, and gets a claim that verifies
    by DNS and matches nobody: every colleague signs in at ``münchen.de``, which
    normalized to itself. A verified, active claim admitting nobody, with no
    error anywhere to explain it.
    """
    assert normalized_domain("münchen.de") == "xn--mnchen-3ya.de"
    assert normalized_domain("MÜNCHEN.de") == "xn--mnchen-3ya.de"
    # Already-punycoded input is left where it is, so both spellings are one claim.
    assert normalized_domain("xn--mnchen-3ya.de") == "xn--mnchen-3ya.de"
    assert email_domain("person@münchen.de") == "xn--mnchen-3ya.de"
    assert email_domain("person@MÜNCHEN.de.") == "xn--mnchen-3ya.de"
    assert normalized_domain("münchen.de") == email_domain("person@münchen.de")


def test_an_internationalized_claim_passes_the_shape_check_once_encoded() -> None:
    """The other half: it must also be claimable, not merely consistent."""
    assert is_registrable_domain(normalized_domain("münchen.de"))
    assert is_registrable_domain(normalized_domain("bücher.example"))


def test_a_domain_the_codec_refuses_is_passed_through_rather_than_raised_on() -> None:
    """An unencodable value is refused as a claim and matches nothing on sign-in."""
    assert normalized_domain("a..b") == "a..b"
    assert not is_registrable_domain(normalized_domain("a..b"))
    assert email_domain("person@a..b") == "a..b"


def test_a_registrable_domain_needs_labels_and_an_alphabetic_tld() -> None:
    assert is_registrable_domain("example.com")
    assert is_registrable_domain("sub.example.co.uk")
    assert not is_registrable_domain("example")
    assert not is_registrable_domain("example.")
    assert not is_registrable_domain(".example.com")
    assert not is_registrable_domain("-bad.example.com")
    assert not is_registrable_domain("example.c0m")
    assert not is_registrable_domain("https://example.com")
    assert not is_registrable_domain("example.com/path")
    assert not is_registrable_domain("")


def test_a_registrable_domain_is_bounded_at_the_dns_name_limit() -> None:
    label = "a" * 63
    assert is_registrable_domain(f"{label}.com")
    assert not is_registrable_domain(f"{'a' * 64}.com")
    assert not is_registrable_domain(".".join([label] * 4) + ".com")


def test_public_providers_are_refused_including_under_a_subdomain() -> None:
    """Prefixing a label must not be enough to claim a provider's population."""
    assert is_public_email_domain("gmail.com")
    assert is_public_email_domain("GMAIL.COM")
    assert is_public_email_domain("mail.gmail.com")
    assert is_public_email_domain("a.b.yahoo.co.uk")


def test_a_company_domain_is_not_mistaken_for_a_public_one() -> None:
    assert not is_public_email_domain("example.com")
    # Ends with a blocked domain's characters but is a different registrable
    # domain: the check compares whole labels, not string suffixes.
    assert not is_public_email_domain("notgmail.com")
    assert not is_public_email_domain("gmail.com.example.org")


def test_every_blocked_provider_is_stored_in_the_form_the_check_compares() -> None:
    """A capitalized or dotted entry would sit in the set and never match."""
    for domain in PUBLIC_EMAIL_DOMAINS:
        assert domain == domain.lower().strip(".")
        assert is_registrable_domain(domain), domain
        assert is_public_email_domain(domain)
