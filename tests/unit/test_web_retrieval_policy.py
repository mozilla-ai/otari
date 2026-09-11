"""Focused tests for web-retrieval canonical identities and domain policy."""

import ipaddress

import pytest

from gateway.services.web_retrieval_policy import (
    CanonicalHost,
    DisjointDomainAllowListsError,
    DomainPolicy,
    DomainRuleValidationError,
    WebURLValidationError,
    canonicalize_domain_rule,
    canonicalize_domain_rules,
    canonicalize_host,
    canonicalize_web_url,
    domain_rule_matches,
    intersect_domain_allow_lists,
    resolve_redirect_url,
    union_domain_block_lists,
)


@pytest.mark.parametrize(
    ("raw", "expected_host", "expected_port", "expected_url"),
    [
        ("HTTPS://Example.COM/path", "example.com", 443, "https://example.com/path"),
        ("https://example.com:443/path", "example.com", 443, "https://example.com/path"),
        ("http://example.com:80/path", "example.com", 80, "http://example.com/path"),
        ("https://example.com:8443/path", "example.com", 8443, "https://example.com:8443/path"),
        ("https://example.com./path", "example.com", 443, "https://example.com/path"),
        ("http://[2001:0db8:0:0::1]:80/path", "2001:db8::1", 80, "http://[2001:db8::1]/path"),
        ("http://192.0.2.1./path", "192.0.2.1", 80, "http://192.0.2.1/path"),
    ],
)
def test_canonicalize_web_url_normalizes_only_origin(
    raw: str,
    expected_host: str,
    expected_port: int,
    expected_url: str,
) -> None:
    result = canonicalize_web_url(raw)

    assert result.origin.host.value == expected_host
    assert result.origin.port == expected_port
    assert str(result.url) == expected_url


def test_unicode_and_punycode_hosts_have_one_identity() -> None:
    unicode_url = canonicalize_web_url("https://BÜCHER.example./article")
    punycode_url = canonicalize_web_url("https://xn--bcher-kva.example/article")

    assert unicode_url.origin == punycode_url.origin
    assert unicode_url.redirect_loop_key == punycode_url.redirect_loop_key
    assert unicode_url.origin.host.value == "xn--bcher-kva.example"


@pytest.mark.parametrize("separator", ["。", "．", "｡"])
def test_uts46_terminal_separators_have_one_policy_identity(separator: str) -> None:
    blocked = canonicalize_domain_rule(f"example.com{separator}")
    plain_destination = canonicalize_web_url("https://example.com/path").origin.host
    mapped_destination = canonicalize_web_url(f"https://example.com{separator}/path").origin.host

    assert blocked.value == "example.com"
    assert plain_destination == mapped_destination
    assert domain_rule_matches(blocked, plain_destination)


@pytest.mark.parametrize("separator", ["。", "．", "｡"])
def test_uts46_mapped_ip_literal_is_reclassified(separator: str) -> None:
    rule = canonicalize_domain_rule(f"192.0.2.1{separator}")

    assert rule.value == "192.0.2.1"
    assert rule.is_ip


def test_signed_path_and_query_serialization_is_preserved() -> None:
    result = canonicalize_web_url("https://EXAMPLE.com./a%2Fb//c?b=2&a=%2F&a=3#client-only")

    assert result.url.raw_path == b"/a%2Fb//c?b=2&a=%2F&a=3"
    assert str(result.url) == "https://example.com/a%2Fb//c?b=2&a=%2F&a=3"
    assert result.display_url == "https://example.com/a%2Fb//c"


def test_fragment_does_not_affect_redirect_loop_identity() -> None:
    first = canonicalize_web_url("https://example.com/path?q=1#first")
    second = canonicalize_web_url("https://example.com/path?q=1#second")

    assert first.redirect_loop_key == second.redirect_loop_key


def test_relative_redirect_preserves_signed_path_and_query() -> None:
    current = canonicalize_web_url("https://EXAMPLE.com./one/start?old=1")

    redirected = resolve_redirect_url(current, "../a%2Fb?signature=z%2Fz&part=2#client")

    assert str(redirected.url) == "https://example.com/a%2Fb?signature=z%2Fz&part=2"
    assert redirected.url.raw_path == b"/a%2Fb?signature=z%2Fz&part=2"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "/relative",
        "ftp://example.com/file",
        "https:///missing-host",
        "https://user@example.com/path",
        "https://user:secret@example.com/path",
        "https://example.com../path",
        "https://bad..example/path",
        "https://%65xample.com/path",
        "https://[fe80::1%25en0]/path",
        "https://example.com/line\nbreak",
        "https://example.com:0/path",
        "https://example.com:70000/path",
    ],
)
def test_invalid_web_urls_are_rejected(value: str) -> None:
    with pytest.raises(WebURLValidationError):
        canonicalize_web_url(value)


def test_url_length_limit_is_enforced_before_parsing() -> None:
    with pytest.raises(WebURLValidationError, match="8192"):
        canonicalize_web_url("https://example.com/" + "a" * 8192)


@pytest.mark.parametrize(
    ("value", "expected", "is_ip"),
    [
        ("Example.COM", "example.com", False),
        ("example.com.", "example.com", False),
        ("BÜCHER.example", "xn--bcher-kva.example", False),
        ("xn--bcher-kva.example", "xn--bcher-kva.example", False),
        ("192.0.2.1", "192.0.2.1", True),
        ("192.0.2.1.", "192.0.2.1", True),
        ("2001:0db8:0:0::1", "2001:db8::1", True),
        ("[2001:0db8::1]", "2001:db8::1", True),
    ],
)
def test_domain_rule_canonicalization(value: str, expected: str, is_ip: bool) -> None:
    rule = canonicalize_domain_rule(value)

    assert rule.value == expected
    assert rule.is_ip is is_ip


@pytest.mark.parametrize(
    "value",
    [
        "",
        ".",
        "example.com..",
        "bad..example",
        "https://example.com",
        "example.com/path",
        "*.example.com",
        "user@example.com",
        "example.com:443",
        "[fe80::1%25eth0]",
        "[2001:db8::1",
    ],
)
def test_invalid_domain_rules_are_rejected(value: str) -> None:
    with pytest.raises(DomainRuleValidationError):
        canonicalize_domain_rule(value)


def test_domain_rule_deduplication_uses_canonical_identity() -> None:
    rules = canonicalize_domain_rules(("BÜCHER.example", "xn--bcher-kva.example", "EXAMPLE.com."))

    assert [rule.value for rule in rules] == ["xn--bcher-kva.example", "example.com"]


def test_dns_rule_matches_named_host_and_subdomains() -> None:
    rule = canonicalize_domain_rule("example.com")

    assert domain_rule_matches(rule, canonicalize_host("example.com"))
    assert domain_rule_matches(rule, canonicalize_host("www.api.example.com"))
    assert not domain_rule_matches(rule, canonicalize_host("notexample.com"))


def test_ip_rule_matches_only_exact_canonical_literal() -> None:
    rule = canonicalize_domain_rule("2001:0db8::1")

    assert domain_rule_matches(rule, canonicalize_host("2001:db8::1"))
    assert not domain_rule_matches(rule, canonicalize_host("2001:db8::2"))
    assert not domain_rule_matches(rule, canonicalize_host("example.com"))


def test_allowed_intersection_retains_narrower_dns_scopes() -> None:
    workspace = canonicalize_domain_rules(("example.com", "mozilla.org"))
    request = canonicalize_domain_rules(("docs.example.com", "developer.mozilla.org"))

    result = intersect_domain_allow_lists(workspace, request)

    assert result is not None
    assert [rule.value for rule in result] == ["docs.example.com", "developer.mozilla.org"]


def test_allowed_intersection_is_symmetric_for_parent_child_relationship() -> None:
    parent = canonicalize_domain_rules(("example.com",))
    child = canonicalize_domain_rules(("api.example.com",))

    assert intersect_domain_allow_lists(parent, child) == child
    assert intersect_domain_allow_lists(child, parent) == child


def test_ip_allow_intersection_requires_exact_address() -> None:
    first = canonicalize_domain_rules(("192.0.2.1",))
    same = canonicalize_domain_rules(("192.0.2.1",))
    other = canonicalize_domain_rules(("192.0.2.2",))

    assert intersect_domain_allow_lists(first, same) == first
    with pytest.raises(DisjointDomainAllowListsError):
        intersect_domain_allow_lists(first, other)


def test_disjoint_nonempty_allow_lists_fail_explicitly() -> None:
    first = canonicalize_domain_rules(("example.com",))
    second = canonicalize_domain_rules(("mozilla.org",))

    with pytest.raises(DisjointDomainAllowListsError):
        intersect_domain_allow_lists(first, second)


def test_absent_and_empty_allow_lists_do_not_accidentally_exclude_rules() -> None:
    rules = canonicalize_domain_rules(("example.com",))

    assert intersect_domain_allow_lists(None, rules) == rules
    assert intersect_domain_allow_lists(rules, None) == rules
    assert intersect_domain_allow_lists((), rules) == rules
    assert intersect_domain_allow_lists(rules, ()) == rules


def test_block_union_is_canonical_stable_and_deduplicated() -> None:
    first = canonicalize_domain_rules(("EXAMPLE.com", "192.0.2.1"))
    second = canonicalize_domain_rules(("example.com.", "spam.example"))

    result = union_domain_block_lists(first, second)

    assert [rule.value for rule in result] == ["example.com", "192.0.2.1", "spam.example"]


def test_domain_policy_applies_block_precedence() -> None:
    policy = DomainPolicy(
        allowed=canonicalize_domain_rules(("example.com",)),
        blocked=canonicalize_domain_rules(("private.example.com",)),
    )

    assert policy.permits(canonicalize_host("public.example.com"))
    assert not policy.permits(canonicalize_host("private.example.com"))
    assert not policy.permits(canonicalize_host("child.private.example.com"))
    assert not policy.permits(canonicalize_host("mozilla.org"))


def test_canonical_host_ip_value_uses_standard_library_identity() -> None:
    host = canonicalize_host("2001:0db8::1")

    assert host == CanonicalHost("2001:db8::1", ipaddress.ip_address("2001:db8::1"))
