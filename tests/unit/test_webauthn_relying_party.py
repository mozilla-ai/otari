"""How a deployment decides which relying party its passkeys are bound to.

The derivation is the part of otari#652 that is explicit by requirement rather
than by taste: a passkey is scoped by the authenticator to the relying-party ID
it was created under, so an ID that moves silently orphans every credential
registered before it. These are the rules that keep it from moving by accident,
and the refusals that catch a pair of settings no browser would ever accept.

Pure: nothing here touches a database or a request.
"""

import pytest

from gateway.core.config import GatewayConfig, RelyingParty


def _config(**overrides: object) -> GatewayConfig:
    return GatewayConfig(master_key="test-master-key", **overrides)  # type: ignore[arg-type]


def test_the_relying_party_id_derives_from_the_deployments_own_address() -> None:
    relying_party = _config(public_base_url="https://otari.example.com").webauthn_relying_party
    assert relying_party == RelyingParty(
        rp_id="otari.example.com", name="otari", origins=("https://otari.example.com",)
    )


def test_the_port_and_scheme_are_dropped_from_the_id_but_not_the_origin() -> None:
    """A relying-party ID is a bare domain; an origin is not.

    This is what makes passkeys work against a local development server:
    'localhost' is the one host browsers treat as a secure context without TLS.
    """
    relying_party = _config(public_base_url="http://localhost:8000").webauthn_relying_party
    assert relying_party is not None
    assert relying_party.rp_id == "localhost"
    assert relying_party.origins == ("http://localhost:8000",)


def test_a_trailing_slash_does_not_become_part_of_the_origin() -> None:
    relying_party = _config(public_base_url="https://otari.example.com/").webauthn_relying_party
    assert relying_party is not None
    assert relying_party.origins == ("https://otari.example.com",)


def test_an_explicit_id_wins_over_the_derived_one() -> None:
    """Binding to a parent domain, so one passkey works across subdomains."""
    relying_party = _config(
        public_base_url="https://otari.example.com",
        webauthn_rp_id="example.com",
    ).webauthn_relying_party
    assert relying_party is not None
    assert relying_party.rp_id == "example.com"
    assert relying_party.origins == ("https://otari.example.com",)


def test_a_configured_id_is_normalized_to_lowercase() -> None:
    """A domain is case-insensitive, and the comparisons around it are not.

    Left as written, a mixed-case ID passes validation (which lowercases both
    sides) and then fails ``covers`` against the lowercased origin host, so
    startup blames the origins list for a casing mistake in the ID.
    """
    config = _config(
        public_base_url="https://Otari.Example.com",
        webauthn_rp_id="Otari.Example.COM",
    )
    relying_party = config.webauthn_relying_party
    assert relying_party is not None
    assert relying_party.rp_id == "otari.example.com"
    # And the pair that used to be refused now validates.
    config.validate_webauthn_relying_party()


def test_a_deployment_that_knows_no_address_has_no_relying_party() -> None:
    """Not an error: it is a deployment that does not offer passkeys."""
    config = _config()
    assert config.webauthn_relying_party is None
    assert config.webauthn_enabled is False
    # And it is not refused at startup, because nothing was configured wrongly.
    config.validate_webauthn_relying_party()


def test_an_id_with_no_address_to_serve_it_from_is_not_a_relying_party() -> None:
    """``expected_origin`` has no safe default, so guessing one is not offered."""
    assert _config(webauthn_rp_id="example.com").webauthn_relying_party is None


def test_explicit_origins_replace_the_derived_one() -> None:
    relying_party = _config(
        public_base_url="https://otari.example.com",
        webauthn_rp_id="example.com",
        webauthn_allowed_origins=["https://otari.example.com", "https://admin.example.com/"],
    ).webauthn_relying_party
    assert relying_party is not None
    assert relying_party.origins == ("https://otari.example.com", "https://admin.example.com")


@pytest.mark.parametrize(
    ("rp_id", "origin", "covered"),
    [
        ("example.com", "https://example.com", True),
        ("example.com", "https://otari.example.com", True),
        ("example.com", "https://deep.otari.example.com", True),
        # The boundary: a suffix match without the dot would accept this.
        ("example.com", "https://notexample.com", False),
        ("otari.example.com", "https://example.com", False),
        ("example.com", "https://example.com.evil.test", False),
    ],
)
def test_which_origins_a_relying_party_id_covers(rp_id: str, origin: str, covered: bool) -> None:
    assert RelyingParty(rp_id=rp_id, name="otari", origins=()).covers(origin) is covered


@pytest.mark.parametrize(
    ("written", "suggested"),
    [
        ("https://example.com", "example.com"),
        # The commonest mistake, and the one a naive parse gets wrong: without
        # the "//" this reads as a scheme and a path, not a host and a port.
        ("localhost:8000", "localhost"),
        ("example.com/path", "example.com"),
        ("https://otari.example.com:8443/", "otari.example.com"),
    ],
)
def test_an_id_that_is_not_a_bare_domain_is_refused_with_the_right_suggestion(
    written: str, suggested: str
) -> None:
    """The refusal has to name the value to use, or it is a puzzle rather than a fix."""
    config = _config(public_base_url="https://example.com", webauthn_rp_id=written)
    with pytest.raises(ValueError, match="bare domain") as refusal:
        config.validate_webauthn_relying_party()
    assert repr(suggested) in str(refusal.value)


def test_a_scheme_less_origin_is_refused_by_naming_the_scheme() -> None:
    """The refusal has to blame the right setting.

    A bare domain fails the coverage check too, but 'otari.example.com' really
    is a subdomain of 'example.com', so the coverage message would send an
    operator to widen ``webauthn_rp_id``, which is not what is wrong.
    """
    config = _config(
        public_base_url="https://example.com",
        webauthn_rp_id="example.com",
        webauthn_allowed_origins=["otari.example.com"],
    )
    with pytest.raises(ValueError, match="missing a scheme") as refusal:
        config.validate_webauthn_relying_party()
    assert "webauthn_rp_id" not in str(refusal.value)


def test_an_origin_the_id_cannot_cover_is_refused_at_startup() -> None:
    """The plausible-looking pair that can never complete a ceremony."""
    config = _config(
        public_base_url="https://example.com",
        webauthn_rp_id="example.com",
        webauthn_allowed_origins=["https://otari.example.net"],
    )
    with pytest.raises(ValueError, match="not the relying-party ID"):
        config.validate_webauthn_relying_party()


def test_origins_with_no_id_to_check_them_against_are_refused_at_startup() -> None:
    config = _config(webauthn_allowed_origins=["https://otari.example.com"])
    with pytest.raises(ValueError, match="no relying-party ID"):
        config.validate_webauthn_relying_party()


def test_a_well_formed_configuration_passes_startup_validation() -> None:
    config = _config(
        public_base_url="https://otari.example.com",
        webauthn_rp_id="example.com",
        webauthn_allowed_origins=["https://otari.example.com", "https://admin.example.com"],
    )
    config.validate_webauthn_relying_party()
    assert config.webauthn_enabled is True
