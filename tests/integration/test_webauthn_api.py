"""Passkey registration, sign-in and management, end to end against PostgreSQL.

Every ceremony here is a real one: ``webauthn_helpers.SoftwareAuthenticator``
builds the payloads a browser and an authenticator produce and signs them with a
P-256 key, and py_webauthn verifies them unmodified. Nothing in
``webauthn_service`` is stubbed, so a test that passes describes a ceremony that
would pass in a browser and a refusal that would happen there too.

The relying party is pinned per test rather than in the shared config fixture.
``TestClient`` serves ``http://testserver``, so ``public_base_url`` is set to
that and the relying-party ID derives to ``testserver``; a test that wants a
different one sets it explicitly. Both are plain attributes read through
``GatewayConfig.webauthn_relying_party``, a property, so monkeypatching them
after the app has booted is enough.
"""

import logging
import re
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME

from .webauthn_helpers import SoftwareAuthenticator, challenge_of

ORIGIN = "http://testserver"
RP_ID = "testserver"
PASSWORD = "a-real-password"  # pragma: allowlist secret

_TOKEN_IN_LINK = re.compile(r"token=([\w-]+)")


@pytest.fixture
def passkeys_configured(test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the deployment's relying party at the address TestClient serves."""
    monkeypatch.setattr(test_config, "public_base_url", ORIGIN)


@pytest.fixture
def authenticator() -> SoftwareAuthenticator:
    return SoftwareAuthenticator(rp_id=RP_ID, origin=ORIGIN)


def _register(
    client: TestClient,
    headers: dict[str, str],
    authenticator: SoftwareAuthenticator,
    *,
    name: str | None = None,
) -> dict[str, Any]:
    """Run a whole registration ceremony and return the stored passkey."""
    options = client.post("/v1/auth/webauthn/register/options", headers=headers)
    assert options.status_code == 200, options.text
    response = authenticator.register(challenge_of(options.json()))
    body: dict[str, Any] = {"credential": response}
    if name is not None:
        body["name"] = name
    created = client.post("/v1/auth/webauthn/register", json=body, headers=headers)
    assert created.status_code == 201, created.text
    result: dict[str, Any] = created.json()
    return result


def _sign_in(client: TestClient, authenticator: SoftwareAuthenticator, **kwargs: Any) -> Any:
    """Run a whole sign-in ceremony and return the raw response."""
    options = client.post("/v1/auth/webauthn/authenticate/options")
    assert options.status_code == 200, options.text
    assertion = authenticator.authenticate(challenge_of(options.json()), **kwargs)
    return client.post("/v1/auth/webauthn/authenticate", json={"credential": assertion})


def test_a_passkey_registers_and_then_signs_in(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    authenticator: SoftwareAuthenticator,
    passkeys_configured: None,
) -> None:
    """The definition of done: a passkey registers, and it authenticates."""
    # Who the master key resolves to, captured before the passkey exists, so the
    # sign-in below can be shown to resolve the same identity rather than merely
    # some identity.
    bootstrap = client.post("/v1/auth/session", json={"master_key": test_config.master_key})
    assert bootstrap.status_code == 200, bootstrap.text
    operator_id = bootstrap.json()["user_id"]
    client.cookies.clear()

    passkey = _register(client, master_key_header, authenticator, name="Work laptop")
    assert passkey["name"] == "Work laptop"
    assert passkey["rp_id"] == RP_ID
    assert passkey["credential_id"] == authenticator.credential_id_b64
    assert passkey["transports"] == ["internal"]
    assert passkey["last_used_at"] is None
    assert passkey["is_usable"] is True
    # Key material never crosses the wire, in either direction.
    assert "public_key" not in passkey

    signed_in = _sign_in(client, authenticator)
    assert signed_in.status_code == 200, signed_in.text
    assert signed_in.json()["user_id"] == operator_id
    assert signed_in.json()["active_organization_id"]
    assert client.cookies.get(SESSION_COOKIE_NAME)

    # The cookie now authenticates on its own, with no header credential: the
    # passkey minted the same session a password would have.
    listed = client.get("/v1/auth/webauthn/credentials")
    assert listed.status_code == 200, listed.text
    assert listed.json()["count"] == 1
    assert listed.json()["data"][0]["last_used_at"] is not None


def test_passkeys_are_unavailable_when_the_deployment_has_no_address(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """No ``public_base_url`` and no ``webauthn_rp_id`` means no relying party.

    503 naming the setting, not a 500 and not a ceremony that starts and then
    fails inside the browser.
    """
    options = client.post("/v1/auth/webauthn/register/options", headers=master_key_header)
    assert options.status_code == 503, options.text
    assert "public_base_url" in options.json()["detail"]

    # Listing is deliberately not gated: a deployment that lost its relying
    # party still has to let somebody see and remove what it left behind.
    listed = client.get("/v1/auth/webauthn/credentials", headers=master_key_header)
    assert listed.status_code == 200, listed.text
    assert listed.json()["count"] == 0


def test_an_explicit_rp_id_overrides_the_derived_one(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A parent domain can be configured, and it is what the ceremony uses."""
    monkeypatch.setattr(test_config, "public_base_url", "http://sub.testserver")
    monkeypatch.setattr(test_config, "webauthn_rp_id", "testserver")
    options = client.post("/v1/auth/webauthn/register/options", headers=master_key_header)
    assert options.status_code == 200, options.text
    assert options.json()["rp"]["id"] == "testserver"


def test_a_challenge_is_spent_once(
    client: TestClient,
    master_key_header: dict[str, str],
    authenticator: SoftwareAuthenticator,
    passkeys_configured: None,
) -> None:
    """Replaying a whole assertion is refused: the challenge row is gone."""
    _register(client, master_key_header, authenticator)
    options = client.post("/v1/auth/webauthn/authenticate/options")
    assertion = authenticator.authenticate(challenge_of(options.json()))

    first = client.post("/v1/auth/webauthn/authenticate", json={"credential": assertion})
    assert first.status_code == 200, first.text

    replay = client.post("/v1/auth/webauthn/authenticate", json={"credential": assertion})
    assert replay.status_code == 401, replay.text


def test_a_failed_ceremony_leaves_its_challenge_spendable(
    client: TestClient,
    master_key_header: dict[str, str],
    authenticator: SoftwareAuthenticator,
    passkeys_configured: None,
) -> None:
    """A refusal rolls back, so the retry a person immediately makes still works.

    The challenge is retired by the *commit*, not by being read, which is what
    keeps a mistyped-PIN style failure from costing a fresh options round trip.
    """
    _register(client, master_key_header, authenticator)
    options = client.post("/v1/auth/webauthn/authenticate/options")
    challenge = challenge_of(options.json())

    from_elsewhere = authenticator.authenticate(challenge, origin="https://evil.example.com")
    refused = client.post("/v1/auth/webauthn/authenticate", json={"credential": from_elsewhere})
    assert refused.status_code == 401, refused.text

    honest = authenticator.authenticate(challenge)
    accepted = client.post("/v1/auth/webauthn/authenticate", json={"credential": honest})
    assert accepted.status_code == 200, accepted.text


def test_an_assertion_from_another_origin_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
    authenticator: SoftwareAuthenticator,
    passkeys_configured: None,
) -> None:
    """The origin is checked, which is what stops a phishing page replaying one."""
    _register(client, master_key_header, authenticator)
    refused = _sign_in(client, authenticator, origin="https://otari.example.evil")
    assert refused.status_code == 401, refused.text
    assert client.cookies.get(SESSION_COOKIE_NAME) is None


def test_an_assertion_signed_for_another_relying_party_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
    authenticator: SoftwareAuthenticator,
    passkeys_configured: None,
) -> None:
    """The RP-ID hash inside authenticatorData has to be this deployment's."""
    _register(client, master_key_header, authenticator)
    refused = _sign_in(client, authenticator, rp_id="otari.example.evil")
    assert refused.status_code == 401, refused.text


def test_an_unknown_credential_does_not_sign_in(
    client: TestClient,
    master_key_header: dict[str, str],
    authenticator: SoftwareAuthenticator,
    passkeys_configured: None,
) -> None:
    """A well-formed assertion from an authenticator nobody registered."""
    _register(client, master_key_header, authenticator)
    stranger = SoftwareAuthenticator(rp_id=RP_ID, origin=ORIGIN)
    refused = _sign_in(client, stranger)
    assert refused.status_code == 401, refused.text


def test_a_replayed_signature_counter_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
    authenticator: SoftwareAuthenticator,
    passkeys_configured: None,
) -> None:
    """A counter that goes backwards is the clone signal, and it is enforced.

    The stored counter is what makes the next check meaningful, so this asserts
    the update landed as much as it asserts the refusal.
    """
    _register(client, master_key_header, authenticator)
    first = _sign_in(client, authenticator, sign_count=5)
    assert first.status_code == 200, first.text

    cloned = _sign_in(client, authenticator, sign_count=3)
    assert cloned.status_code == 401, cloned.text


def test_a_credential_registered_under_another_relying_party_is_inert(
    client: TestClient,
    master_key_header: dict[str, str],
    authenticator: SoftwareAuthenticator,
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
    passkeys_configured: None,
) -> None:
    """Moving the relying-party ID orphans existing passkeys, visibly.

    This is otari-ai#1716's standing constraint in miniature: a row is bound to
    the ID it was registered under, so a deployment that moves loses them.

    The orphan stays *listed*, marked unusable. Hiding it would leave somebody
    looking at an empty page with no explanation, and would withhold the id they
    need in order to delete the row.
    """
    passkey = _register(client, master_key_header, authenticator)
    assert passkey["is_usable"] is True

    monkeypatch.setattr(test_config, "webauthn_rp_id", "moved.testserver")
    monkeypatch.setattr(test_config, "public_base_url", "http://moved.testserver")

    listed = client.get("/v1/auth/webauthn/credentials", headers=master_key_header)
    assert listed.status_code == 200, listed.text
    assert listed.json()["count"] == 1
    assert listed.json()["data"][0]["is_usable"] is False

    # And the one action left for it still works.
    assert (
        client.delete(f"/v1/auth/webauthn/credentials/{passkey['id']}", headers=master_key_header).status_code == 204
    )


def test_a_passkey_cannot_assert_under_a_relying_party_it_was_not_registered_for(
    client: TestClient,
    master_key_header: dict[str, str],
    authenticator: SoftwareAuthenticator,
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
    passkeys_configured: None,
) -> None:
    """Even holding the same key pair, the row is scoped to the old ID."""
    _register(client, master_key_header, authenticator)
    monkeypatch.setattr(test_config, "webauthn_rp_id", "moved.testserver")
    monkeypatch.setattr(test_config, "public_base_url", "http://moved.testserver")

    moved = SoftwareAuthenticator(
        rp_id="moved.testserver",
        origin="http://moved.testserver",
        credential_id=authenticator.credential_id,
        private_key=authenticator.private_key,
    )
    options = client.post("/v1/auth/webauthn/authenticate/options")
    assertion = moved.authenticate(challenge_of(options.json()))
    refused = client.post("/v1/auth/webauthn/authenticate", json={"credential": assertion})
    assert refused.status_code == 401, refused.text


def test_a_registration_challenge_cannot_be_answered_as_a_sign_in(
    client: TestClient,
    master_key_header: dict[str, str],
    authenticator: SoftwareAuthenticator,
    passkeys_configured: None,
) -> None:
    """The ceremony a challenge was issued for is the server's choice, not the caller's."""
    _register(client, master_key_header, authenticator)
    options = client.post("/v1/auth/webauthn/register/options", headers=master_key_header)
    assertion = authenticator.authenticate(challenge_of(options.json()))
    refused = client.post("/v1/auth/webauthn/authenticate", json={"credential": assertion})
    assert refused.status_code == 401, refused.text


def test_the_same_authenticator_cannot_register_twice(
    client: TestClient,
    master_key_header: dict[str, str],
    authenticator: SoftwareAuthenticator,
    passkeys_configured: None,
) -> None:
    """The unique index decides it, whatever the browser did with exclude_credentials."""
    _register(client, master_key_header, authenticator)
    options = client.post("/v1/auth/webauthn/register/options", headers=master_key_header)
    again = client.post(
        "/v1/auth/webauthn/register",
        json={"credential": authenticator.register(challenge_of(options.json()))},
        headers=master_key_header,
    )
    assert again.status_code == 409, again.text


def test_registration_options_exclude_what_this_identity_already_holds(
    client: TestClient,
    master_key_header: dict[str, str],
    authenticator: SoftwareAuthenticator,
    passkeys_configured: None,
) -> None:
    _register(client, master_key_header, authenticator)
    options = client.post("/v1/auth/webauthn/register/options", headers=master_key_header)
    excluded = [item["id"] for item in options.json()["excludeCredentials"]]
    assert excluded == [authenticator.credential_id_b64]


def test_sign_in_options_name_no_credentials(
    client: TestClient,
    master_key_header: dict[str, str],
    authenticator: SoftwareAuthenticator,
    passkeys_configured: None,
) -> None:
    """Publishing a list would make this endpoint an oracle; see begin_authentication."""
    _register(client, master_key_header, authenticator)
    options = client.post("/v1/auth/webauthn/authenticate/options")
    assert options.status_code == 200, options.text
    assert not options.json().get("allowCredentials")


def test_unnamed_passkeys_are_numbered_rather_than_refused(
    client: TestClient,
    master_key_header: dict[str, str],
    passkeys_configured: None,
) -> None:
    """The unique (user_id, name) constraint must not turn into a dead end."""
    first = _register(client, master_key_header, SoftwareAuthenticator(rp_id=RP_ID, origin=ORIGIN))
    second = _register(client, master_key_header, SoftwareAuthenticator(rp_id=RP_ID, origin=ORIGIN))
    assert first["name"] == "Passkey"
    assert second["name"] == "Passkey 2"


def test_registering_a_name_that_is_taken_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
    passkeys_configured: None,
) -> None:
    """A name the caller chose is refused rather than silently altered."""
    _register(client, master_key_header, SoftwareAuthenticator(rp_id=RP_ID, origin=ORIGIN), name="Laptop")
    options = client.post("/v1/auth/webauthn/register/options", headers=master_key_header)
    clash = client.post(
        "/v1/auth/webauthn/register",
        json={
            "credential": SoftwareAuthenticator(rp_id=RP_ID, origin=ORIGIN).register(challenge_of(options.json())),
            "name": "Laptop",
        },
        headers=master_key_header,
    )
    assert clash.status_code == 409, clash.text


def test_a_passkey_renames_and_deletes(
    client: TestClient,
    master_key_header: dict[str, str],
    authenticator: SoftwareAuthenticator,
    passkeys_configured: None,
) -> None:
    passkey = _register(client, master_key_header, authenticator, name="Old name")

    renamed = client.patch(
        f"/v1/auth/webauthn/credentials/{passkey['id']}",
        json={"name": "  New name  "},
        headers=master_key_header,
    )
    assert renamed.status_code == 200, renamed.text
    # Trimmed, so a name that differs only in whitespace is not a second name.
    assert renamed.json()["name"] == "New name"

    deleted = client.delete(f"/v1/auth/webauthn/credentials/{passkey['id']}", headers=master_key_header)
    assert deleted.status_code == 204, deleted.text
    assert client.get("/v1/auth/webauthn/credentials", headers=master_key_header).json()["count"] == 0

    # And it no longer signs anybody in.
    assert _sign_in(client, authenticator).status_code == 401


def test_renaming_onto_another_passkeys_name_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
    passkeys_configured: None,
) -> None:
    _register(client, master_key_header, SoftwareAuthenticator(rp_id=RP_ID, origin=ORIGIN), name="Laptop")
    other = _register(client, master_key_header, SoftwareAuthenticator(rp_id=RP_ID, origin=ORIGIN), name="Phone")

    clash = client.patch(
        f"/v1/auth/webauthn/credentials/{other['id']}",
        json={"name": "Laptop"},
        headers=master_key_header,
    )
    assert clash.status_code == 409, clash.text


def test_a_blank_rename_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
    authenticator: SoftwareAuthenticator,
    passkeys_configured: None,
) -> None:
    passkey = _register(client, master_key_header, authenticator)
    blank = client.patch(
        f"/v1/auth/webauthn/credentials/{passkey['id']}",
        json={"name": "   "},
        headers=master_key_header,
    )
    assert blank.status_code == 422, blank.text


def test_an_unknown_passkey_is_a_404(
    client: TestClient,
    master_key_header: dict[str, str],
    passkeys_configured: None,
) -> None:
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.delete(f"/v1/auth/webauthn/credentials/{missing}", headers=master_key_header).status_code == 404


def test_managing_passkeys_needs_a_credential(
    client: TestClient,
    passkeys_configured: None,
) -> None:
    """The management half is not public; only the two sign-in calls are."""
    assert client.get("/v1/auth/webauthn/credentials").status_code == 401
    assert client.post("/v1/auth/webauthn/register/options").status_code == 401


def test_bootstrap_publishes_passkey_only_once_one_can_answer(
    client: TestClient,
    master_key_header: dict[str, str],
    authenticator: SoftwareAuthenticator,
    passkeys_configured: None,
) -> None:
    """A sign-in button whose only outcome is "no passkey found" is not offered."""
    before = client.get("/v1/bootstrap")
    assert "passkey" not in before.json()["sign_in_methods"]

    _register(client, master_key_header, authenticator)

    after = client.get("/v1/bootstrap")
    assert "passkey" in after.json()["sign_in_methods"]
    # Additive: it appears beside the credential the deployment already took,
    # rather than replacing it.
    assert "master_key" in after.json()["sign_in_methods"]


def test_bootstrap_omits_passkey_when_the_deployment_is_not_configured(
    client: TestClient,
) -> None:
    assert "passkey" not in client.get("/v1/bootstrap").json()["sign_in_methods"]


def test_one_identitys_registration_challenge_is_not_another_identitys(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    passkeys_configured: None,
) -> None:
    """A registration challenge names its identity, and that is checked on the way back.

    Without the check, a caller who can obtain *any* registration challenge
    could complete it against their own session and attach an authenticator to
    somebody else's identity.
    """
    monkeypatch.setattr(test_config, "mail_transport", "console")

    # A second identity, signed in with its own password.
    added = client.post(
        "/v1/organizations/me/members",
        json={"email": "ada@example.com", "role": "member"},
        headers=master_key_header,
    )
    assert added.status_code == 201, added.text
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="gateway")
    try:
        signed_up = client.post("/v1/auth/signup", json={"email": "ada@example.com", "password": PASSWORD})
    finally:
        gateway_logger.removeHandler(caplog.handler)
    assert signed_up.status_code == 200, signed_up.text
    token = _TOKEN_IN_LINK.search(caplog.text)
    assert token, caplog.text
    assert client.post("/v1/auth/verify-email", json={"token": token.group(1)}).status_code == 200

    # The operator's challenge, taken while they were the caller.
    operators = client.post("/v1/auth/webauthn/register/options", headers=master_key_header)
    assert operators.status_code == 200, operators.text
    challenge = challenge_of(operators.json())

    # Now Ada signs in and answers it as herself.
    session = client.post("/v1/auth/session", json={"email": "ada@example.com", "password": PASSWORD})
    assert session.status_code == 200, session.text
    stolen = SoftwareAuthenticator(rp_id=RP_ID, origin=ORIGIN).register(challenge)
    refused = client.post("/v1/auth/webauthn/register", json={"credential": stolen})
    assert refused.status_code == 400, refused.text
