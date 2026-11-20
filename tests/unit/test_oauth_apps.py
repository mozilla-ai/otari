"""Protocol boundaries independent of the dashboard."""

import uuid
from typing import get_args

import pytest
from apron_auth.pkce import generate_code_challenge as pkce_challenge
from pydantic import ValidationError

from gateway.core.oauth_apps import APP_KEY_LIFETIMES, open_app_site, redirect_matches
from gateway.core.settings.identity import IdentitySettings, OAuthAppConfig
from gateway.schemas.identity import AppAuthorizationRequest
from gateway.schemas.identity.oauth_apps import AppAuthorizationParameters, AppKeyExpiration, AppTokenRequest
from gateway.types.key_scope import DEFAULT_KEY_SCOPES, KeyScope


def test_rfc7636_s256_vector() -> None:
    assert (
        pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk") == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    )


@pytest.mark.parametrize(
    "uri",
    [
        "http://remote.example/cb",
        "https://u:p@example.com/cb",
        "https://example.com/cb#fragment",
        "https://example.com/cb?query=1",
        "https://example.com/cb?",
        "https://example.com/cb#",
        "https://@example.com/cb",
        "https://example.com:bad/cb",
        "https://example.com:999999/cb",
        "https://example.com:0/cb",
        "https://example.com:/cb",
        "https://exa\tmple.com/cb",
        " https://example.com/cb",
        "https://example.com/cb\n",
        "https://example.com\\@attacker.example/cb",
        "http://[broken/cb",
        "http://localhost%2eattacker.example/cb",
        "https://example.com/" + "x" * 2048,
        "javascript:alert(1)",
        "/relative",
        "",
    ],
)
def test_registration_rejects_unsafe_redirects(uri: str) -> None:
    with pytest.raises(ValidationError):
        OAuthAppConfig(name="App", redirect_uris=[uri])


def test_native_redirect_varies_only_loopback_port() -> None:
    assert redirect_matches("http://localhost/", "http://localhost:45000/")
    assert not redirect_matches("http://localhost/", "http://localhost.attacker.example/")
    assert not redirect_matches("http://localhost/", "http://localhost:45000/other")
    assert not redirect_matches("http://localhost/", "http://localhost:45000/?code=evil")
    assert not redirect_matches("https://example.com/cb", "https://example.com:444/cb")


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
def test_valid_loopback_redirects_allow_only_the_port_to_change(host: str) -> None:
    registered = f"http://{host}/oauth/otari"
    OAuthAppConfig(name="App", redirect_uris=[registered])
    assert redirect_matches(registered, registered)
    assert redirect_matches(registered, f"http://{host}:65535/oauth/otari")
    for suffix in ("/other", "/oauth/otari?", "/oauth/otari#", "/oauth/otari\n"):
        assert not redirect_matches(registered, f"http://{host}:4567{suffix}")
    for port in ("bad", "65536", "0", ""):
        assert not redirect_matches(registered, f"http://{host}:{port}/oauth/otari")


def test_https_redirects_are_exact() -> None:
    uri = "https://example.com/callback"
    OAuthAppConfig(name="App", redirect_uris=[uri])
    assert redirect_matches(uri, uri)
    for other in (uri + "/", uri + "?", uri.replace("https", "http"), uri.replace("example.com", "example.com:443")):
        assert not redirect_matches(uri, other)
    assert not redirect_matches("http://localhost/cb", "http://[bad/cb")


@pytest.mark.parametrize("client_id", ["", "x" * 129])
def test_registration_refuses_unusable_client_identifiers(client_id: str) -> None:
    with pytest.raises(ValidationError):
        IdentitySettings(oauth_apps={client_id: OAuthAppConfig(name="App", redirect_uris=["https://example.com/cb"])})


@pytest.mark.parametrize("verifier", ["x" * 42, "x" * 129, "x" * 42 + "!", "x" * 42 + "é", "x" * 43 + "\n"])
def test_exchange_rejects_invalid_pkce_verifiers(verifier: str) -> None:
    with pytest.raises(ValidationError):
        AppTokenRequest(
            grant_type="authorization_code",
            client_id="app",
            redirect_uri="https://example.com/cb",
            code="c" * 43,
            code_verifier=verifier,
        )


def _authorization() -> dict[str, str]:
    return {
        "client_id": "app",
        "redirect_uri": "https://example.com/cb",
        "state": "s" * 32,
        "code_challenge": "x" * 43,
        "code_challenge_method": "S256",
    }


@pytest.mark.parametrize("model", [AppAuthorizationParameters, AppAuthorizationRequest])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("client_id", ""),
        ("client_id", "x" * 129),
        ("redirect_uri", "https://example.com/" + "x" * 2048),
        ("state", "s" * 15),
        ("state", "s" * 257),
        ("state", "s" * 32 + "\n"),
        ("code_challenge", "x" * 42),
        ("code_challenge", "x" * 44),
        ("code_challenge", "x" * 42 + "+"),
        ("code_challenge", "x" * 43 + "\n"),
        ("code_challenge_method", "plain"),
        ("response_type", "token"),
    ],
)
def test_authorization_boundaries(
    model: type[AppAuthorizationParameters] | type[AppAuthorizationRequest], field: str, value: str
) -> None:
    payload = {**_authorization(), "workspace_id": str(uuid.uuid4()), field: value}
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_dashboard_consent_has_its_own_workspace_and_extra_field_contract() -> None:
    payload = _authorization()
    wire = AppAuthorizationParameters.model_validate({**payload, "scope": "models"})
    assert wire.model_dump(exclude_none=True) == {**payload, "response_type": "code"}
    with pytest.raises(ValidationError, match="workspace_id"):
        AppAuthorizationRequest.model_validate(payload)
    consent = {**payload, "workspace_id": str(uuid.uuid4()), "scopes": ["inference:run"]}
    assert str(AppAuthorizationRequest.model_validate(consent).workspace_id) == consent["workspace_id"]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AppAuthorizationRequest.model_validate({**consent, "scope": "models"})


def test_an_app_key_gets_the_default_scopes_unless_its_registration_names_its_own() -> None:
    assert OAuthAppConfig(name="App", redirect_uris=["https://app.example/cb"]).scopes == list(DEFAULT_KEY_SCOPES)
    widened = OAuthAppConfig.model_validate(
        {"name": "App", "redirect_uris": ["https://app.example/cb"], "scopes": ["usage:read", "inference:run"]}
    )
    assert widened.scopes == [KeyScope.INFERENCE_RUN, KeyScope.USAGE_READ]


@pytest.mark.parametrize("scopes", [[], ["end-users:read"], ["keys:write"]], ids=["none", "end-users", "unknown"])
def test_an_app_registration_refuses_scopes_an_app_key_cannot_hold(scopes: list[str]) -> None:
    with pytest.raises(ValidationError):
        OAuthAppConfig.model_validate({"name": "App", "redirect_uris": ["https://app.example/cb"], "scopes": scopes})


@pytest.mark.parametrize(
    "client_id,redirect_uri,site",
    [
        ("https://someone.github.io", "https://someone.github.io/pizza/", "someone.github.io"),
        ("https://shop.example:8443", "https://shop.example:8443/", "shop.example:8443"),
        ("http://localhost:8080", "http://localhost:8080/", "localhost:8080"),
        ("http://127.0.0.1:8080", "http://127.0.0.1:8080/index.html", "127.0.0.1:8080"),
        ("https://xn--pple-43d.com", "https://xn--pple-43d.com/cb", "xn--pple-43d.com"),
        ("http://[::1]:8080", "http://[::1]:8080/", "[::1]:8080"),
        ("https://[2001:db8::1]", "https://[2001:db8::1]/", "[2001:db8::1]"),
    ],
)
def test_an_open_app_is_the_site_its_callback_is_on(client_id: str, redirect_uri: str, site: str) -> None:
    assert open_app_site(client_id, redirect_uri) == site


@pytest.mark.parametrize(
    "client_id,redirect_uri",
    [
        ("https://someone.github.io", "https://other.github.io/"),
        ("https://someone.github.io/", "https://someone.github.io/"),
        ("https://SOMEONE.github.io", "https://someone.github.io/"),
        ("https://someone.github.io:443", "https://someone.github.io/"),
        ("https://someone.github.io:443", "https://someone.github.io:443/"),
        ("https://Someone.github.io", "https://Someone.github.io/"),
        ("https://shop.example:08443", "https://shop.example:08443/"),
        ("http://[0:0:0:0:0:0:0:1]:8080", "http://[0:0:0:0:0:0:0:1]:8080/"),
        ("https://[2001:DB8::1]", "https://[2001:DB8::1]/"),
        ("https://[::ffff:1.2.3.4]", "https://[::ffff:1.2.3.4]/"),
        # Lookalike hosts: a browser writes these in punycode, so the name shown would lie.
        ("https://\u0430pple.com", "https://\u0430pple.com/cb"),
        ("https://\uff41pple.com", "https://\uff41pple.com/cb"),
        ("https://apple.com\u200b.evil.io", "https://apple.com\u200b.evil.io/cb"),
        ("https://a\u00adb.com", "https://a\u00adb.com/cb"),
        ("https://apple.com\u3002evil.com", "https://apple.com\u3002evil.com/cb"),
        ("http://someone.example", "http://someone.example/"),
        ("http://localhost:8080", "http://localhost:9090/"),
        ("https://someone.github.io", "https://someone.github.io/?next=/"),
        ("thunderbolt", "https://app.thunderbolt.io/oauth/callback"),
    ],
)
def test_an_open_app_client_id_must_be_its_callbacks_origin(client_id: str, redirect_uri: str) -> None:
    assert open_app_site(client_id, redirect_uri) is None


@pytest.mark.parametrize("limit", ["0", "-1", "1000000", "12.345", "lots"])
def test_consent_refuses_a_spending_limit_the_screen_cannot_offer(limit: str) -> None:
    consent = {**_authorization(), "workspace_id": str(uuid.uuid4()), "scopes": ["inference:run"], "limit": limit}
    with pytest.raises(ValidationError):
        AppAuthorizationRequest.model_validate(consent)


def test_consent_defaults_to_no_limit_and_no_expiry_and_refuses_an_unknown_expiry() -> None:
    consent = {**_authorization(), "workspace_id": str(uuid.uuid4()), "scopes": ["inference:run"]}
    approved = AppAuthorizationRequest.model_validate(consent)
    assert approved.limit is None
    assert approved.expiration == "never"
    assert set(APP_KEY_LIFETIMES) == set(get_args(AppKeyExpiration))
    with pytest.raises(ValidationError):
        AppAuthorizationRequest.model_validate({**consent, "expiration": "2-days"})


def test_an_app_may_suggest_a_limit_and_expiry_the_screen_validates_later() -> None:
    wire = AppAuthorizationParameters.model_validate({**_authorization(), "limit": "lots", "expiration": "2-days"})
    assert (wire.limit, wire.expiration) == ("lots", "2-days")
    too_long = AppAuthorizationParameters.model_validate(
        {**_authorization(), "limit": "1" * 33, "expiration": "2026-11-18T00:00:00Z"}
    )
    assert (too_long.limit, too_long.expiration) == (None, None)
