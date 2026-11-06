"""Protocol boundaries independent of the dashboard."""

import uuid

import pytest
from pydantic import ValidationError

from gateway.core.settings.api_keys import ApiKeySettings, OAuthAppConfig
from gateway.schemas.api_keys import AppAuthorizationRequest, AppTokenRequest
from gateway.services.api_keys._oauth_apps import pkce_challenge, redirect_matches


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
        ApiKeySettings(oauth_apps={client_id: OAuthAppConfig(name="App", redirect_uris=["https://example.com/cb"])})


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


@pytest.mark.parametrize("challenge", ["x" * 42, "x" * 44, "x" * 42 + "+", "x" * 43 + "\n"])
def test_authorization_rejects_invalid_s256_challenges(challenge: str) -> None:
    with pytest.raises(ValidationError):
        AppAuthorizationRequest(
            client_id="app",
            redirect_uri="https://example.com/cb",
            state="s" * 32,
            code_challenge=challenge,
            code_challenge_method="S256",
            workspace_id=uuid.uuid4(),
        )
