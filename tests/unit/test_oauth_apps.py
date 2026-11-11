"""Protocol boundaries independent of the dashboard."""

import uuid

import pytest
from apron_auth.pkce import generate_code_challenge as pkce_challenge
from pydantic import ValidationError

from gateway.core.oauth_apps import redirect_matches
from gateway.core.settings.identity import IdentitySettings, OAuthAppConfig
from gateway.schemas.identity import AppAuthorizationRequest
from gateway.schemas.identity.oauth_apps import AppAuthorizationParameters, AppTokenRequest
from gateway.types.key_scope import KeyScope


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
    assert wire.model_dump() == {**payload, "response_type": "code"}
    with pytest.raises(ValidationError, match="workspace_id"):
        AppAuthorizationRequest.model_validate(payload)
    consent = {**payload, "workspace_id": str(uuid.uuid4())}
    assert str(AppAuthorizationRequest.model_validate(consent).workspace_id) == consent["workspace_id"]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AppAuthorizationRequest.model_validate({**consent, "scope": "models"})


def test_an_app_key_calls_models_alone_unless_its_registration_says_more() -> None:
    assert OAuthAppConfig(name="App", redirect_uris=["https://app.example/cb"]).scopes == [KeyScope.INFERENCE]
    widened = OAuthAppConfig.model_validate(
        {"name": "App", "redirect_uris": ["https://app.example/cb"], "scopes": ["usage:read", "inference"]}
    )
    assert widened.scopes == [KeyScope.INFERENCE, KeyScope.USAGE_READ]


@pytest.mark.parametrize("scopes", [[], ["end_users:read"], ["keys:write"]], ids=["none", "end-users", "unknown"])
def test_an_app_registration_refuses_scopes_an_app_key_cannot_hold(scopes: list[str]) -> None:
    with pytest.raises(ValidationError):
        OAuthAppConfig.model_validate({"name": "App", "redirect_uris": ["https://app.example/cb"], "scopes": scopes})
