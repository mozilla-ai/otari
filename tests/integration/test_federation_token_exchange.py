"""The token exchange: ``POST /api/v1/oauth/token`` trades a workload's identity token for a short-lived access token.

The contract is the one the Anthropic SDKs and Claude Code speak (the RFC 7523 ``jwt-bearer`` grant), so one test
drives the real ``anthropic`` client through it. The issuer's keys are generated here and served by a stub, so nothing
reaches the network.
"""

import time
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import anthropic
import httpx
import jwt
import pytest
from anthropic.lib.credentials import WorkloadIdentityCredentials, WorkloadIdentityError
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from any_llm.types.messages import MessageResponse, MessageUsage, TextBlock
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from jwt.algorithms import RSAAlgorithm
from sqlalchemy import event
from sqlalchemy.engine import Engine

from gateway.api.deps import AUTH_FAILURES
from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.settings.api_keys import FederationIssuer
from gateway.services.api_keys import FederatedTokenVerifier
from gateway.services.secret_box import generate_secret_key

from .conftest import MODEL_NAME, build_test_client

ISSUER = "https://issuer.example.com"
JWKS_URI = "https://issuer.example.com/keys"
AUDIENCE = "https://otari.example.com"
SUBJECT = "113051416744238958061"
EMAIL = "hackbot-api@project.iam.gserviceaccount.com"
KEYS = f"{API_ROOT}/keys"
TOKEN_URL = f"{API_ROOT}/oauth/token"
GRANT = "urn:ietf:params:oauth:grant-type:jwt-bearer"

SIGNING_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class _Jwks:
    """Serves the issuer's keys and counts how often they are fetched."""

    def __init__(self) -> None:
        self.fetches = 0

    def transport(self, _url: str) -> httpx.AsyncBaseTransport:
        return httpx.MockTransport(self._respond)

    def _respond(self, request: httpx.Request) -> httpx.Response:
        assert str(request.url) == JWKS_URI
        self.fetches += 1
        jwk = RSAAlgorithm.to_jwk(SIGNING_KEY.public_key(), as_dict=True)
        return httpx.Response(200, json={"keys": [{**jwk, "kid": "k1", "alg": "RS256", "use": "sig"}]})


def _assertion(*, kid: str = "k1", **overrides: Any) -> str:
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": SUBJECT,
        "email": EMAIL,
        "email_verified": True,
        "iat": now,
        "exp": now + 3600,
    }
    claims.update(overrides)
    payload = {name: value for name, value in claims.items() if value is not None}
    return jwt.encode(payload, SIGNING_KEY, algorithm="RS256", headers={"kid": kid})


@pytest.fixture
def federation_config(test_config: GatewayConfig) -> GatewayConfig:
    return test_config.model_copy(
        update={"federation_issuers": [FederationIssuer(issuer=ISSUER, jwks_uri=JWKS_URI)]}, deep=True
    )


@pytest.fixture
def jwks() -> _Jwks:
    return _Jwks()


@pytest.fixture
def client(
    federation_config: GatewayConfig, clean_database: None, jwks: _Jwks, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    for test_client in build_test_client(federation_config):
        test_client.app.state.federated_token_verifier = FederatedTokenVerifier(  # type: ignore[attr-defined]
            federation_config.federation_issuers, transport_factory=jwks.transport
        )
        yield test_client


async def _mock_acompletion(**_kwargs: Any) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-x",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _chat(client: TestClient, headers: dict[str, str]) -> Any:
    body = {"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}]}
    with patch("gateway.api.routes.chat.acompletion") as mock:
        mock.side_effect = _mock_acompletion
        return client.post(f"{API_ROOT}/chat/completions", json=body, headers=headers)


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class _Federated:
    """A key with one federation rule, and the ids a client names it by."""

    def __init__(self, client: TestClient, master_key_header: dict[str, str], **settings: Any) -> None:
        created = client.post(
            KEYS, json={"key_name": "hackbot", "user_id": "hackbot", **settings}, headers=master_key_header
        )
        assert created.status_code == 200, created.text
        self.key_id: str = created.json()["id"]
        read = client.get(f"{KEYS}/{self.key_id}", headers=master_key_header)
        self.workspace_id: str = read.json()["workspace_id"]
        rule = client.post(
            f"{KEYS}/{self.key_id}/federation-rules",
            json={"issuer": ISSUER, "audience": AUDIENCE, "email": EMAIL},
            headers=master_key_header,
        )
        assert rule.status_code == 201, rule.text
        self.rule_id: str = rule.json()["id"]
        organization = client.get(f"{API_ROOT}/organizations/me", headers=master_key_header)
        assert organization.status_code == 200, organization.text
        self.organization_id: str = organization.json()["organization"]["id"]

    def grant(self, **overrides: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "grant_type": GRANT,
            "assertion": _assertion(),
            "federation_rule_id": self.rule_id,
            "organization_id": self.organization_id,
        }
        body.update(overrides)
        return {name: value for name, value in body.items() if value is not None}


@pytest.fixture
def federated(client: TestClient, master_key_header: dict[str, str]) -> _Federated:
    return _Federated(client, master_key_header)


_SDK_HEADERS = {"anthropic-beta": "oauth-2025-04-20,oidc-federation-2026-04-01"}


def _exchange(client: TestClient, body: dict[str, Any]) -> Any:
    return client.post(TOKEN_URL, json=body, headers=_SDK_HEADERS)


def _failures(reason: str) -> float:
    return float(AUTH_FAILURES.labels(reason=reason)._value.get())


@contextmanager
def _writes() -> Iterator[list[str]]:
    """Collect every INSERT, UPDATE or DELETE any engine runs while the block is open."""
    seen: list[str] = []

    def record(_conn: Any, _cursor: Any, statement: str, *_args: Any) -> None:
        if statement.lstrip().split(" ", 1)[0].upper() in {"INSERT", "UPDATE", "DELETE"}:
            seen.append(statement)

    event.listen(Engine, "before_cursor_execute", record)
    try:
        yield seen
    finally:
        event.remove(Engine, "before_cursor_execute", record)


# The exchange


def test_an_exchanged_token_authenticates_as_the_rules_key(client: TestClient, federated: _Federated) -> None:
    exchanged = _exchange(client, federated.grant())

    assert exchanged.status_code == 200, exchanged.text
    body = exchanged.json()
    assert set(body) == {"access_token", "token_type", "expires_in"}
    assert body["token_type"] == "Bearer"
    assert 590 <= body["expires_in"] <= 600
    assert body["access_token"].startswith("otari-fat1.")
    assert "no-store" in exchanged.headers["Cache-Control"]
    served = _chat(client, _bearer(body["access_token"]))
    assert served.status_code == 200, served.text


def test_the_beta_header_is_not_required(client: TestClient, federated: _Federated) -> None:
    assert client.post(TOKEN_URL, json=federated.grant()).status_code == 200


def test_the_token_never_outlives_the_identity_token(client: TestClient, federated: _Federated) -> None:
    exchanged = _exchange(client, federated.grant(assertion=_assertion(exp=int(time.time()) + 120)))

    assert exchanged.status_code == 200
    assert exchanged.json()["expires_in"] <= 120


def test_workspace_and_service_account_are_accepted_when_they_are_the_keys(
    client: TestClient, federated: _Federated
) -> None:
    exchanged = _exchange(
        client, federated.grant(workspace_id=federated.workspace_id, service_account_id=federated.key_id)
    )

    assert exchanged.status_code == 200, exchanged.text


_INVALID_GRANT = {
    "error": "invalid_grant",
    "error_description": "The identity token does not match the federation rule.",
}


@pytest.mark.parametrize(
    "overrides",
    [
        {"organization_id": "00000000-0000-0000-0000-000000000000"},
        {"organization_id": "not-a-uuid"},
        {"workspace_id": "00000000-0000-0000-0000-000000000000"},
        {"service_account_id": "another-key"},
        {"federation_rule_id": "no-such-rule"},
        {"assertion_overrides": {"email": "someone@project.iam.gserviceaccount.com"}},
        {"assertion_overrides": {"email_verified": False}},
        {"assertion_overrides": {"aud": "https://elsewhere.example.com"}},
        {"assertion_overrides": {"iat": int(time.time()) - 7200, "exp": int(time.time()) - 600}},
        {"assertion": "not-a-jwt"},
    ],
    ids=[
        "other-organization",
        "malformed-organization",
        "other-workspace",
        "other-service-account",
        "unknown-rule",
        "other-email",
        "unverified-email",
        "wrong-audience",
        "expired-assertion",
        "not-a-jwt",
    ],
)
def test_a_refused_grant_is_invalid_grant_with_one_description(
    client: TestClient, federated: _Federated, overrides: dict[str, Any]
) -> None:
    assertion_overrides = overrides.pop("assertion_overrides", None)
    if assertion_overrides is not None:
        overrides["assertion"] = _assertion(**assertion_overrides)

    refused = _exchange(client, federated.grant(**overrides))

    assert refused.status_code == 401
    assert refused.json() == _INVALID_GRANT


def test_an_identity_token_valid_for_another_rule_is_refused_for_this_one(
    client: TestClient, master_key_header: dict[str, str], federated: _Federated
) -> None:
    other = client.post(KEYS, json={"key_name": "other", "user_id": "other"}, headers=master_key_header).json()
    other_rule = client.post(
        f"{KEYS}/{other['id']}/federation-rules",
        json={"issuer": ISSUER, "audience": AUDIENCE, "subject": "another-workload"},
        headers=master_key_header,
    ).json()

    refused = _exchange(client, federated.grant(federation_rule_id=other_rule["id"]))

    assert refused.status_code == 401
    assert refused.json() == _INVALID_GRANT


def test_a_wrong_grant_type_is_unsupported(client: TestClient, federated: _Federated) -> None:
    refused = _exchange(client, federated.grant(grant_type="client_credentials"))

    assert refused.status_code == 400
    assert refused.json()["error"] == "unsupported_grant_type"


@pytest.mark.parametrize("missing", ["assertion", "federation_rule_id"])
def test_a_missing_field_is_an_invalid_request(client: TestClient, federated: _Federated, missing: str) -> None:
    body = federated.grant()
    del body[missing]

    refused = _exchange(client, body)

    assert refused.status_code == 400
    assert refused.json()["error"] == "invalid_request"


def test_an_inactive_key_cannot_exchange(
    client: TestClient, master_key_header: dict[str, str], federated: _Federated
) -> None:
    client.patch(f"{KEYS}/{federated.key_id}", json={"is_active": False}, headers=master_key_header)

    refused = _exchange(client, federated.grant())

    assert refused.status_code == 401
    assert refused.json() == _INVALID_GRANT


def test_a_secret_key_removed_after_start_makes_the_exchange_unavailable(
    client: TestClient, federated: _Federated, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OTARI_SECRET_KEY")

    refused = _exchange(client, federated.grant())

    assert refused.status_code == 503
    assert refused.json()["error"] == "temporarily_unavailable"


def test_with_federation_off_the_endpoint_is_absent(test_config: GatewayConfig, clean_database: None) -> None:
    for client in build_test_client(test_config):
        assert client.post(TOKEN_URL, json={"grant_type": GRANT}).status_code == 404


# Refusals are cheap


def test_an_exchange_refused_or_not_writes_nothing(client: TestClient, federated: _Federated) -> None:
    with _writes() as writes:
        for body in (
            federated.grant(organization_id="00000000-0000-0000-0000-000000000000"),
            federated.grant(assertion=_assertion(sub="x", email="someone@example.com")),
            federated.grant(federation_rule_id="no-such-rule"),
            federated.grant(grant_type="password"),
        ):
            assert _exchange(client, body).status_code in (400, 401)
        exchanged = _exchange(client, federated.grant())
        assert exchanged.status_code == 200
        exchange_writes = list(writes)
        # The listener does see writes: the access token's first use records the key's first use.
        assert _chat(client, _bearer(exchanged.json()["access_token"])).status_code == 200

    assert exchange_writes == []
    assert any(statement.lstrip().upper().startswith("UPDATE API_KEYS") for statement in writes)


def test_a_flood_of_unknown_key_ids_does_not_refetch_the_jwks(
    client: TestClient, federated: _Federated, jwks: _Jwks
) -> None:
    assert _exchange(client, federated.grant()).status_code == 200
    fetched = jwks.fetches

    for attempt in range(50):
        refused = _exchange(client, federated.grant(assertion=_assertion(kid=f"made-up-{attempt}")))
        assert refused.status_code == 401

    assert jwks.fetches <= fetched + 1


# Using the access token


def _token(client: TestClient, federated: _Federated) -> str:
    exchanged = _exchange(client, federated.grant())
    assert exchanged.status_code == 200, exchanged.text
    return str(exchanged.json()["access_token"])


def test_the_access_token_stops_working_when_the_rule_is_deleted(
    client: TestClient, master_key_header: dict[str, str], federated: _Federated
) -> None:
    token = _token(client, federated)
    client.delete(f"{KEYS}/{federated.key_id}/federation-rules/{federated.rule_id}", headers=master_key_header)
    before = _failures("federation_rule_gone")

    refused = _chat(client, _bearer(token))

    assert refused.status_code == 401
    assert refused.json()["detail"] == "Invalid API key"
    assert _failures("federation_rule_gone") == before + 1


def test_the_access_token_stops_working_when_the_key_is_deactivated(
    client: TestClient, master_key_header: dict[str, str], federated: _Federated
) -> None:
    token = _token(client, federated)
    client.patch(f"{KEYS}/{federated.key_id}", json={"is_active": False}, headers=master_key_header)

    refused = _chat(client, _bearer(token))

    assert refused.status_code == 401
    assert refused.json()["detail"] == "API key is inactive"


def test_the_access_token_stops_working_when_its_issuer_is_no_longer_trusted(
    client: TestClient, federated: _Federated, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = _token(client, federated)
    state = client.app.state  # type: ignore[attr-defined]
    other = [FederationIssuer(issuer="https://other-issuer.example.com")]
    monkeypatch.setattr(state, "config", state.config.model_copy(update={"federation_issuers": other}))
    before = _failures("federation_unknown_issuer")

    refused = _chat(client, _bearer(token))

    assert refused.status_code == 401
    assert refused.json()["detail"] == "Invalid API key"
    assert _failures("federation_unknown_issuer") == before + 1


def test_the_access_token_stops_working_when_it_expires(client: TestClient, federated: _Federated) -> None:
    token = _token(client, federated)
    later = time.time() + 601
    before = _failures("federation_access_token_expired")

    with patch("gateway.api.deps.time.time", return_value=later):
        refused = _chat(client, _bearer(token))

    assert refused.status_code == 401
    assert _failures("federation_access_token_expired") == before + 1


def test_a_tampered_access_token_is_refused(client: TestClient, federated: _Federated) -> None:
    token = _token(client, federated)
    prefix, payload, signature = token.split(".")
    flipped = signature[:-2] + ("AA" if signature[-2:] != "AA" else "BB")
    before = _failures("federation_access_token_invalid")

    for forged in (f"{prefix}.{payload}.{flipped}", f"{prefix}.{payload[:-1]}x.{signature}"):
        refused = _chat(client, _bearer(forged))
        assert refused.status_code == 401
        assert refused.json()["detail"] == "Invalid API key"

    assert _failures("federation_access_token_invalid") == before + 2


def test_an_access_token_signed_under_another_secret_is_refused(
    client: TestClient, federated: _Federated, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = _token(client, federated)
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    assert _chat(client, _bearer(token)).status_code == 401


def test_a_rotated_secret_still_verifies_tokens_from_the_old_one(
    client: TestClient, federated: _Federated, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    token = _token(client, federated)
    monkeypatch.setenv("OTARI_SECRET_KEY", f"{generate_secret_key()} {os.environ['OTARI_SECRET_KEY']}")

    assert _chat(client, _bearer(token)).status_code == 200


def test_the_access_token_is_never_the_master_key(client: TestClient, federated: _Federated) -> None:
    token = _token(client, federated)

    assert client.get(KEYS, headers=_bearer(token)).status_code == 401
    assert client.get(f"{API_ROOT}/users", headers=_bearer(token)).status_code == 401


def test_the_access_token_is_a_service_key_only_when_its_key_is(
    client: TestClient, master_key_header: dict[str, str], federated: _Federated
) -> None:
    ordinary = _token(client, federated)
    assert client.get(f"{API_ROOT}/end-users", headers=_bearer(ordinary)).status_code == 403

    client.delete(f"{KEYS}/{federated.key_id}", headers=master_key_header)
    service = _Federated(client, master_key_header, is_service_key=True)

    assert client.get(f"{API_ROOT}/end-users", headers=_bearer(_token(client, service))).status_code == 200


def test_the_access_token_and_the_identity_token_each_take_their_own_path(
    client: TestClient, federated: _Federated
) -> None:
    token = _token(client, federated)
    jwt_before = _failures("federation_invalid_token")
    access_before = _failures("federation_access_token_invalid")

    # An access token whose prefix is removed is no JWT; a JWT given the prefix is no access token.
    assert _chat(client, _bearer(token.removeprefix("otari-fat1."))).status_code == 401
    assert _chat(client, _bearer(f"otari-fat1.{_assertion()}")).status_code == 401

    assert _failures("federation_invalid_token") == jwt_before
    assert _failures("federation_access_token_invalid") == access_before + 1
    # And both still authenticate as themselves.
    assert _chat(client, _bearer(token)).status_code == 200
    assert _chat(client, _bearer(_assertion())).status_code == 200


def test_a_key_secret_still_authenticates_beside_access_tokens(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    created = client.post(KEYS, json={"user_id": "plain"}, headers=master_key_header).json()

    assert _chat(client, {API_KEY_HEADER: f"Bearer {created['key']}"}).status_code == 200


def test_an_expired_key_stops_its_access_tokens(
    client: TestClient, master_key_header: dict[str, str], federated: _Federated
) -> None:
    token = _token(client, federated)
    past = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    client.patch(f"{KEYS}/{federated.key_id}", json={"expires_at": past}, headers=master_key_header)

    assert _chat(client, _bearer(token)).status_code == 401


# The Anthropic SDK, end to end


class _ThroughTestClient(httpx.BaseTransport):
    """Hands each request the SDK makes to the test client, which serves it from the app in process."""

    def __init__(self, client: TestClient) -> None:
        self._client = client
        self.seen: list[tuple[str, str, str]] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.seen.append((request.method, request.url.path, request.headers.get("Authorization", "")[:18]))
        served = self._client.request(request.method, str(request.url), headers=request.headers, content=request.read())
        headers = [(name, value) for name, value in served.headers.multi_items() if name.lower() != "content-encoding"]
        return httpx.Response(served.status_code, headers=headers, content=served.content)


def _sdk_http(transport: _ThroughTestClient) -> httpx.Client:
    """A plain httpx client, which the SDK insists on, served by the test app."""
    return httpx.Client(transport=transport)


def test_the_anthropic_sdk_exchanges_and_calls_messages(client: TestClient, federated: _Federated) -> None:
    """The SDK's own workload-identity client, pointed at Otari as ANTHROPIC_BASE_URL=<origin>/api would."""
    transport = _ThroughTestClient(client)
    credentials = WorkloadIdentityCredentials(
        identity_token_provider=_assertion,
        federation_rule_id=federated.rule_id,
        organization_id=federated.organization_id,
        http_client=_sdk_http(transport),
    )
    sdk = anthropic.Anthropic(
        base_url="http://localhost/api", credentials=credentials, http_client=_sdk_http(transport)
    )
    reply = MessageResponse(
        id="msg_1",
        type="message",
        role="assistant",
        content=[TextBlock(type="text", text="Hello!")],
        model="claude-3-5-sonnet",
        stop_reason="end_turn",
        usage=MessageUsage(input_tokens=10, output_tokens=5),
    )

    with patch("gateway.api.routes.messages.amessages", new_callable=AsyncMock, return_value=reply):
        message = sdk.messages.create(
            model="anthropic:claude-3-5-sonnet", max_tokens=16, messages=[{"role": "user", "content": "Hello"}]
        )

    assert message.content[0].type == "text"
    assert message.content[0].text == "Hello!"
    assert transport.seen == [
        ("POST", f"{API_ROOT}/oauth/token", ""),
        ("POST", f"{API_ROOT}/messages", "Bearer otari-fat1."),
    ]


def test_the_anthropic_sdk_reports_a_refusal_with_its_rule_hint(client: TestClient, federated: _Federated) -> None:
    credentials = WorkloadIdentityCredentials(
        identity_token_provider=lambda: _assertion(sub="someone-else", email="someone@example.com"),
        federation_rule_id=federated.rule_id,
        organization_id=federated.organization_id,
        http_client=_sdk_http(_ThroughTestClient(client)),
    )
    credentials.bind_base_url("http://localhost/api")

    with pytest.raises(WorkloadIdentityError) as refused:
        credentials()

    assert refused.value.status_code == 401
    assert refused.value.body == _INVALID_GRANT
    assert "Ensure your federation rule matches your identity token" in str(refused.value)
