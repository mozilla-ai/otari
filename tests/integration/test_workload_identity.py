"""Workload identity: an OpenID Connect identity token authenticates as the key whose federation rule it matches.

The issuer's keys are generated here and served by a stub fetcher, so nothing
reaches the network. Every refusal answers the same 401 as an unknown key.
"""

import time
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import httpx
import jwt
import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from jwt.algorithms import RSAAlgorithm

from gateway.api.deps import AUTH_FAILURES
from gateway.core.config import API_KEY_HEADER, API_ROOT, X_API_KEY_HEADER, GatewayConfig
from gateway.core.settings.api_keys import FederationIssuer
from gateway.services.api_keys import FederatedTokenVerifier
from gateway.services.secret_box import generate_secret_key

from .conftest import MODEL_NAME, build_test_client

ISSUER = "https://issuer.example.com"
UNLISTED_ISSUER = "https://unlisted.example.com"
JWKS_URI = "https://issuer.example.com/keys"
AUDIENCE = "https://otari.example.com"
SUBJECT = "113051416744238958061"
EMAIL = "hackbot-api@project.iam.gserviceaccount.com"
KEYS = f"{API_ROOT}/keys"

SIGNING_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _serve_keys(request: httpx.Request) -> httpx.Response:
    assert str(request.url) == JWKS_URI
    jwk = RSAAlgorithm.to_jwk(SIGNING_KEY.public_key(), as_dict=True)
    return httpx.Response(200, json={"keys": [{**jwk, "kid": "k1", "alg": "RS256", "use": "sig"}]})


def _issuer(_url: str) -> httpx.AsyncBaseTransport:
    return httpx.MockTransport(_serve_keys)


def _token(private_key: Any = SIGNING_KEY, *, algorithm: str = "RS256", **overrides: Any) -> str:
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
    return jwt.encode(payload, private_key, algorithm=algorithm, headers={"kid": "k1"})


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def federation_config(test_config: GatewayConfig) -> GatewayConfig:
    return test_config.model_copy(
        update={"federation_issuers": [FederationIssuer(issuer=ISSUER, jwks_uri=JWKS_URI)]}, deep=True
    )


@pytest.fixture
def client(
    federation_config: GatewayConfig, clean_database: None, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient]:
    # Federation needs a secret to start, for the access tokens the exchange signs.
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    for test_client in build_test_client(federation_config):
        test_client.app.state.federated_token_verifier = FederatedTokenVerifier(  # type: ignore[attr-defined]
            federation_config.federation_issuers, transport_factory=_issuer
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


def _key(client: TestClient, master_key_header: dict[str, str], owner: str = "hackbot", **settings: Any) -> str:
    created = client.post(KEYS, json={"key_name": owner, "user_id": owner, **settings}, headers=master_key_header)
    assert created.status_code == 200, created.text
    return str(created.json()["id"])


def _rule(client: TestClient, master_key_header: dict[str, str], key_id: str, **rule: Any) -> Any:
    body = {"issuer": ISSUER, "audience": AUDIENCE, **rule}
    return client.post(f"{KEYS}/{key_id}/federation-rules", json=body, headers=master_key_header)


def _federated_key(client: TestClient, master_key_header: dict[str, str], **settings: Any) -> str:
    key_id = _key(client, master_key_header, **settings)
    assert _rule(client, master_key_header, key_id, email=EMAIL).status_code == 201
    return key_id


def _failures(reason: str) -> float:
    return float(AUTH_FAILURES.labels(reason=reason)._value.get())


# Managing rules


def test_an_operator_adds_lists_and_deletes_a_keys_rules(client: TestClient, master_key_header: dict[str, str]) -> None:
    key_id = _key(client, master_key_header)

    by_email = _rule(client, master_key_header, key_id, email=EMAIL.upper())
    by_subject = _rule(client, master_key_header, key_id, subject=SUBJECT)
    listed = client.get(f"{KEYS}/{key_id}/federation-rules", headers=master_key_header)
    deleted = client.delete(f"{KEYS}/{key_id}/federation-rules/{by_email.json()['id']}", headers=master_key_header)
    again = client.delete(f"{KEYS}/{key_id}/federation-rules/{by_email.json()['id']}", headers=master_key_header)
    after = client.get(f"{KEYS}/{key_id}/federation-rules", headers=master_key_header)

    assert by_email.status_code == 201, by_email.text
    assert by_email.json() | {"api_key_id": key_id, "issuer": ISSUER, "email": EMAIL, "subject": None} == (
        by_email.json()
    )
    assert by_subject.status_code == 201
    assert [rule["id"] for rule in listed.json()] == [by_email.json()["id"], by_subject.json()["id"]]
    assert deleted.status_code == 204
    assert again.status_code == 404
    assert [rule["id"] for rule in after.json()] == [by_subject.json()["id"]]


def test_a_rule_needs_an_allow_listed_issuer_and_a_subject_or_email(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    key_id = _key(client, master_key_header)

    unlisted = _rule(client, master_key_header, key_id, issuer=UNLISTED_ISSUER, email=EMAIL)
    anyone = _rule(client, master_key_header, key_id)

    assert unlisted.status_code == 400
    assert "federation_issuers" in unlisted.json()["detail"]
    assert anyone.status_code == 422


def test_one_identity_cannot_be_given_to_two_keys(client: TestClient, master_key_header: dict[str, str]) -> None:
    first = _key(client, master_key_header, "first")
    second = _key(client, master_key_header, "second")

    assert _rule(client, master_key_header, first, email=EMAIL).status_code == 201
    assert _rule(client, master_key_header, first, email=EMAIL).status_code == 409
    assert _rule(client, master_key_header, second, email=EMAIL).status_code == 409


def test_a_rule_overlapping_another_keys_rule_is_refused(client: TestClient, master_key_header: dict[str, str]) -> None:
    first = _key(client, master_key_header, "first")
    second = _key(client, master_key_header, "second")
    assert _rule(client, master_key_header, first, subject=SUBJECT, email=EMAIL).status_code == 201

    # Each would match every token the first key's rule matches.
    by_email = _rule(client, master_key_header, second, email=EMAIL)
    by_subject = _rule(client, master_key_header, second, subject=SUBJECT)
    # Another subject with the same email, or another audience, matches a different set of tokens.
    other_subject = _rule(client, master_key_header, second, subject="another-subject", email=EMAIL)
    other_audience = _rule(client, master_key_header, second, audience="https://elsewhere.example.com", email=EMAIL)

    assert by_email.status_code == 409
    assert by_subject.status_code == 409
    assert other_subject.status_code == 201
    assert other_audience.status_code == 201


def test_rules_are_managed_by_the_operator_only(client: TestClient, master_key_header: dict[str, str]) -> None:
    key_id = _federated_key(client, master_key_header)
    plain_key = client.post(KEYS, json={"user_id": "other"}, headers=master_key_header).json()["key"]
    rules = f"{KEYS}/{key_id}/federation-rules"

    assert client.get(rules, headers={API_KEY_HEADER: f"Bearer {plain_key}"}).status_code == 401
    assert client.get(rules, headers=_bearer(_token())).status_code == 401
    assert client.get(f"{KEYS}/no-such-key/federation-rules", headers=master_key_header).status_code == 404


def test_deleting_the_key_deletes_its_rules(client: TestClient, master_key_header: dict[str, str]) -> None:
    key_id = _federated_key(client, master_key_header)

    assert client.delete(f"{KEYS}/{key_id}", headers=master_key_header).status_code == 204
    replacement = _key(client, master_key_header, "replacement")

    assert _rule(client, master_key_header, replacement, email=EMAIL).status_code == 201


# Authenticating


@pytest.mark.parametrize("header", ["Authorization", API_KEY_HEADER, X_API_KEY_HEADER])
def test_a_matching_token_authenticates_as_the_key(
    client: TestClient, master_key_header: dict[str, str], header: str
) -> None:
    _federated_key(client, master_key_header)
    token = _token()
    value = token if header == X_API_KEY_HEADER else f"Bearer {token}"

    served = _chat(client, {header: value})

    assert served.status_code == 200, served.text
    usage = client.get(f"{API_ROOT}/usage", params={"user_id": "hackbot"}, headers=master_key_header)
    assert [row["user_id"] for row in usage.json()] == ["hackbot"]


def test_a_subject_rule_authenticates_a_token_with_no_email(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    key_id = _key(client, master_key_header)
    assert _rule(client, master_key_header, key_id, subject=SUBJECT).status_code == 201

    assert _chat(client, _bearer(_token(email=None, email_verified=None))).status_code == 200


def test_a_token_for_a_service_key_manages_its_end_users(client: TestClient, master_key_header: dict[str, str]) -> None:
    _federated_key(client, master_key_header, is_service_key=True)

    listed = client.get(f"{API_ROOT}/end-users", headers=_bearer(_token()))

    assert listed.status_code == 200, listed.text


def test_a_token_for_an_ordinary_key_is_not_a_service_key(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _federated_key(client, master_key_header)

    assert client.get(f"{API_ROOT}/end-users", headers=_bearer(_token())).status_code == 403


def test_a_token_is_never_the_master_key(client: TestClient, master_key_header: dict[str, str]) -> None:
    _federated_key(client, master_key_header)

    assert client.get(KEYS, headers=_bearer(_token())).status_code == 401
    assert client.get(f"{API_ROOT}/users", headers=_bearer(_token())).status_code == 401


_REFUSALS: dict[str, tuple[dict[str, Any], str]] = {
    "unlisted-issuer": ({"iss": UNLISTED_ISSUER}, "federation_unknown_issuer"),
    "wrong-audience": ({"aud": "https://elsewhere.example.com"}, "federation_no_rule"),
    "other-email": ({"email": "someone@project.iam.gserviceaccount.com"}, "federation_no_rule"),
    "unverified-email": ({"email_verified": False}, "federation_no_rule"),
    "expired": ({"iat": int(time.time()) - 7200, "exp": int(time.time()) - 600}, "federation_expired"),
    "bad-signature": ({"private_key": OTHER_KEY}, "federation_invalid_token"),
}


@pytest.mark.parametrize("case", list(_REFUSALS))
def test_a_token_that_does_not_verify_or_match_is_refused_like_an_unknown_key(
    client: TestClient, master_key_header: dict[str, str], case: str
) -> None:
    _federated_key(client, master_key_header)
    overrides, reason = _REFUSALS[case]
    before = _failures(reason)

    refused = _chat(client, _bearer(_token(**overrides)))

    assert refused.status_code == 401
    assert refused.json()["detail"] == "Invalid API key"
    assert _failures(reason) == before + 1


def test_a_subject_rule_refuses_another_subject(client: TestClient, master_key_header: dict[str, str]) -> None:
    key_id = _key(client, master_key_header)
    assert _rule(client, master_key_header, key_id, subject=SUBJECT).status_code == 201

    refused = _chat(client, _bearer(_token(sub="another-workload")))

    assert refused.status_code == 401
    assert refused.json()["detail"] == "Invalid API key"


@pytest.mark.parametrize("algorithm", ["none", "HS256"])
def test_unsigned_and_hmac_tokens_are_refused(
    client: TestClient, master_key_header: dict[str, str], algorithm: str
) -> None:
    _federated_key(client, master_key_header)
    now = int(time.time())
    claims = {"iss": ISSUER, "aud": AUDIENCE, "sub": SUBJECT, "email": EMAIL, "email_verified": True}
    secret: Any = None if algorithm == "none" else "a-shared-secret-of-at-least-32-bytes!"
    token = jwt.encode(
        {**claims, "iat": now, "exp": now + 3600},
        secret,
        algorithm=algorithm,
        headers={"kid": "k1"},
    )

    before = _failures("federation_invalid_token")

    refused = _chat(client, _bearer(token))

    assert refused.status_code == 401
    assert refused.json()["detail"] == "Invalid API key"
    assert _failures("federation_invalid_token") == before + 1


def test_the_matched_key_is_still_checked(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Active, expiry, the owner's block and the model allow-list apply as they do to the key's secret."""
    key_id = _federated_key(client, master_key_header)
    token = _token()

    client.patch(f"{KEYS}/{key_id}", json={"allowed_models": []}, headers=master_key_header)
    model_refused = _chat(client, _bearer(token))
    client.patch(f"{KEYS}/{key_id}", json={"allowed_models": None}, headers=master_key_header)

    client.patch(f"{API_ROOT}/users/hackbot", json={"blocked": True}, headers=master_key_header)
    blocked = _chat(client, _bearer(token))
    client.patch(f"{API_ROOT}/users/hackbot", json={"blocked": False}, headers=master_key_header)

    past = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    client.patch(f"{KEYS}/{key_id}", json={"expires_at": past}, headers=master_key_header)
    expired = _chat(client, _bearer(token))
    client.patch(f"{KEYS}/{key_id}", json={"is_active": False, "expires_at": None}, headers=master_key_header)
    inactive = _chat(client, _bearer(token))

    assert model_refused.status_code == 403, model_refused.text
    assert blocked.status_code == 403, blocked.text
    assert blocked.headers["Otari-Error-Code"] == "user_blocked"
    assert expired.status_code == 401
    assert expired.json()["detail"] == "API key has expired"
    assert inactive.status_code == 401
    assert inactive.json()["detail"] == "API key is inactive"


def test_a_deleted_rule_stops_authenticating(client: TestClient, master_key_header: dict[str, str]) -> None:
    key_id = _key(client, master_key_header)
    rule = _rule(client, master_key_header, key_id, email=EMAIL).json()
    assert _chat(client, _bearer(_token())).status_code == 200

    client.delete(f"{KEYS}/{key_id}/federation-rules/{rule['id']}", headers=master_key_header)

    assert _chat(client, _bearer(_token())).status_code == 401


def test_rules_of_two_keys_matching_one_token_authenticate_neither(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    by_email = _key(client, master_key_header, "by-email")
    by_subject = _key(client, master_key_header, "by-subject")
    _rule(client, master_key_header, by_email, email=EMAIL)
    _rule(client, master_key_header, by_subject, subject=SUBJECT)
    before = _failures("federation_ambiguous_rule")

    refused = _chat(client, _bearer(_token()))

    assert refused.status_code == 401
    assert _failures("federation_ambiguous_rule") == before + 1


def test_an_unreachable_issuer_fails_closed(client: TestClient, master_key_header: dict[str, str]) -> None:
    _federated_key(client, master_key_header)

    def unreachable(_url: str) -> httpx.AsyncBaseTransport:
        return httpx.MockTransport(lambda _request: httpx.Response(503))

    client.app.state.federated_token_verifier = FederatedTokenVerifier(  # type: ignore[attr-defined]
        [FederationIssuer(issuer=ISSUER, jwks_uri=JWKS_URI)], transport_factory=unreachable
    )

    refused = _chat(client, _bearer(_token()))

    assert refused.status_code == 503


def test_a_plain_api_key_still_authenticates(client: TestClient, master_key_header: dict[str, str]) -> None:
    created = client.post(KEYS, json={"user_id": "plain"}, headers=master_key_header).json()

    assert _chat(client, {API_KEY_HEADER: f"Bearer {created['key']}"}).status_code == 200


def test_with_federation_off_a_token_is_an_unknown_key(
    test_config: GatewayConfig, clean_database: None, master_key_header: dict[str, str]
) -> None:
    for client in build_test_client(test_config):
        before = _failures("invalid_key")

        refused = _chat(client, _bearer(_token()))

        assert refused.status_code == 401
        assert refused.json()["detail"] == "Invalid API key"
        assert _failures("invalid_key") == before + 1
