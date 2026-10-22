"""Members' own federation rules: ``federation_member_rules`` lets a member give their own key a rule.

A shared deployment cannot trust one list of issuers for every tenant, and its operator should not have to write every
tenant's rules. So a member names the issuer on the rule itself, and the rule is only ever checked in the token
exchange that names it by id. The suite pins both halves: the exchange works with an issuer the deployment does not
list, and a member's rule never answers a bare identity token, never blocks or is blocked by another key's rule, and
stops working when the setting is turned off.

The issuer's discovery document and keys are served by a stub, so nothing reaches the network.
"""

import time
import uuid
from collections.abc import Callable, Generator
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
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.settings.api_keys import FederationIssuer
from gateway.models.tenancy import DashboardSession, OrganizationMember, User, WorkspaceMember
from gateway.services.api_keys import FederatedTokenVerifier
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.secret_box import generate_secret_key

from .conftest import MODEL_NAME, build_test_client

ISSUER = "https://issuer.example.com"
AUDIENCE = "https://api.anthropic.com"
EMAIL = "hackbot@project.iam.gserviceaccount.com"
OWN_KEYS = f"{API_ROOT}/organizations/me/keys"
KEYS = f"{API_ROOT}/keys"
TOKEN_URL = f"{API_ROOT}/oauth/token"
GRANT = "urn:ietf:params:oauth:grant-type:jwt-bearer"

SIGNING_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class _Issuer:
    """Serves the issuer's discovery document and keys, and records every URL fetched."""

    def __init__(self) -> None:
        self.fetched: list[str] = []

    def transport(self, _url: str) -> httpx.AsyncBaseTransport:
        return httpx.MockTransport(self._respond)

    def _respond(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.fetched.append(url)
        if url == f"{ISSUER}/.well-known/openid-configuration":
            return httpx.Response(200, json={"issuer": ISSUER, "jwks_uri": f"{ISSUER}/keys"})
        if url == f"{ISSUER}/keys":
            jwk = RSAAlgorithm.to_jwk(SIGNING_KEY.public_key(), as_dict=True)
            return httpx.Response(200, json={"keys": [{**jwk, "kid": "k1", "alg": "RS256", "use": "sig"}]})
        return httpx.Response(404)


def _assertion(**overrides: Any) -> str:
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "113051416744238958061",
        "email": EMAIL,
        "email_verified": True,
        "iat": now,
        "exp": now + 3600,
    }
    claims.update(overrides)
    return jwt.encode(claims, SIGNING_KEY, algorithm="RS256", headers={"kid": "k1"})


@pytest.fixture
def issuer() -> _Issuer:
    return _Issuer()


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # One secret per test, so an access token minted by one client is still readable by a second over the same data.
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


def _client(config: GatewayConfig, issuer: _Issuer) -> Generator[TestClient]:
    for test_client in build_test_client(config):
        test_client.app.state.federated_token_verifier = FederatedTokenVerifier(  # type: ignore[attr-defined]
            config.federation_issuers,
            member_issuers=config.federation_member_rules,
            transport_factory=issuer.transport,
        )
        yield test_client


@pytest.fixture
def client(
    test_config: GatewayConfig, clean_database: None, issuer: _Issuer, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient]:
    """A deployment that lists no issuer and lets members name their own, as a hosted one would."""
    config = test_config.model_copy(update={"federation_member_rules": True}, deep=True)
    yield from _client(config, issuer)


def _own_key(client: TestClient, headers: dict[str, str]) -> str:
    created = client.post(OWN_KEYS, json={"key_name": "hackbot"}, headers=headers)
    assert created.status_code == 200, created.text
    return str(created.json()["id"])


def _own_rule(client: TestClient, headers: dict[str, str], key_id: str, **overrides: Any) -> Any:
    body = {"issuer": ISSUER, "audience": AUDIENCE, "email": EMAIL, **overrides}
    return client.post(f"{OWN_KEYS}/{key_id}/federation-rules", json=body, headers=headers)


def _organization_id(client: TestClient, headers: dict[str, str]) -> str:
    return str(client.get(f"{API_ROOT}/organizations/me", headers=headers).json()["organization"]["id"])


def _exchange(client: TestClient, rule_id: str, organization_id: str, **overrides: Any) -> Any:
    body = {
        "grant_type": GRANT,
        "assertion": _assertion(),
        "federation_rule_id": rule_id,
        "organization_id": organization_id,
        **overrides,
    }
    return client.post(TOKEN_URL, json=body)


async def _mock_acompletion(**_kwargs: Any) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-x",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _chat(client: TestClient, token: str) -> Any:
    body = {"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}]}
    with patch("gateway.api.routes.chat.acompletion") as mock:
        mock.side_effect = _mock_acompletion
        return client.post(f"{API_ROOT}/chat/completions", json=body, headers={"Authorization": f"Bearer {token}"})


# A member's rule, end to end


def test_a_members_rule_exchanges_with_an_issuer_the_deployment_does_not_list(
    client: TestClient, master_key_header: dict[str, str], issuer: _Issuer
) -> None:
    key_id = _own_key(client, master_key_header)
    rule = _own_rule(client, master_key_header, key_id)
    assert rule.status_code == 201, rule.text
    assert rule.json()["member_managed"] is True

    exchanged = _exchange(client, rule.json()["id"], _organization_id(client, master_key_header))

    assert exchanged.status_code == 200, exchanged.text
    assert f"{ISSUER}/.well-known/openid-configuration" in issuer.fetched
    assert _chat(client, exchanged.json()["access_token"]).status_code == 200


def test_a_members_rule_is_listed_and_deleted_on_their_own_key(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    key_id = _own_key(client, master_key_header)
    rule_id = _own_rule(client, master_key_header, key_id).json()["id"]

    listed = client.get(f"{OWN_KEYS}/{key_id}/federation-rules", headers=master_key_header)
    assert [row["id"] for row in listed.json()] == [rule_id]

    deleted = client.delete(f"{OWN_KEYS}/{key_id}/federation-rules/{rule_id}", headers=master_key_header)
    assert deleted.status_code == 204
    assert _exchange(client, rule_id, _organization_id(client, master_key_header)).status_code == 401


def test_a_token_from_another_issuer_does_not_match_a_members_rule(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    key_id = _own_key(client, master_key_header)
    rule_id = _own_rule(client, master_key_header, key_id).json()["id"]

    # The issuer a member trusts is the rule's; a token naming any other is refused before its keys are fetched.
    other = _assertion(iss="https://attacker.example.com")
    exchanged = _exchange(client, rule_id, _organization_id(client, master_key_header), assertion=other)

    assert exchanged.status_code == 401
    assert exchanged.json()["error"] == "invalid_grant"


def _session(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session], role: str
) -> dict[str, str]:
    """A dashboard session for a new identity holding ``role`` in the default organization and its workspace."""
    organization_id = uuid.UUID(_organization_id(client, master_key_header))
    listed = client.get(f"{API_ROOT}/workspaces?skip=0&limit=1", headers=master_key_header)
    workspace_id = uuid.UUID(listed.json()["data"][0]["id"])
    token = f"otari-sess-{role}-{uuid.uuid4()}"
    session = db_session_factory()
    try:
        user = User(
            email=f"{role}-{uuid.uuid4()}@example.test", full_name=role.title(), active_organization_id=organization_id
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        session.add(OrganizationMember(organization_id=organization_id, user_id=user.id, role=role, status="active"))
        session.add(WorkspaceMember(workspace_id=workspace_id, user_id=user.id, role="member", status="active"))
        session.add(
            DashboardSession(
                token_hash=hash_session_token(token),
                user_id=user.id,
                created_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(hours=1),
            )
        )
        session.commit()
    finally:
        session.close()
    return {"Cookie": f"{SESSION_COOKIE_NAME}={token}"}


@pytest.mark.parametrize(("role", "expected"), [("owner", 201), ("admin", 201), ("member", 403)])
def test_an_owner_or_admin_gives_their_own_key_a_rule_and_a_member_cannot(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    role: str,
    expected: int,
) -> None:
    caller = _session(client, master_key_header, db_session_factory, role)
    key_id = _own_key(client, caller)

    assert _own_rule(client, caller, key_id).status_code == expected


def test_an_admin_cannot_give_a_rule_to_a_colleagues_key(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    colleague = _session(client, master_key_header, db_session_factory, "member")
    admin = _session(client, master_key_header, db_session_factory, "admin")

    refused = _own_rule(client, admin, _own_key(client, colleague))

    assert refused.status_code == 404


def test_the_operator_gives_a_rule_to_any_key_in_their_organization(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    member = _session(client, master_key_header, db_session_factory, "member")
    key_id = _own_key(client, member)

    created = _own_rule(client, master_key_header, key_id)

    assert created.status_code == 201, created.text
    # The member cannot add a rule, but sees the one on their key and may take it back.
    listed = client.get(f"{OWN_KEYS}/{key_id}/federation-rules", headers=member)
    assert [rule["id"] for rule in listed.json()] == [created.json()["id"]]
    deleted = client.delete(f"{OWN_KEYS}/{key_id}/federation-rules/{created.json()['id']}", headers=member)
    assert deleted.status_code == 204


@pytest.mark.parametrize(
    "issuer_url",
    [
        "http://issuer.example.com",
        "https://user:pass@issuer.example.com",
        "https://issuer.example.com/?q=1",
        "https://[issuer.example.com",
    ],
    ids=["http", "credentials", "query", "malformed"],
)
def test_a_members_rule_names_an_https_issuer(
    client: TestClient, master_key_header: dict[str, str], issuer_url: str
) -> None:
    key_id = _own_key(client, master_key_header)

    refused = _own_rule(client, master_key_header, key_id, issuer=issuer_url)

    assert refused.status_code == 400, refused.text


def test_a_members_rule_still_names_a_subject_or_an_email(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    key_id = _own_key(client, master_key_header)

    refused = _own_rule(client, master_key_header, key_id, email=None)

    assert refused.status_code == 422


def test_the_same_rule_twice_on_one_key_is_a_conflict(client: TestClient, master_key_header: dict[str, str]) -> None:
    key_id = _own_key(client, master_key_header)
    assert _own_rule(client, master_key_header, key_id).status_code == 201

    assert _own_rule(client, master_key_header, key_id).status_code == 409


def test_two_keys_may_each_hold_a_members_rule_for_one_identity(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    # Each is named in its own exchange, so neither takes anything from the other, and a 409 would tell a member
    # that someone else's key trusts that identity.
    first = _own_rule(client, master_key_header, _own_key(client, master_key_header))
    second = _own_rule(client, master_key_header, _own_key(client, master_key_header))
    assert first.status_code == 201
    assert second.status_code == 201

    organization_id = _organization_id(client, master_key_header)
    assert _exchange(client, first.json()["id"], organization_id).status_code == 200
    assert _exchange(client, second.json()["id"], organization_id).status_code == 200


# The isolation a member's rule keeps


@pytest.fixture
def listing_client(
    test_config: GatewayConfig, clean_database: None, issuer: _Issuer, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient]:
    """A deployment that also lists the issuer, so a bare identity token from it is looked up across every key."""
    config = test_config.model_copy(
        update={
            "federation_member_rules": True,
            "federation_issuers": [FederationIssuer(issuer=ISSUER, jwks_uri=f"{ISSUER}/keys")],
        },
        deep=True,
    )
    yield from _client(config, issuer)


def test_a_members_rule_never_answers_a_bare_identity_token(
    listing_client: TestClient, master_key_header: dict[str, str]
) -> None:
    key_id = _own_key(listing_client, master_key_header)
    assert _own_rule(listing_client, master_key_header, key_id).status_code == 201

    assert _chat(listing_client, _assertion()).status_code == 401


def test_a_members_rule_does_not_block_or_take_the_operators_rule(
    listing_client: TestClient, master_key_header: dict[str, str]
) -> None:
    member_key = _own_key(listing_client, master_key_header)
    assert _own_rule(listing_client, master_key_header, member_key).status_code == 201
    created = listing_client.post(KEYS, json={"key_name": "operator", "user_id": "operator"}, headers=master_key_header)
    operator_key = created.json()["id"]

    operator_rule = listing_client.post(
        f"{KEYS}/{operator_key}/federation-rules",
        json={"issuer": ISSUER, "audience": AUDIENCE, "email": EMAIL},
        headers=master_key_header,
    )

    assert operator_rule.status_code == 201, operator_rule.text
    assert operator_rule.json()["member_managed"] is False
    # The bare token is the operator's key's alone; the member's rule for the same identity does not make it
    # ambiguous.
    assert _chat(listing_client, _assertion()).status_code == 200


# Turning it off


@pytest.fixture
def off_client(
    test_config: GatewayConfig, clean_database: None, issuer: _Issuer, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient]:
    """A deployment with an issuer listed and members' rules off, the default."""
    config = test_config.model_copy(
        update={"federation_issuers": [FederationIssuer(issuer=ISSUER, jwks_uri=f"{ISSUER}/keys")]}, deep=True
    )
    yield from _client(config, issuer)


def test_members_cannot_add_rules_when_the_setting_is_off(
    off_client: TestClient, master_key_header: dict[str, str]
) -> None:
    key_id = _own_key(off_client, master_key_header)

    refused = _own_rule(off_client, master_key_header, key_id)

    assert refused.status_code == 403


def test_turning_the_setting_off_ends_a_members_rule(
    client: TestClient,
    test_config: GatewayConfig,
    master_key_header: dict[str, str],
    issuer: _Issuer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_id = _own_key(client, master_key_header)
    rule_id = _own_rule(client, master_key_header, key_id).json()["id"]
    organization_id = _organization_id(client, master_key_header)
    access_token = _exchange(client, rule_id, organization_id).json()["access_token"]

    # The same database, now served by a deployment that lists the rule's issuer but admits no member's rule.
    config = test_config.model_copy(
        update={
            "federation_member_rules": False,
            "federation_issuers": [FederationIssuer(issuer=ISSUER, jwks_uri=f"{ISSUER}/keys")],
        },
        deep=True,
    )
    for off in _client(config, issuer):
        assert _exchange(off, rule_id, organization_id).status_code == 401
        assert _chat(off, access_token).status_code == 401
        # Taking back a grant never needs the setting.
        assert (
            off.delete(f"{OWN_KEYS}/{key_id}/federation-rules/{rule_id}", headers=master_key_header).status_code == 204
        )
