"""A service key's control plane signs end-user tokens offline, and each authenticates as a key bound to one end user.

The token's first use materializes that key, through the same service the mint
route uses and within the same bounds; every later use authenticates as it. Every
refusal is the 401 an unknown key gets, and the auth-failure metric names why.
"""

import asyncio
import time
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session

from gateway.api.deps import build_budget_service
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.api_keys import APIKey, EndUserSigningKey, EndUserTokenRedemption
from gateway.models.usage import UsageLog
from gateway.repositories.api_keys import ApiKeyRepository
from gateway.services.api_keys import END_USER_TOKEN_PREFIX

from .conftest import MODEL_NAME, _to_async_url, build_test_client
from .test_end_user_keys import KEYS, _bearer, _chat, _end_user, _service_key, _set_key, _setup

SIGNING_KEYS = f"{API_ROOT}/end-user-signing-keys"
END_USER_KEYS = f"{API_ROOT}/end-user-keys"
TYP = "otari-eut+jwt"


@pytest.fixture
def token_config(test_config: GatewayConfig) -> GatewayConfig:
    return test_config.model_copy(update={"end_user_tokens_enabled": True})


@pytest.fixture
def client(token_config: GatewayConfig, clean_database: None) -> Generator[TestClient]:
    """A client on a deployment with signed end-user tokens on."""
    yield from build_test_client(token_config)


@pytest.fixture
def failures(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The auth-failure reasons recorded during the test."""
    recorded: list[str] = []
    monkeypatch.setattr("gateway.api.deps.record_auth_failure", recorded.append)
    return recorded


def _pem(private_key: Ed25519PrivateKey) -> str:
    return private_key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode()


def _register(client: TestClient, service: dict[str, str], name: str | None = None) -> tuple[str, Ed25519PrivateKey]:
    private_key = Ed25519PrivateKey.generate()
    response = client.post(SIGNING_KEYS, json={"public_key": _pem(private_key), "name": name}, headers=service)
    assert response.status_code == 201, response.text
    return response.json()["kid"], private_key


def _token(
    private_key: Any,
    kid: str,
    parent_id: str,
    *,
    ttl: int = 600,
    iat: int | None = None,
    algorithm: str = "EdDSA",
    typ: str = TYP,
    prefix: str = END_USER_TOKEN_PREFIX,
    **claims: Any,
) -> str:
    issued = int(time.time()) if iat is None else iat
    payload = {
        "iss": f"otari:api-key:{parent_id}",
        "sub": "run-1",
        "iat": issued,
        "exp": issued + ttl,
        "jti": "run-1-attempt-1",
        **claims,
    }
    return prefix + jwt.encode(payload, private_key, algorithm=algorithm, headers={"kid": kid, "typ": typ})


def _signed(*args: Any, **kwargs: Any) -> dict[str, str]:
    return _bearer(_token(*args, **kwargs))


def _refused(client: TestClient, headers: dict[str, str], failures: list[str]) -> str:
    """Send a request that must be refused as an unknown key would be, and return the recorded reason."""
    failures.clear()
    response = _chat(client, headers)
    assert response.status_code == 401, response.text
    assert response.json()["detail"] == "Invalid API key"
    assert len(failures) == 1, failures
    return failures[0]


def _children(db_session: Session, service_id: str) -> list[APIKey]:
    db_session.expire_all()
    return db_session.query(APIKey).filter(APIKey.parent_key_id == service_id).all()


# =============================================================================
# Registering signing keys
# =============================================================================


def test_a_service_key_registers_lists_and_deletes_its_signing_keys(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _setup(client, master_key_header)
    service_id, service = _service_key(client, master_key_header)

    private_key = Ed25519PrivateKey.generate()
    created = client.post(SIGNING_KEYS, json={"public_key": _pem(private_key), "name": "2026-10"}, headers=service)
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["public_key"] == _pem(private_key)
    assert body["name"] == "2026-10"
    kid = body["kid"]
    second, _ = _register(client, service)
    assert kid != second

    listed = client.get(SIGNING_KEYS, headers=service)
    assert [row["kid"] for row in listed.json()] == [kid, second]
    operator = client.get(f"{KEYS}/{service_id}/end-user-signing-keys", headers=master_key_header)
    assert [row["kid"] for row in operator.json()] == [kid, second]

    assert client.delete(f"{SIGNING_KEYS}/{kid}", headers=service).status_code == 204
    assert client.delete(f"{SIGNING_KEYS}/{kid}", headers=service).status_code == 404
    revoked = client.delete(f"{KEYS}/{service_id}/end-user-signing-keys/{second}", headers=master_key_header)
    assert revoked.status_code == 204
    assert client.get(SIGNING_KEYS, headers=service).json() == []
    db_session.expire_all()
    assert db_session.query(EndUserSigningKey).count() == 0


def test_only_a_service_key_that_may_mint_registers_a_signing_key(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _setup(client, master_key_header)
    _, no_ttl = _service_key(client, master_key_header, owner="other", max_child_ttl_sec=None)
    plain = client.post(KEYS, json={"user_id": "plain"}, headers=master_key_header).json()["key"]
    body = {"public_key": _pem(Ed25519PrivateKey.generate())}

    assert client.post(SIGNING_KEYS, json=body, headers=no_ttl).status_code == 403
    assert client.post(SIGNING_KEYS, json=body, headers=_bearer(plain)).status_code == 403
    assert client.post(SIGNING_KEYS, json=body, headers=master_key_header).status_code == 401


def test_a_signing_key_must_be_an_ed25519_public_key(client: TestClient, master_key_header: dict[str, str]) -> None:
    _setup(client, master_key_header)
    _, service = _service_key(client, master_key_header)
    private_pem = Ed25519PrivateKey.generate().private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()).decode()

    assert client.post(SIGNING_KEYS, json={"public_key": private_pem}, headers=service).status_code == 400
    assert client.post(SIGNING_KEYS, json={"public_key": "not a key"}, headers=service).status_code == 400


def test_a_service_key_holds_at_most_five_signing_keys(client: TestClient, master_key_header: dict[str, str]) -> None:
    _setup(client, master_key_header)
    _, service = _service_key(client, master_key_header)
    kids = [_register(client, service)[0] for _ in range(5)]

    over = client.post(SIGNING_KEYS, json={"public_key": _pem(Ed25519PrivateKey.generate())}, headers=service)
    assert over.status_code == 409, over.text
    assert client.delete(f"{SIGNING_KEYS}/{kids[0]}", headers=service).status_code == 204
    _register(client, service)


def test_a_service_key_reaches_only_its_own_signing_keys(client: TestClient, master_key_header: dict[str, str]) -> None:
    _setup(client, master_key_header)
    _, service = _service_key(client, master_key_header)
    _, sibling = _service_key(client, master_key_header, owner="agent")
    kid, _ = _register(client, service)

    assert client.get(SIGNING_KEYS, headers=sibling).json() == []
    assert client.delete(f"{SIGNING_KEYS}/{kid}", headers=sibling).status_code == 404
    assert [row["kid"] for row in client.get(SIGNING_KEYS, headers=service).json()] == [kid]


def test_signed_tokens_are_off_unless_the_deployment_turns_them_on(
    test_config: GatewayConfig,
    token_config: GatewayConfig,
    clean_database: None,
    master_key_header: dict[str, str],
    failures: list[str],
) -> None:
    on_clients = build_test_client(token_config)
    on = next(on_clients)
    try:
        _setup(on, master_key_header)
        service_id, service = _service_key(on, master_key_header)
        kid, private_key = _register(on, service)
        token = _signed(private_key, kid, service_id)
        assert _chat(on, token).status_code == 200
    finally:
        on_clients.close()

    off_clients = build_test_client(test_config)
    off = next(off_clients)
    try:
        registered = off.post(SIGNING_KEYS, json={"public_key": _pem(Ed25519PrivateKey.generate())}, headers=service)
        assert registered.status_code == 403, registered.text
        assert _refused(off, token, failures) == "end_user_token_disabled"
        assert off.get(SIGNING_KEYS, headers=service).status_code == 200
    finally:
        off_clients.close()


# =============================================================================
# The request path
# =============================================================================


def test_a_signed_token_authenticates_as_a_key_bound_to_its_end_user(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _setup(client, master_key_header)
    service_id, service = _service_key(client, master_key_header, reject_user_mismatch=True)
    kid, private_key = _register(client, service)
    token = _signed(private_key, kid, service_id, ttl=900, budget_id="run-five", tags={"agent": "bug-fix"})

    first = _chat(client, token, user='{"device_id":"telemetry"}', metadata={"agent": "spoofed", "team": "x"})
    assert first.status_code == 200, first.text
    assert _chat(client, token).status_code == 200

    end_user = _end_user(db_session, "run-1")
    assert end_user.budget_id == "run-five"
    assert end_user.current_requests == 2
    [child] = _children(db_session, service_id)
    assert child.user_id == end_user.user_id
    assert child.key_prefix == END_USER_TOKEN_PREFIX
    assert child.tags == {"agent": "bug-fix"}
    assert child.expires_at is not None
    assert child.expires_at.astimezone(UTC) - datetime.now(UTC) <= timedelta(seconds=900)
    rows = db_session.query(UsageLog).filter(UsageLog.user_id == end_user.user_id).all()
    assert {row.api_key_id for row in rows} == {child.id}
    assert rows[0].tags == {"agent": "bug-fix", "team": "x"}

    found = client.get(END_USER_KEYS, params={"jti": "run-1-attempt-1"}, headers=service)
    assert [row["id"] for row in found.json()] == [child.id]
    assert client.get(END_USER_KEYS, params={"jti": "other"}, headers=service).json() == []
    operator = client.get(
        f"{KEYS}/{service_id}/end-user-keys", params={"jti": "run-1-attempt-1"}, headers=master_key_header
    )
    assert [row["id"] for row in operator.json()] == [child.id]


def test_a_signed_token_is_refused_by_its_end_users_budget(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _setup(client, master_key_header)
    service_id, service = _service_key(client, master_key_header)
    kid, private_key = _register(client, service)
    token = _signed(private_key, kid, service_id, budget_id="run-one")

    assert _chat(client, token).status_code == 200
    refused = _chat(client, token)
    assert refused.status_code == 403, refused.text


def test_signed_tokens_and_minted_keys_are_pooled_under_the_service_keys_ceiling(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _setup(client, master_key_header)
    service_id, service = _service_key(client, master_key_header)
    client.put(
        f"{API_ROOT}/budgets/agent-day",
        json={"budget_duration_sec": 86400, "request_limit": 2},
        headers=master_key_header,
    )
    ceiling = client.post(
        f"{API_ROOT}/scoped-budgets",
        json={"scope_type": "api_token", "scope_id": service_id, "budget_id": "agent-day"},
        headers=master_key_header,
    )
    assert ceiling.status_code == 200, ceiling.text
    kid, private_key = _register(client, service)
    minted = client.post(END_USER_KEYS, json={"external_id": "run-2", "expires_in_sec": 600}, headers=service)
    assert minted.status_code == 201, minted.text

    assert _chat(client, _signed(private_key, kid, service_id)).status_code == 200
    assert _chat(client, _bearer(minted.json()["key"])).status_code == 200
    assert _chat(client, _signed(private_key, kid, service_id, sub="run-3", jti="run-3")).status_code == 403


def test_a_signed_tokens_models_are_cut_to_its_service_keys(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _setup(client, master_key_header)
    service_id, service = _service_key(client, master_key_header)
    kid, private_key = _register(client, service)
    token = _signed(private_key, kid, service_id, allowed_models=[MODEL_NAME])
    assert _chat(client, token).status_code == 200

    narrowed = client.patch(
        f"{KEYS}/{service_id}", json={"allowed_models": ["gemini:other-model"]}, headers=master_key_header
    )
    assert narrowed.status_code == 200, narrowed.text
    assert _chat(client, token).status_code == 403


def test_concurrent_first_uses_of_a_token_materialize_one_key(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
    postgres_url: str,
    token_config: GatewayConfig,
) -> None:
    _setup(client, master_key_header)
    service_id, service = _service_key(client, master_key_header)
    kid, private_key = _register(client, service)
    token = _token(private_key, kid, service_id)
    original = ApiKeyRepository.find_redemption
    original_add = ApiKeyRepository.add_token_child
    inserts: list[bool] = []

    async def add_and_record(self: ApiKeyRepository, **fields: Any) -> Any:
        added = await original_add(self, **fields)
        inserts.append(added is not None)
        return added

    async def race() -> list[str]:
        barrier = asyncio.Barrier(3)
        gated: set[asyncio.Task[Any] | None] = set()

        async def find_then_wait(self: ApiKeyRepository, parent_id: str, jti: str) -> Any:
            # Every first use looks before any inserts, so all of them race to insert.
            found = await original(self, parent_id, jti)
            task = asyncio.current_task()
            if task not in gated:
                gated.add(task)
                await barrier.wait()
            return found

        engine = create_async_engine(_to_async_url(postgres_url))
        sessions = async_sessionmaker(engine, expire_on_commit=False)

        async def authenticate() -> str:
            async with sessions() as session:
                return (await build_budget_service(session).authenticate_end_user_token(token, token_config)).id

        try:
            with (
                patch.object(ApiKeyRepository, "find_redemption", find_then_wait),
                patch.object(ApiKeyRepository, "add_token_child", add_and_record),
            ):
                return list(await asyncio.gather(authenticate(), authenticate(), authenticate()))
        finally:
            await engine.dispose()

    ids = asyncio.run(race())
    assert len(set(ids)) == 1
    # Every use tried to insert, and the unique key turned two of them back.
    assert sorted(inserts) == [False, False, True]
    [child] = _children(db_session, service_id)
    assert child.id == ids[0]
    assert db_session.query(EndUserTokenRedemption).count() == 1
    assert _chat(client, _bearer(token)).status_code == 200


# =============================================================================
# Revocation
# =============================================================================


def test_revoking_the_materialized_key_refuses_its_token(
    client: TestClient, master_key_header: dict[str, str], db_session: Session, failures: list[str]
) -> None:
    _setup(client, master_key_header)
    service_id, service = _service_key(client, master_key_header)
    kid, private_key = _register(client, service)
    token = _signed(private_key, kid, service_id)
    other = _signed(private_key, kid, service_id, sub="run-2", jti="run-2")
    assert _chat(client, token).status_code == 200
    assert _chat(client, other).status_code == 200

    [child_id] = [
        row["id"] for row in client.get(END_USER_KEYS, params={"jti": "run-1-attempt-1"}, headers=service).json()
    ]
    revoked = client.patch(f"{END_USER_KEYS}/{child_id}", json={"is_active": False}, headers=service)
    assert revoked.status_code == 200, revoked.text
    assert _refused(client, token, failures) == "end_user_token_inactive_key"
    assert _chat(client, other).status_code == 200


def test_deleting_the_signing_key_refuses_every_token_it_signed(
    client: TestClient, master_key_header: dict[str, str], db_session: Session, failures: list[str]
) -> None:
    _setup(client, master_key_header)
    service_id, service = _service_key(client, master_key_header)
    kid, private_key = _register(client, service)
    kept_kid, kept_key = _register(client, service)
    used = _signed(private_key, kid, service_id)
    unused = _signed(private_key, kid, service_id, sub="run-2", jti="run-2")
    kept = _signed(kept_key, kept_kid, service_id, sub="run-3", jti="run-3")
    assert _chat(client, used).status_code == 200
    assert _chat(client, kept).status_code == 200

    assert client.delete(f"{SIGNING_KEYS}/{kid}", headers=service).status_code == 204
    assert _refused(client, used, failures) == "end_user_token_unknown_kid"
    assert _refused(client, unused, failures) == "end_user_token_unknown_kid"
    assert _chat(client, kept).status_code == 200
    by_user = {child.key_name: child.is_active for child in _children(db_session, service_id)}
    assert by_user == {"run-1": False, "run-3": True}


def test_a_token_whose_key_was_deleted_is_not_materialized_again(
    client: TestClient, master_key_header: dict[str, str], db_session: Session, failures: list[str]
) -> None:
    _setup(client, master_key_header)
    service_id, service = _service_key(client, master_key_header)
    kid, private_key = _register(client, service)
    token = _signed(private_key, kid, service_id)
    assert _chat(client, token).status_code == 200
    [child] = _children(db_session, service_id)

    assert client.delete(f"{KEYS}/{child.id}", headers=master_key_header).status_code == 204
    assert _refused(client, token, failures) == "end_user_token_key_deleted"
    assert _children(db_session, service_id) == []


def test_tokens_stop_with_their_service_key(
    client: TestClient, master_key_header: dict[str, str], db_session: Session, failures: list[str]
) -> None:
    _setup(client, master_key_header)
    service_id, service = _service_key(client, master_key_header)
    kid, private_key = _register(client, service)
    token = _signed(private_key, kid, service_id)
    assert _chat(client, token).status_code == 200

    client.patch(f"{KEYS}/{service_id}", json={"is_active": False}, headers=master_key_header)
    assert _refused(client, token, failures) == "end_user_token_issuer_unusable"
    client.patch(f"{KEYS}/{service_id}", json={"is_active": True}, headers=master_key_header)
    assert _chat(client, token).status_code == 200

    cleared = client.patch(f"{KEYS}/{service_id}", json={"max_child_ttl_sec": None}, headers=master_key_header)
    assert cleared.status_code == 200, cleared.text
    assert _refused(client, token, failures) == "end_user_token_not_allowed"
    client.patch(f"{KEYS}/{service_id}", json={"max_child_ttl_sec": 3600}, headers=master_key_header)

    assert client.delete(f"{KEYS}/{service_id}", headers=master_key_header).status_code == 204
    assert _refused(client, token, failures) == "end_user_token_unknown_kid"
    db_session.expire_all()
    assert db_session.query(EndUserTokenRedemption).count() == 0


def test_an_expired_materialized_key_refuses_its_token(
    client: TestClient, master_key_header: dict[str, str], db_session: Session, failures: list[str]
) -> None:
    _setup(client, master_key_header)
    service_id, service = _service_key(client, master_key_header)
    kid, private_key = _register(client, service)
    token = _signed(private_key, kid, service_id)
    assert _chat(client, token).status_code == 200
    [child] = _children(db_session, service_id)

    _set_key(db_session, child.id, expires_at=datetime.now(UTC) - timedelta(seconds=1))
    assert _refused(client, token, failures) == "end_user_token_expired_key"


# =============================================================================
# Refusals
# =============================================================================


def test_a_token_is_held_to_the_bounds_of_the_mint_route(
    client: TestClient, master_key_header: dict[str, str], db_session: Session, failures: list[str]
) -> None:
    _setup(client, master_key_header)
    service_id, service = _service_key(client, master_key_header, allowed_models=[MODEL_NAME])
    kid, key = _register(client, service)

    assert _refused(client, _signed(key, kid, service_id, ttl=3601), failures) == "end_user_token_lifetime"
    assert _refused(client, _signed(key, kid, service_id, budget_id="run-other"), failures) == "end_user_token_budget"
    assert _refused(client, _signed(key, kid, service_id, budget_id="missing"), failures) == "end_user_token_budget"
    wider = _signed(key, kid, service_id, allowed_models=["gemini:*"])
    assert _refused(client, wider, failures) == "end_user_token_models"
    assert _refused(client, _signed(key, kid, service_id, sub="x" * 257), failures) == "end_user_token_claims"
    many_tags = _signed(key, kid, service_id, tags={f"k{n}": "v" for n in range(17)})
    assert _refused(client, many_tags, failures) == "end_user_token_claims"
    assert _refused(client, _signed(key, kid, service_id, jti="j" * 129), failures) == "end_user_token_claims"
    assert _refused(client, _signed(key, kid, service_id, aud="otari"), failures) == "end_user_token_claims"
    assert _children(db_session, service_id) == []

    assert _chat(client, _signed(key, kid, service_id, ttl=3600, allowed_models=[MODEL_NAME])).status_code == 200


def test_a_forged_or_stale_token_is_refused(
    client: TestClient, master_key_header: dict[str, str], db_session: Session, failures: list[str]
) -> None:
    _setup(client, master_key_header)
    service_id, service = _service_key(client, master_key_header)
    kid, key = _register(client, service)
    now = int(time.time())

    hmac = _signed("a-shared-secret-of-at-least-thirty-two-bytes", kid, service_id, algorithm="HS256")
    assert _refused(client, hmac, failures) == "end_user_token_alg"
    assert _refused(client, _signed(key, kid, service_id, typ="JWT"), failures) == "end_user_token_typ"
    forged = _signed(Ed25519PrivateKey.generate(), kid, service_id)
    assert _refused(client, forged, failures) == "end_user_token_bad_signature"
    assert _refused(client, _signed(key, kid, service_id, iat=now - 700), failures) == "end_user_token_expired"
    future = _signed(key, kid, service_id, iat=now + 300)
    assert _refused(client, future, failures) == "end_user_token_future_iat"
    assert _refused(client, _signed(key, "no-such-kid", service_id), failures) == "end_user_token_unknown_kid"
    assert _refused(client, _bearer(END_USER_TOKEN_PREFIX + "garbage"), failures) == "end_user_token_malformed"
    assert _children(db_session, service_id) == []


def test_a_token_cannot_be_signed_for_another_service_key(
    client: TestClient, master_key_header: dict[str, str], db_session: Session, failures: list[str]
) -> None:
    """A control plane holding one key's signing key cannot reach a sibling key, even one with the same owner."""
    _setup(client, master_key_header)
    service_id, service = _service_key(client, master_key_header)
    sibling_id, sibling = _service_key(client, master_key_header, owner="agent")
    kid, key = _register(client, service)
    _register(client, sibling)

    assert _refused(client, _signed(key, kid, sibling_id), failures) == "end_user_token_issuer"
    assert _children(db_session, sibling_id) == []
    assert _chat(client, _signed(key, kid, service_id)).status_code == 200


def test_a_jti_is_one_token(
    client: TestClient, master_key_header: dict[str, str], db_session: Session, failures: list[str]
) -> None:
    _setup(client, master_key_header)
    service_id, service = _service_key(client, master_key_header)
    kid, key = _register(client, service)
    rotated_kid, rotated = _register(client, service)
    issued = int(time.time())
    assert _chat(client, _signed(key, kid, service_id, iat=issued)).status_code == 200

    assert _chat(client, _signed(key, kid, service_id, iat=issued)).status_code == 200
    other_user = _signed(key, kid, service_id, iat=issued, sub="run-2")
    assert _refused(client, other_user, failures) == "end_user_token_claims_mismatch"
    longer = _signed(key, kid, service_id, iat=issued, ttl=900)
    assert _refused(client, longer, failures) == "end_user_token_claims_mismatch"
    tagged = _signed(key, kid, service_id, iat=issued, tags={"agent": "x"})
    assert _refused(client, tagged, failures) == "end_user_token_claims_mismatch"
    resigned = _signed(rotated, rotated_kid, service_id, iat=issued)
    assert _refused(client, resigned, failures) == "end_user_token_claims_mismatch"
    assert len(_children(db_session, service_id)) == 1


# =============================================================================
# Coexistence with workload identity
# =============================================================================


def test_an_openid_shaped_token_is_never_read_as_an_end_user_token(
    client: TestClient, master_key_header: dict[str, str], db_session: Session, failures: list[str]
) -> None:
    """Workload identity reads a bare JWT whose ``iss`` is an https issuer; that shape never reaches this path.

    Signed with a registered key, under its kid and type, it is still looked up as a key secret, because it lacks
    the prefix.
    """
    _setup(client, master_key_header)
    service_id, service = _service_key(client, master_key_header)
    kid, key = _register(client, service)

    bare = _token(key, kid, service_id, prefix="", iss="https://accounts.google.com")
    assert bare.count(".") == 2
    assert _refused(client, _bearer(bare), failures) == "invalid_key"
    bare_otari = _token(key, kid, service_id, prefix="")
    assert _refused(client, _bearer(bare_otari), failures) == "invalid_key"
    assert _children(db_session, service_id) == []


def test_an_end_user_token_naming_an_openid_issuer_is_refused(
    client: TestClient, master_key_header: dict[str, str], failures: list[str]
) -> None:
    _setup(client, master_key_header)
    service_id, service = _service_key(client, master_key_header)
    kid, key = _register(client, service)

    token = _token(key, kid, service_id, iss="https://accounts.google.com")
    assert token.count(".") == 3
    assert _refused(client, _bearer(token), failures) == "end_user_token_issuer"
