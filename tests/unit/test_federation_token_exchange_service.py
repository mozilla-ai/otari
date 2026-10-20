"""The token exchange service, on a Unit of Work over a stand-in session and in-memory repositories.

Pins how it is built and what it does with its one transaction: the lookups run in a block, a refusal rolls the
block back, and nothing is ever added to the session.
"""

import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.api_keys import APIKey, APIKeyFederationRule
from gateway.repositories.api_keys import ApiKeyRepositories
from gateway.services.api_keys import (
    JWT_BEARER_GRANT,
    FederatedIdentity,
    FederationGrantError,
    FederationTokenExchange,
    read_access_token,
)
from gateway.services.secret_box import generate_secret_key

ISSUER = "https://issuer.example.com"
AUDIENCE = "https://otari.example.com"
EMAIL = "bot@example.com"
SECRET = generate_secret_key()
ORGANIZATION = uuid.uuid4()
WORKSPACE = uuid.uuid4()
# Shaped like a JWT, so the exchange reaches the verifier, which is stubbed.
ASSERTION = "eyJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJodHRwczovL2lzc3Vlci5leGFtcGxlLmNvbSJ9.sig"


class _Rows:
    """In-memory stand-ins for the two repositories the exchange reads, recording each read."""

    def __init__(self, rule: APIKeyFederationRule | None, key: APIKey | None) -> None:
        self.rule = rule
        self.key = key
        self.reads: list[str] = []

    async def get_rule(self, rule_id: str) -> APIKeyFederationRule | None:
        self.reads.append(f"rule:{rule_id}")
        return self.rule if self.rule is not None and self.rule.id == rule_id else None

    async def get_key(self, key_id: str) -> APIKey | None:
        self.reads.append(f"key:{key_id}")
        return self.key if self.key is not None and self.key.id == key_id else None


def _key(**overrides: Any) -> APIKey:
    fields: dict[str, Any] = {
        "id": "key-1",
        "key_hash": "h",
        "workspace_id": WORKSPACE,
        "is_active": True,
        "expires_at": None,
        "internal_secret": None,
    }
    fields.update(overrides)
    return APIKey(**fields)


def _rule() -> APIKeyFederationRule:
    return APIKeyFederationRule(
        id="rule-1", api_key_id="key-1", issuer=ISSUER, audience=AUDIENCE, subject=None, email=EMAIL
    )


def _build(rows: _Rows, *, identity_exp: int | None = None) -> tuple[FederationTokenExchange, MagicMock]:
    session = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    uow = UnitOfWork(session)
    repositories = ApiKeyRepositories.on(uow)
    object.__setattr__(repositories.federation_rules, "get", rows.get_rule)
    object.__setattr__(repositories.keys, "get", rows.get_key)
    verifier = MagicMock()
    verifier.verify = AsyncMock(
        return_value=FederatedIdentity(
            issuer=ISSUER,
            audiences=(AUDIENCE,),
            subject="sub",
            email=EMAIL,
            expires_at=identity_exp if identity_exp is not None else int(time.time()) + 3600,
        )
    )

    async def organization_of_workspace(workspace_id: uuid.UUID) -> uuid.UUID | None:
        return ORGANIZATION if workspace_id == WORKSPACE else None

    exchange = FederationTokenExchange(
        uow,
        repositories,
        verifier=verifier,
        organization_of_workspace=organization_of_workspace,
        ttl_seconds=600,
        secrets=[SECRET],
    )
    return exchange, session


async def _exchange(exchange: FederationTokenExchange, **overrides: Any) -> tuple[str, int]:
    grant: dict[str, Any] = {
        "grant_type": JWT_BEARER_GRANT,
        "assertion": ASSERTION,
        "federation_rule_id": "rule-1",
        "organization_id": str(ORGANIZATION),
        "workspace_id": None,
        "service_account_id": None,
    }
    grant.update(overrides)
    return await exchange.exchange(**grant)


@pytest.mark.asyncio
async def test_a_matching_grant_mints_a_token_for_the_rules_key_and_stages_nothing() -> None:
    rows = _Rows(_rule(), _key())
    exchange, session = _build(rows)

    token, expires_in = await _exchange(exchange)

    claims = read_access_token(token, [SECRET], now=time.time())
    assert (claims.rule_id, claims.api_key_id) == ("rule-1", "key-1")
    assert 595 <= expires_in <= 600
    assert rows.reads == ["rule:rule-1", "key:key-1"]
    session.add.assert_not_called()
    session.rollback.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rows", "overrides", "reason"),
    [
        (_Rows(None, _key()), {}, "federation_no_rule"),
        (_Rows(_rule(), None), {}, "federation_no_rule"),
        (_Rows(_rule(), _key(internal_secret="x")), {}, "federation_no_rule"),
        (_Rows(_rule(), _key()), {"organization_id": str(uuid.uuid4())}, "federation_grant_mismatch"),
        (_Rows(_rule(), _key()), {"workspace_id": str(uuid.uuid4())}, "federation_grant_mismatch"),
        (_Rows(_rule(), _key()), {"service_account_id": "key-2"}, "federation_grant_mismatch"),
        (_Rows(_rule(), _key(is_active=False)), {}, "inactive_key"),
        (
            _Rows(_rule(), _key(expires_at=datetime.now(UTC) - timedelta(minutes=1))),
            {},
            "expired_key",
        ),
    ],
    ids=["no-rule", "no-key", "internal-key", "organization", "workspace", "service-account", "inactive", "expired"],
)
async def test_a_refusal_rolls_the_block_back_and_answers_invalid_grant(
    rows: _Rows, overrides: dict[str, Any], reason: str
) -> None:
    exchange, session = _build(rows)

    with pytest.raises(FederationGrantError) as refused:
        await _exchange(exchange, **overrides)

    assert (refused.value.error, refused.value.status_code, refused.value.reason) == ("invalid_grant", 401, reason)
    session.rollback.assert_awaited_once()
    session.commit.assert_not_awaited()
    session.add.assert_not_called()


@pytest.mark.asyncio
async def test_a_bad_request_is_refused_before_any_read() -> None:
    rows = _Rows(_rule(), _key())
    exchange, session = _build(rows)

    for overrides, error in (
        ({"grant_type": "password"}, "unsupported_grant_type"),
        ({"assertion": None}, "invalid_request"),
        ({"federation_rule_id": None}, "invalid_request"),
        ({"assertion": "not-a-jwt"}, "invalid_grant"),
    ):
        with pytest.raises(FederationGrantError) as refused:
            await _exchange(exchange, **overrides)
        assert refused.value.error == error

    assert rows.reads == []
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_identity_token_about_to_expire_is_not_exchanged() -> None:
    exchange, _session = _build(_Rows(_rule(), _key()), identity_exp=int(time.time()))

    with pytest.raises(FederationGrantError) as refused:
        await _exchange(exchange)

    assert refused.value.reason == "federation_expired"
