"""The access tokens ``POST /api/v1/oauth/token`` mints: signing, verification, expiry and the scope a grant names."""

import base64
import json
import time
import uuid

import pytest

from gateway.services.api_keys import (
    AccessTokenClaims,
    FederationRefusedError,
    access_token_expiry,
    grant_names_organization,
    is_access_token,
    mint_access_token,
    read_access_token,
    unverified_issuer,
)
from gateway.services.secret_box import generate_secret_key

SECRET = generate_secret_key()
OTHER_SECRET = generate_secret_key()
NOW = 1_800_000_000.0
CLAIMS = AccessTokenClaims(rule_id="rule-1", api_key_id="key-1", expires_at=int(NOW) + 600)


def _refusal(token: str, secrets: list[str], now: float = NOW) -> str:
    with pytest.raises(FederationRefusedError) as refused:
        read_access_token(token, secrets, now=now)
    return refused.value.reason


def _b64(data: dict[str, object]) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()


def test_a_minted_token_reads_back_its_claims() -> None:
    token = mint_access_token(CLAIMS, [SECRET])

    assert token.startswith("otari-fat1.")
    assert read_access_token(token, [SECRET], now=NOW) == CLAIMS


def test_an_expired_token_is_refused() -> None:
    token = mint_access_token(CLAIMS, [SECRET])

    assert _refusal(token, [SECRET], now=CLAIMS.expires_at) == "federation_access_token_expired"


def test_a_token_signed_under_another_secret_is_refused() -> None:
    assert _refusal(mint_access_token(CLAIMS, [OTHER_SECRET]), [SECRET]) == "federation_access_token_invalid"


def test_a_rotated_secret_list_verifies_tokens_from_each_and_signs_with_the_first() -> None:
    old = mint_access_token(CLAIMS, [SECRET])
    new = mint_access_token(CLAIMS, [OTHER_SECRET, SECRET])

    assert read_access_token(old, [OTHER_SECRET, SECRET], now=NOW) == CLAIMS
    assert _refusal(new, [SECRET]) == "federation_access_token_invalid"


def test_no_secret_means_nothing_is_minted_or_verified() -> None:
    with pytest.raises(ValueError, match="OTARI_SECRET_KEY"):
        mint_access_token(CLAIMS, [])
    assert _refusal(mint_access_token(CLAIMS, [SECRET]), []) == "federation_access_token_invalid"


def test_a_tampered_payload_is_refused() -> None:
    prefix, _, signature = mint_access_token(CLAIMS, [SECRET]).split(".")
    forged = _b64({"r": "rule-1", "k": "someone-elses-key", "exp": CLAIMS.expires_at})

    assert _refusal(f"{prefix}.{forged}.{signature}", [SECRET]) == "federation_access_token_invalid"


def test_a_signature_spelled_another_way_is_refused() -> None:
    token = mint_access_token(CLAIMS, [SECRET])

    assert _refusal(f"{token}=", [SECRET]) == "federation_access_token_invalid"
    assert _refusal(f"{token}.extra", [SECRET]) == "federation_access_token_invalid"


@pytest.mark.parametrize("token", ["otari-fat1.", "otari-fat1.abc", "otari-fat1.abc.def", "x" * 2000, "tk-abc"])
def test_garbage_is_refused(token: str) -> None:
    assert _refusal(token, [SECRET]) == "federation_access_token_invalid"


def test_an_access_token_and_a_jwt_are_told_apart_by_prefix_and_shape() -> None:
    token = mint_access_token(CLAIMS, [SECRET])
    jwt_like = f"{_b64({'alg': 'RS256'})}.{_b64({'iss': 'https://issuer.example.com'})}.sig"

    assert is_access_token(token)
    assert unverified_issuer(token) is None
    assert not is_access_token(jwt_like)
    assert unverified_issuer(jwt_like) == "https://issuer.example.com"


def test_the_expiry_is_the_ttl_capped_by_the_identity_token() -> None:
    now = time.time()

    assert access_token_expiry(now=now, ttl_seconds=600, identity_expires_at=int(now) + 3600) == int(now) + 600
    assert access_token_expiry(now=now, ttl_seconds=600, identity_expires_at=int(now) + 90) == int(now) + 90


ORGANIZATION = uuid.uuid4()


def test_a_grant_naming_no_organization_or_the_keys_own_is_accepted() -> None:
    assert grant_names_organization(organization_id=None, key_organization_id=ORGANIZATION)
    assert grant_names_organization(organization_id=str(ORGANIZATION).upper(), key_organization_id=ORGANIZATION)


@pytest.mark.parametrize("organization_id", [str(uuid.uuid4()), "org-1"], ids=["another", "not-a-uuid"])
def test_a_grant_naming_another_organization_is_refused(organization_id: str) -> None:
    assert not grant_names_organization(organization_id=organization_id, key_organization_id=ORGANIZATION)
