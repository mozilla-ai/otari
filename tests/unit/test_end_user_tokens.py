"""Verifying signed end-user tokens, with Ed25519 keys generated here.

Each refusal names its reason, which is all the auth-failure metric records; the
caller answers every one with the same 401.
"""

import base64
import json
import time
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ec import SECP256R1, generate_private_key
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.rsa import generate_private_key as generate_rsa_key
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

from gateway.exceptions.api_keys_exceptions import EndUserTokenRefusedError, SigningKeyInvalidError
from gateway.services.api_keys import (
    END_USER_TOKEN_PREFIX,
    claims_digest,
    is_end_user_token,
    read_token_header,
    verify_token_claims,
)
from gateway.services.api_keys._signed_tokens import CLOCK_SKEW_SEC, issuer_for, normalize_public_key

PARENT = "3f1d2c4b-0000-4000-8000-000000000001"
KID = "kid-1"
TYP = "otari-eut+jwt"


def _public_pem(private_key: Ed25519PrivateKey) -> str:
    return private_key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode()


SIGNER = Ed25519PrivateKey.generate()
PUBLIC_PEM = _public_pem(SIGNER)


def _claims(**overrides: Any) -> dict[str, Any]:
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": issuer_for(PARENT),
        "sub": "run-1",
        "iat": now,
        "exp": now + 600,
        "jti": "run-1-attempt-1",
    }
    claims.update(overrides)
    return {key: value for key, value in claims.items() if value is not None}


def _token(claims: dict[str, Any] | None = None, *, key: Any = SIGNER, algorithm: str = "EdDSA", **header: Any) -> str:
    headers = {"kid": KID, "typ": TYP, **header}
    return END_USER_TOKEN_PREFIX + jwt.encode(claims or _claims(), key, algorithm=algorithm, headers=headers)


def _b64(value: Any) -> str:
    raw = value if isinstance(value, bytes) else json.dumps(value).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _verify(credential: str, *, public_pem: str = PUBLIC_PEM, parent: str = PARENT) -> Any:
    return verify_token_claims(
        read_token_header(credential), public_key_pem=public_pem, parent_key_id=parent, now=time.time()
    )


def _refusal(credential: str, **kwargs: Any) -> str:
    with pytest.raises(EndUserTokenRefusedError) as refused:
        _verify(credential, **kwargs)
    assert refused.value.message == "Invalid API key"
    return refused.value.reason


def test_a_well_formed_token_verifies_to_its_claims() -> None:
    claims = _verify(_token(_claims(budget_id="run-one", tags={"agent": "bug-fix"}, allowed_models=["openai:*"])))
    assert (claims.sub, claims.jti, claims.budget_id) == ("run-1", "run-1-attempt-1", "run-one")
    assert claims.tags == {"agent": "bug-fix"}
    assert claims.allowed_models == ["openai:*"]
    assert read_token_header(_token()).kid == KID


# =============================================================================
# The header
# =============================================================================


@pytest.mark.parametrize("credential", ["", "tk-abc", "otari-eut1.", "otari-eut1.not-a-jws", "otari-eut1.a.b.c.d"])
def test_a_malformed_token_is_refused(credential: str) -> None:
    assert _refusal(credential) == "end_user_token_malformed"


def test_an_oversized_token_is_refused_before_it_is_parsed() -> None:
    assert _refusal(_token(_claims(sub="x" * 9000))) == "end_user_token_malformed"


def test_alg_none_is_refused() -> None:
    unsigned = f"{_b64({'alg': 'none', 'typ': TYP, 'kid': KID})}.{_b64(_claims())}."
    assert _refusal(END_USER_TOKEN_PREFIX + unsigned) == "end_user_token_alg"


def test_an_hmac_token_keyed_with_the_public_key_is_refused() -> None:
    """The classic confusion: the verifier's public key used as an HMAC secret."""
    forged = f"{_b64({'alg': 'HS256', 'typ': TYP, 'kid': KID})}.{_b64(_claims())}"
    import hashlib
    import hmac

    signature = hmac.new(PUBLIC_PEM.encode(), forged.encode(), hashlib.sha256).digest()
    assert _refusal(f"{END_USER_TOKEN_PREFIX}{forged}.{_b64(signature)}") == "end_user_token_alg"


@pytest.mark.parametrize(
    ("algorithm", "key"),
    [
        ("RS256", generate_rsa_key(public_exponent=65537, key_size=2048)),
        ("PS256", generate_rsa_key(public_exponent=65537, key_size=2048)),
        ("ES256", generate_private_key(SECP256R1())),
        ("HS256", "a-shared-secret-of-at-least-thirty-two-bytes"),
    ],
)
def test_every_algorithm_but_eddsa_is_refused(algorithm: str, key: Any) -> None:
    assert _refusal(_token(key=key, algorithm=algorithm)) == "end_user_token_alg"


@pytest.mark.parametrize("typ", ["JWT", "at+jwt", None])
def test_a_token_of_another_type_is_refused(typ: str | None) -> None:
    header = {"alg": "EdDSA", "kid": KID} if typ is None else {"alg": "EdDSA", "typ": typ, "kid": KID}
    signing_input = f"{_b64(header)}.{_b64(_claims())}"
    signature = SIGNER.sign(signing_input.encode())
    assert _refusal(f"{END_USER_TOKEN_PREFIX}{signing_input}.{_b64(signature)}") == "end_user_token_typ"


@pytest.mark.parametrize(
    "extra",
    [{"jku": "https://attacker.example/jwks"}, {"crit": ["exp"]}, {"x5u": "https://attacker.example/cert"}],
)
def test_a_header_with_anything_beyond_alg_typ_and_kid_is_refused(extra: dict[str, Any]) -> None:
    # PyJWT refuses some of these itself, as malformed, before the header is read.
    assert _refusal(_token(**extra)) in {"end_user_token_header", "end_user_token_malformed"}


@pytest.mark.parametrize("kid", [None, "", "k" * 65, 7])
def test_a_token_without_a_usable_kid_is_refused(kid: Any) -> None:
    header: dict[str, Any] = {"alg": "EdDSA", "typ": TYP}
    if kid is not None:
        header["kid"] = kid
    signing_input = f"{_b64(header)}.{_b64(_claims())}"
    signature = SIGNER.sign(signing_input.encode())
    assert _refusal(f"{END_USER_TOKEN_PREFIX}{signing_input}.{_b64(signature)}") in {
        "end_user_token_header",
        "end_user_token_malformed",
    }


# =============================================================================
# The signature
# =============================================================================


def test_a_token_signed_by_another_key_is_refused() -> None:
    assert _refusal(_token(key=Ed25519PrivateKey.generate())) == "end_user_token_bad_signature"


def test_a_tampered_payload_is_refused() -> None:
    header, _, signature = _token()[len(END_USER_TOKEN_PREFIX) :].split(".")
    tampered = f"{header}.{_b64(_claims(sub='someone-else'))}.{signature}"
    assert _refusal(END_USER_TOKEN_PREFIX + tampered) == "end_user_token_bad_signature"


# =============================================================================
# The claims
# =============================================================================


def test_a_token_for_another_service_key_is_refused() -> None:
    assert _refusal(_token(), parent="another-key") == "end_user_token_issuer"


@pytest.mark.parametrize("issuer", ["https://accounts.google.com", PARENT, f"otari:api-key:{PARENT}/x"])
def test_an_issuer_of_any_other_form_is_refused(issuer: str) -> None:
    assert _refusal(_token(_claims(iss=issuer))) == "end_user_token_issuer"


def test_an_expired_token_is_refused() -> None:
    now = int(time.time())
    assert _refusal(_token(_claims(iat=now - 600, exp=now))) == "end_user_token_expired"


def test_a_token_issued_in_the_future_is_refused_past_the_skew() -> None:
    now = int(time.time())
    assert _verify(_token(_claims(iat=now + CLOCK_SKEW_SEC - 5, exp=now + 600))).iat == now + CLOCK_SKEW_SEC - 5
    assert _refusal(_token(_claims(iat=now + CLOCK_SKEW_SEC + 30, exp=now + 900))) == "end_user_token_future_iat"


def test_a_token_that_expires_before_it_is_issued_is_refused() -> None:
    now = int(time.time())
    assert _refusal(_token(_claims(iat=now + 30, exp=now + 30))) == "end_user_token_lifetime"


@pytest.mark.parametrize(
    "overrides",
    [
        {"jti": None},
        {"sub": None},
        {"exp": None},
        {"iat": None},
        {"iss": None},
        {"jti": ""},
        {"jti": "j" * 129},
        {"jti": "has space"},
        {"sub": ""},
        {"exp": float(time.time() + 600)},
        {"exp": str(int(time.time()) + 600)},
        {"iat": True},
        {"aud": "otari"},
        {"nbf": 0},
        {"key_name": "run"},
        {"tags": {f"k{index}": "v" for index in range(17)}},
        {"tags": {"k" * 65: "v"}},
        {"tags": {"k": "v" * 513}},
        {"tags": {"k": 1}},
        {"allowed_models": "openai:*"},
        {"budget_id": ""},
    ],
)
def test_claims_outside_the_mint_fields_bounds_are_refused(overrides: dict[str, Any]) -> None:
    claims = _claims()
    for name, value in overrides.items():
        if value is None:
            claims.pop(name)
        else:
            claims[name] = value
    assert _refusal(_token(claims)) == "end_user_token_claims"


def test_the_digest_is_the_same_for_the_same_claims_and_differs_otherwise() -> None:
    claims = _claims(tags={"b": "2", "a": "1"})
    first = claims_digest(_verify(_token(claims)))
    assert claims_digest(_verify(_token({**claims, "tags": {"a": "1", "b": "2"}}))) == first
    assert claims_digest(_verify(_token({**claims, "exp": claims["exp"] + 1}))) != first
    assert claims_digest(_verify(_token({**claims, "budget_id": "run-one"}))) != first


# =============================================================================
# Registration
# =============================================================================


def test_an_ed25519_public_key_is_stored_normalized() -> None:
    assert normalize_public_key(PUBLIC_PEM.replace("\n", "\r\n")) == PUBLIC_PEM


@pytest.mark.parametrize(
    "pem",
    [
        SIGNER.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()).decode(),
        generate_rsa_key(public_exponent=65537, key_size=2048)
        .public_key()
        .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
        .decode(),
        "not a key",
        PUBLIC_PEM + "x" * 1024,
    ],
)
def test_anything_but_an_ed25519_public_key_is_refused(pem: str) -> None:
    with pytest.raises(SigningKeyInvalidError):
        normalize_public_key(pem)


# =============================================================================
# Coexistence with other bearer tokens
# =============================================================================


def test_an_end_user_token_is_never_shaped_like_a_jwt() -> None:
    """Workload identity reads a credential with exactly two dots as an OpenID Connect token.

    The prefix gives an end-user token a third, so that path never reads one, and no JWT parser does either.
    """
    credential = _token()
    assert credential.count(".") == 3
    with pytest.raises(jwt.DecodeError):
        jwt.get_unverified_header(credential)
    assert not issuer_for(PARENT).startswith("https://")


def test_a_bare_jwt_is_never_an_end_user_token() -> None:
    """An OpenID Connect token, however its header and claims look, lacks the prefix."""
    federated = jwt.encode(
        _claims(iss="https://accounts.google.com"), SIGNER, algorithm="EdDSA", headers={"kid": KID, "typ": TYP}
    )
    assert federated.count(".") == 2
    assert not is_end_user_token(federated)
    assert not is_end_user_token(f"Bearer {_token()}")
    assert not is_end_user_token("tk-" + "a" * 60)
    with pytest.raises(EndUserTokenRefusedError):
        read_token_header(federated)
