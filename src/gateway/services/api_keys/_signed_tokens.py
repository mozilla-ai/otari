"""Signed end-user tokens: what a service key's control plane issues offline instead of calling the mint route.

A token is ``otari-eut1.`` followed by a compact JWS, signed with EdDSA (Ed25519)
by a private key whose public half the service key registered. Its claims are
the mint request's fields. The prefix decides, before anything is parsed, that a
credential is one of these; a compact JWS has exactly two dots and the prefix
adds a third, so no JWT parser, and no other credential path, ever reads one as
a JWT. The header's ``typ`` and the non-URL ``iss`` are checked as well.

Nothing here touches the database: the caller finds the signing key by the
header's kid and hands its public key in.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import jwt
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, load_pem_public_key
from pydantic import ValidationError

from gateway.exceptions.api_keys_exceptions import EndUserTokenRefusedError, SigningKeyInvalidError
from gateway.schemas.budgets import EndUserTokenClaims

END_USER_TOKEN_PREFIX = "otari-eut1."
END_USER_TOKEN_TYPE = "otari-eut+jwt"
END_USER_TOKEN_ALGORITHM = "EdDSA"
# ``iss`` names the service key by id under this prefix: not a URL, so it can never be an OpenID issuer.
ISSUER_PREFIX = "otari:api-key:"
# How far ahead of this clock a token's ``iat`` may be.
CLOCK_SKEW_SEC = 60
MAX_TOKEN_LENGTH = 8192
MAX_KID_LENGTH = 64
MAX_PUBLIC_KEY_PEM_LENGTH = 1024
_HEADER_FIELDS = frozenset({"alg", "typ", "kid"})
# Knows EdDSA and nothing else, so no other algorithm can verify even if the checks before it were skipped.
_JWS = jwt.PyJWS(algorithms=[END_USER_TOKEN_ALGORITHM])


@dataclass(frozen=True)
class EndUserTokenHeader:
    """The unverified header of an end-user token: the kid to verify it with, and the JWS itself."""

    kid: str
    jws: str


def is_end_user_token(credential: str) -> bool:
    """Whether a presented credential is an end-user token rather than an API key or another kind of token."""
    return credential.startswith(END_USER_TOKEN_PREFIX)


def issuer_for(api_key_id: str) -> str:
    """The ``iss`` a token signed for this service key carries."""
    return f"{ISSUER_PREFIX}{api_key_id}"


def read_token_header(credential: str) -> EndUserTokenHeader:
    """Read the kid off an end-user token, refusing one whose header is anything but ``{alg, typ, kid}``.

    Unverified: this only says which public key checks the signature. ``alg`` must be EdDSA, so ``none``, HMAC and
    RSA tokens are refused before a key is looked up.
    """
    if not is_end_user_token(credential) or len(credential) > MAX_TOKEN_LENGTH:
        raise EndUserTokenRefusedError("end_user_token_malformed")
    jws = credential[len(END_USER_TOKEN_PREFIX) :]
    try:
        header = _JWS.get_unverified_header(jws)
    except jwt.InvalidTokenError as exc:
        raise EndUserTokenRefusedError("end_user_token_malformed") from exc
    if header.get("alg") != END_USER_TOKEN_ALGORITHM:
        raise EndUserTokenRefusedError("end_user_token_alg")
    if header.get("typ") != END_USER_TOKEN_TYPE:
        raise EndUserTokenRefusedError("end_user_token_typ")
    kid = header.get("kid")
    if set(header) != _HEADER_FIELDS or not isinstance(kid, str) or not 0 < len(kid) <= MAX_KID_LENGTH:
        raise EndUserTokenRefusedError("end_user_token_header")
    return EndUserTokenHeader(kid=kid, jws=jws)


def verify_token_claims(
    header: EndUserTokenHeader, *, public_key_pem: str, parent_key_id: str, now: float
) -> EndUserTokenClaims:
    """Verify an end-user token's signature and return its claims, refusing any the service key could not mint.

    The claims must name exactly this service key as ``iss``, carry nothing beyond the mint fields, be unexpired at
    ``now`` and issued no later than ``CLOCK_SKEW_SEC`` after it. The bounds that depend on the service key's
    settings are the caller's.
    """
    public_key = load_pem_public_key(public_key_pem.encode())
    if not isinstance(public_key, Ed25519PublicKey):
        raise EndUserTokenRefusedError("end_user_token_bad_signature")
    try:
        verified = _JWS.decode_complete(header.jws, key=public_key, algorithms=[END_USER_TOKEN_ALGORITHM])
    except jwt.InvalidSignatureError as exc:
        raise EndUserTokenRefusedError("end_user_token_bad_signature") from exc
    except jwt.InvalidTokenError as exc:
        raise EndUserTokenRefusedError("end_user_token_malformed") from exc
    if verified["header"] != {"alg": END_USER_TOKEN_ALGORITHM, "typ": END_USER_TOKEN_TYPE, "kid": header.kid}:
        raise EndUserTokenRefusedError("end_user_token_header")
    try:
        payload = json.loads(verified["payload"])
        claims = EndUserTokenClaims.model_validate(payload)
    except (ValueError, ValidationError) as exc:
        raise EndUserTokenRefusedError("end_user_token_claims") from exc
    if claims.iss != issuer_for(parent_key_id):
        raise EndUserTokenRefusedError("end_user_token_issuer")
    if now >= claims.exp:
        raise EndUserTokenRefusedError("end_user_token_expired")
    if claims.iat > now + CLOCK_SKEW_SEC:
        raise EndUserTokenRefusedError("end_user_token_future_iat")
    if claims.exp <= claims.iat:
        raise EndUserTokenRefusedError("end_user_token_lifetime")
    return claims


def claims_digest(claims: EndUserTokenClaims) -> str:
    """A SHA-256 over the claims, the same for any two tokens that say the same thing."""
    canonical: dict[str, Any] = claims.model_dump(mode="json")
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def normalize_public_key(pem: str) -> str:
    """Return an Ed25519 public key's PEM as Otari stores it, refusing anything else, a private key included."""
    if len(pem) > MAX_PUBLIC_KEY_PEM_LENGTH:
        raise SigningKeyInvalidError()
    try:
        public_key = load_pem_public_key(pem.encode())
    except (ValueError, TypeError, UnsupportedAlgorithm) as exc:
        raise SigningKeyInvalidError() from exc
    if not isinstance(public_key, Ed25519PublicKey):
        raise SigningKeyInvalidError()
    return public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode()
