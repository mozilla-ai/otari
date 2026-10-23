"""Short-lived access tokens a workload gets by exchanging its identity token (``POST /oauth/token``).

The RFC 7523 ``jwt-bearer`` grant, as the Anthropic SDKs and Claude Code speak
it: a client presents a verified identity token naming one federation rule and
receives a bearer token it then sends instead of a key. The token is stateless:
it names the rule and its key and when it expires, signed with HMAC-SHA256
under a key derived from ``OTARI_SECRET_KEY``, so every replica sharing that
secret verifies it and nothing is written to mint one. Revocation needs no
token list either, because each request reloads the rule and the key: deleting
the rule or deactivating the key cuts every token off at once.

The fixed prefix lets the verify path tell an access token from a key secret or
a raw identity token before parsing anything.
"""

import base64
import binascii
import hmac
import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import hmac as crypto_hmac
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from gateway.services.api_keys._federation import FederationRefusedError

ACCESS_TOKEN_PREFIX = "otari-fat1."
JWT_BEARER_GRANT = "urn:ietf:params:oauth:grant-type:jwt-bearer"
_KDF_INFO = b"otari federation access token v1"
_MAX_TOKEN_BYTES = 1024


@dataclass(frozen=True)
class AccessTokenClaims:
    """What a verified access token names."""

    rule_id: str
    api_key_id: str
    expires_at: int


class FederationGrantError(Exception):
    """A token request refused, rendered as an RFC 6749 section 5.2 error body.

    ``reason`` is the auth-failure metric label; the description is generic, so a refusal never says which check
    failed.
    """

    def __init__(self, error: str, description: str, *, status_code: int, reason: str) -> None:
        super().__init__(reason)
        self.error = error
        self.description = description
        self.status_code = status_code
        self.reason = reason


def _signing_key(secret: str) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=_KDF_INFO).derive(secret.encode())


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(segment: str) -> bytes:
    """Decode unpadded base64url strictly, so one token has exactly one spelling."""
    return base64.b64decode(segment + "=" * (-len(segment) % 4), altchars=b"-_", validate=True)


def _signature(secret: str, payload_segment: str) -> bytes:
    """HMAC-SHA256 of the signed part, under the key HKDF derives from ``secret``; both from ``cryptography``."""
    mac = crypto_hmac.HMAC(_signing_key(secret), hashes.SHA256())
    mac.update(f"{ACCESS_TOKEN_PREFIX}{payload_segment}".encode())
    return mac.finalize()


def mint_access_token(claims: AccessTokenClaims, secrets: Sequence[str]) -> str:
    """Sign ``claims`` with the first of ``secrets``.

    Raises:
        ValueError: no secret is configured.
    """
    if not secrets:
        msg = "OTARI_SECRET_KEY is not set"
        raise ValueError(msg)
    payload = json.dumps(
        {"r": claims.rule_id, "k": claims.api_key_id, "exp": claims.expires_at}, separators=(",", ":")
    ).encode()
    segment = _b64(payload)
    return f"{ACCESS_TOKEN_PREFIX}{segment}.{_b64(_signature(secrets[0], segment))}"


def is_access_token(token: str) -> bool:
    """Whether ``token`` claims to be an access token, by its prefix alone."""
    return token.startswith(ACCESS_TOKEN_PREFIX)


def read_access_token(token: str, secrets: Sequence[str], *, now: float) -> AccessTokenClaims:
    """Verify an access token's signature, under any of ``secrets``, and its expiry.

    Raises:
        FederationRefusedError: the token is malformed, signed with no configured secret, or expired.
    """
    if len(token) > _MAX_TOKEN_BYTES or not is_access_token(token):
        raise FederationRefusedError("federation_access_token_invalid")
    payload_segment, _, signature_segment = token[len(ACCESS_TOKEN_PREFIX) :].partition(".")
    try:
        signature = _unb64(signature_segment)
    except (ValueError, binascii.Error) as exc:
        raise FederationRefusedError("federation_access_token_invalid") from exc
    if _b64(signature) != signature_segment or not any(
        hmac.compare_digest(_signature(secret, payload_segment), signature) for secret in secrets
    ):
        raise FederationRefusedError("federation_access_token_invalid")
    try:
        payload = json.loads(_unb64(payload_segment))
        claims = AccessTokenClaims(rule_id=payload["r"], api_key_id=payload["k"], expires_at=payload["exp"])
    except (ValueError, binascii.Error, KeyError, TypeError) as exc:
        raise FederationRefusedError("federation_access_token_invalid") from exc
    if not (
        isinstance(claims.rule_id, str) and isinstance(claims.api_key_id, str) and isinstance(claims.expires_at, int)
    ):
        raise FederationRefusedError("federation_access_token_invalid")
    if claims.expires_at <= now:
        raise FederationRefusedError("federation_access_token_expired")
    return claims


def _same_uuid(value: str, expected: uuid.UUID) -> bool:
    try:
        return uuid.UUID(value) == expected
    except ValueError:
        return False


def grant_names_key(
    *,
    organization_id: str | None,
    workspace_id: str | None,
    service_account_id: str | None,
    key_organization_id: uuid.UUID,
    key_workspace_id: uuid.UUID,
    api_key_id: str,
) -> bool:
    """Whether each scope a token request names, where it names one, is the rule's key's own.

    ``organization_id`` and ``workspace_id`` are the UUIDs of the key's organization and workspace;
    ``service_account_id`` is the key's id.
    """
    return (
        (organization_id is None or _same_uuid(organization_id, key_organization_id))
        and (workspace_id is None or _same_uuid(workspace_id, key_workspace_id))
        and (service_account_id is None or service_account_id == api_key_id)
    )


def access_token_expiry(*, now: float, ttl_seconds: int, identity_expires_at: int) -> int:
    """When an access token minted now expires: after ``ttl_seconds``, and never after the identity token."""
    return min(int(now) + ttl_seconds, identity_expires_at)
