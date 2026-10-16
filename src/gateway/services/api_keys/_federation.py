"""Workload identity: verify an OpenID Connect identity token and find the API key it stands for.

A backend on a cloud platform (Cloud Run, GKE, GCE) can fetch a token its
platform signed, naming the workload and an audience the caller chose. When
the token's issuer is one the deployment lists in ``federation_issuers`` and
the token matches a federation rule on a key, the request authenticates as
that key, so the backend holds no Otari secret.

Signature verification is PyJWT's, against the issuer's published keys, and
only with an asymmetric algorithm: ``none`` and the HMAC family are refused
before any key is looked up, and a key is bound to its own algorithm, so a
token cannot pick a weaker one. Every refusal carries a reason for the
authentication-failure metric and nothing from the token itself.
"""

import asyncio
import base64
import binascii
import json
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
import jwt
from jwt.exceptions import ExpiredSignatureError, PyJWTError

from gateway.core.settings.api_keys import FederationIssuer
from gateway.models.api_keys import APIKeyFederationRule
from gateway.services.url_safety import UnsafeURLError, validate_federation_url

ALLOWED_ALGORITHMS = ("RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512")
CLOCK_SKEW_SECONDS = 60
JWKS_TTL_SECONDS = 3600
# An unknown key id refetches the issuer's keys at most this often, so a token
# naming made-up key ids cannot turn each request into a fetch.
JWKS_REFETCH_INTERVAL_SECONDS = 60
# After a failed fetch, requests are refused for this long without another try.
JWKS_RETRY_INTERVAL_SECONDS = 10
FETCH_TIMEOUT_SECONDS = 5.0
MAX_DOCUMENT_BYTES = 256 * 1024
MAX_TOKEN_BYTES = 16 * 1024

FetchJson = Callable[[str], Awaitable[Any]]


class FederationRefusedError(Exception):
    """A token that does not authenticate. ``reason`` is the metric label; nothing from the token is kept."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class FederationUnavailableError(Exception):
    """The issuer's signing keys could not be fetched, so no token from it can be verified now."""


class _FetchError(Exception):
    pass


@dataclass(frozen=True)
class FederatedIdentity:
    """What a verified token says about its bearer. ``email`` is set only when the token marks it verified.

    ``expires_at`` is the token's ``exp``, in seconds since the epoch.
    """

    issuer: str
    audiences: tuple[str, ...]
    subject: str
    email: str | None
    expires_at: int


@dataclass(frozen=True)
class FederationMatch:
    """The key a verified token authenticates as, and the rules that matched it."""

    api_key_id: str
    rule_ids: tuple[str, ...]


def _b64_json(segment: str) -> Any:
    padded = segment + "=" * (-len(segment) % 4)
    return json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))


def unverified_issuer(token: str) -> str | None:
    """The ``iss`` of ``token`` when it is shaped like a signed JWT, or None.

    Read without verification, only to decide which path checks the
    credential; nothing here is trusted. An API key never has this shape.
    """
    if len(token) > MAX_TOKEN_BYTES or token.count(".") != 2:
        return None
    header_segment, payload_segment, _ = token.split(".")
    try:
        header = _b64_json(header_segment)
        payload = _b64_json(payload_segment)
    except (ValueError, binascii.Error, UnicodeError):
        return None
    if not isinstance(header, dict) or not isinstance(payload, dict) or "alg" not in header:
        return None
    issuer = payload.get("iss")
    return issuer if isinstance(issuer, str) else None


async def fetch_json(url: str) -> Any:
    """GET a JSON document over https, through the outbound URL checks, refusing redirects and large bodies."""
    try:
        await validate_federation_url(url)
        async with (
            httpx.AsyncClient(timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=False) as client,
            client.stream("GET", url, headers={"Accept": "application/json"}) as response,
        ):
            if response.status_code != httpx.codes.OK:
                raise _FetchError
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > MAX_DOCUMENT_BYTES:
                    raise _FetchError
        return json.loads(bytes(body))
    except (httpx.HTTPError, UnsafeURLError, ValueError) as exc:
        raise _FetchError from exc


@dataclass
class _IssuerKeys:
    keys: dict[str, jwt.PyJWK]
    fetched_at: float


def _usable_keys(document: Any) -> dict[str, jwt.PyJWK]:
    """The signing keys of a JWKS document that carry an id and an allowed algorithm."""
    if not isinstance(document, dict) or not isinstance(document.get("keys"), list):
        raise _FetchError
    keys: dict[str, jwt.PyJWK] = {}
    for entry in document["keys"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("kid"), str):
            continue
        if entry.get("use", "sig") != "sig":
            continue
        try:
            key = jwt.PyJWK(entry)
        except (PyJWTError, ValueError, TypeError, KeyError):
            continue
        if key.algorithm_name in ALLOWED_ALGORITHMS:
            keys[entry["kid"]] = key
    if not keys:
        raise _FetchError
    return keys


class FederatedTokenVerifier:
    """Verify identity tokens from the allow-listed issuers, caching each issuer's signing keys.

    One per app. Keys are refetched when older than ``JWKS_TTL_SECONDS`` and,
    at most once a minute, when a token names a key id the cache lacks. A
    failed fetch leaves keys still within their lifetime in use and refuses
    every other token from that issuer until a later fetch succeeds; keys
    past their lifetime are never used.
    """

    def __init__(
        self,
        issuers: Sequence[FederationIssuer],
        *,
        fetch: FetchJson = fetch_json,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.issuers = tuple(issuers)
        self._by_issuer = {entry.issuer: entry for entry in self.issuers}
        self._fetch = fetch
        self._clock = clock
        self._cache: dict[str, _IssuerKeys] = {}
        self._failed_at: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def verify(self, token: str) -> FederatedIdentity:
        """Verify ``token`` and return what it says about its bearer.

        Raises:
            FederationRefusedError: the token does not verify.
            FederationUnavailableError: the issuer's keys could not be fetched.
        """
        try:
            header = jwt.get_unverified_header(token)
        except PyJWTError as exc:
            raise FederationRefusedError("federation_invalid_token") from exc
        algorithm = header.get("alg")
        if algorithm not in ALLOWED_ALGORITHMS:
            raise FederationRefusedError("federation_bad_algorithm")
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid:
            raise FederationRefusedError("federation_invalid_token")
        issuer = self._by_issuer.get(unverified_issuer(token) or "")
        if issuer is None:
            raise FederationRefusedError("federation_unknown_issuer")

        key = await self._signing_key(issuer, kid)
        try:
            claims = jwt.decode(
                token,
                key=key,
                algorithms=list(ALLOWED_ALGORITHMS),
                issuer=issuer.issuer,
                leeway=CLOCK_SKEW_SECONDS,
                options={
                    "require": ["iss", "sub", "aud", "exp", "iat"],
                    # The audience is checked against the rules, which are what name the accepted ones.
                    "verify_aud": False,
                    "enforce_minimum_key_length": True,
                },
            )
        except ExpiredSignatureError as exc:
            raise FederationRefusedError("federation_expired") from exc
        except PyJWTError as exc:
            raise FederationRefusedError("federation_invalid_token") from exc
        return _identity(issuer.issuer, claims)

    async def _signing_key(self, issuer: FederationIssuer, kid: str) -> jwt.PyJWK:
        cached = self._cache.get(issuer.issuer)
        if cached is not None and self._fresh(cached) and kid in cached.keys:
            return cached.keys[kid]
        async with self._locks.setdefault(issuer.issuer, asyncio.Lock()):
            now = self._clock()
            cached = self._cache.get(issuer.issuer)
            if cached is not None and not self._fresh(cached):
                del self._cache[issuer.issuer]
                cached = None
            if cached is not None:
                if kid in cached.keys:
                    return cached.keys[kid]
                if now - cached.fetched_at < JWKS_REFETCH_INTERVAL_SECONDS:
                    raise FederationRefusedError("federation_unknown_key")
            failed_at = self._failed_at.get(issuer.issuer)
            if failed_at is not None and now - failed_at < JWKS_RETRY_INTERVAL_SECONDS:
                raise FederationRefusedError("federation_unknown_key") if cached else FederationUnavailableError
            try:
                keys = await self._fetch_keys(issuer)
            except _FetchError as exc:
                self._failed_at[issuer.issuer] = now
                # Keys still within their lifetime keep serving the tokens they sign.
                if cached is not None:
                    raise FederationRefusedError("federation_unknown_key") from exc
                raise FederationUnavailableError from exc
            self._failed_at.pop(issuer.issuer, None)
            self._cache[issuer.issuer] = _IssuerKeys(keys=keys, fetched_at=now)
        if kid not in keys:
            raise FederationRefusedError("federation_unknown_key")
        return keys[kid]

    def _fresh(self, cached: _IssuerKeys) -> bool:
        return self._clock() - cached.fetched_at < JWKS_TTL_SECONDS

    async def _fetch_keys(self, issuer: FederationIssuer) -> dict[str, jwt.PyJWK]:
        jwks_uri = issuer.jwks_uri
        if jwks_uri is None:
            discovery = await self._fetch(f"{issuer.issuer.rstrip('/')}/.well-known/openid-configuration")
            # OpenID Connect Discovery 1.0, section 4.3: the document must name the issuer it was fetched for.
            if not isinstance(discovery, dict) or discovery.get("issuer") != issuer.issuer:
                raise _FetchError
            jwks_uri = discovery.get("jwks_uri")
            if not isinstance(jwks_uri, str):
                raise _FetchError
        return _usable_keys(await self._fetch(jwks_uri))


def _identity(issuer: str, claims: dict[str, Any]) -> FederatedIdentity:
    audience = claims.get("aud")
    audiences = (audience,) if isinstance(audience, str) else audience
    if not isinstance(audiences, list | tuple) or not all(isinstance(value, str) for value in audiences):
        raise FederationRefusedError("federation_invalid_token")
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        raise FederationRefusedError("federation_invalid_token")
    email = claims.get("email")
    # Only a boolean true counts; a string "true" is not a verification.
    verified = claims.get("email_verified") is True
    return FederatedIdentity(
        issuer=issuer,
        audiences=tuple(audiences),
        subject=subject,
        email=email.strip().lower() if verified and isinstance(email, str) and email.strip() else None,
        expires_at=int(claims["exp"]),
    )


def rule_matches(rule: APIKeyFederationRule, identity: FederatedIdentity) -> bool:
    """Whether ``identity`` satisfies every condition ``rule`` sets."""
    if rule.subject is None and rule.email is None:
        return False
    return (
        rule.issuer == identity.issuer
        and rule.audience in identity.audiences
        and (rule.subject is None or rule.subject == identity.subject)
        and (rule.email is None or (identity.email is not None and rule.email == identity.email))
    )


def match_federation_rules(rules: Iterable[APIKeyFederationRule], identity: FederatedIdentity) -> FederationMatch:
    """The one key the matching rules point at.

    Several rules of one key may match; rules of two keys may not, since the
    token would then name no one key.

    Raises:
        FederationRefusedError: no rule matches, or rules of more than one key do.
    """
    matched = [rule for rule in rules if rule_matches(rule, identity)]
    if not matched:
        raise FederationRefusedError("federation_no_rule")
    keys = {rule.api_key_id for rule in matched}
    if len(keys) != 1:
        raise FederationRefusedError("federation_ambiguous_rule")
    return FederationMatch(api_key_id=keys.pop(), rule_ids=tuple(sorted(rule.id for rule in matched)))
