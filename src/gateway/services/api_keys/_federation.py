"""Workload identity: verify an OpenID Connect identity token and find the API key it stands for.

A backend on a cloud platform (Cloud Run, GKE, GCE) can fetch a token its
platform signed, naming the workload and an audience the caller chose. When
the token's issuer is one the deployment lists in ``federation_issuers`` and
the token matches a federation rule on a key, the request authenticates as
that key, so the backend holds no Otari secret.

Verification is apron-auth's (``apron_auth.workload``): the signature against
the issuer's published keys, public-key algorithms only, and the claims. What
is Otari's is which issuers are trusted, where their keys may be fetched from,
and the rules. Every refusal carries a reason for the authentication-failure
metric and nothing from the token itself.
"""

import base64
import binascii
import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
from apron_auth.errors import ConfigurationError, IdentityFetchError, TokenExpiredError, TokenVerificationError
from apron_auth.workload import WorkloadTokenVerifier

from gateway.core.settings.api_keys import FederationIssuer
from gateway.models.api_keys import APIKeyFederationRule
from gateway.services.url_safety import UnsafeURLError, validate_federation_url

MAX_TOKEN_BYTES = 16 * 1024
# A token with longer than this left before its ``exp`` is refused: platform identity tokens last an hour, so a
# far-off expiry means an issuer minting what is in effect a static secret.
MAX_TOKEN_LIFETIME_SECONDS = 24 * 3600


class FederationRefusedError(Exception):
    """A token that does not authenticate. ``reason`` is the metric label; nothing from the token is kept."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class FederationUnavailableError(Exception):
    """The issuer's signing keys could not be fetched, so no token from it can be verified now."""


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


class _CheckedTransport(httpx.AsyncBaseTransport):
    """Send a discovery or key set request only to a URL that passes the outbound checks."""

    def __init__(self, *, allow_private_hosts: bool) -> None:
        self._allow_private_hosts = allow_private_hosts
        self._transport = httpx.AsyncHTTPTransport(trust_env=False)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            await validate_federation_url(str(request.url), allow_private_hosts=self._allow_private_hosts)
        except UnsafeURLError as exc:
            # A request error is what apron-auth reports as a failed fetch.
            raise httpx.ConnectError("refused by the outbound URL checks", request=request) from exc
        return await self._transport.handle_async_request(request)

    async def aclose(self) -> None:
        await self._transport.aclose()


TransportFactory = Callable[[str], httpx.AsyncBaseTransport]


class FederatedTokenVerifier:
    """Verify identity tokens from the allow-listed issuers, one apron-auth verifier per issuer.

    One per app, because each verifier caches its issuer's signing keys.
    ``transport_factory`` replaces the outbound checks, for tests only.
    """

    def __init__(
        self,
        issuers: Sequence[FederationIssuer],
        *,
        allow_private_hosts: bool = False,
        transport_factory: TransportFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.issuers = tuple(issuers)
        self.allow_private_hosts = allow_private_hosts
        factory = transport_factory or (lambda _url: _CheckedTransport(allow_private_hosts=allow_private_hosts))
        self._by_issuer: dict[str, WorkloadTokenVerifier | None] = {}
        for entry in self.issuers:
            try:
                verifier = WorkloadTokenVerifier(
                    entry.issuer, jwks_url=entry.jwks_uri, transport_factory=factory, clock=clock
                )
            except ConfigurationError:
                # Settings validation admits only https issuers, so only an unfetchable URL lands here.
                verifier = None
            self._by_issuer[entry.issuer] = verifier

    async def verify(self, token: str) -> FederatedIdentity:
        """Verify ``token`` and return what it says about its bearer.

        The audience is left to the rules, which are what name the accepted ones.

        Raises:
            FederationRefusedError: the token does not verify.
            FederationUnavailableError: the issuer's keys could not be fetched.
        """
        issuer = unverified_issuer(token)
        if issuer is None:
            raise FederationRefusedError("federation_invalid_token")
        if issuer not in self._by_issuer:
            raise FederationRefusedError("federation_unknown_issuer")
        verifier = self._by_issuer[issuer]
        if verifier is None:
            raise FederationUnavailableError
        try:
            identity = await verifier.verify(token, audience=None)
        except TokenExpiredError as exc:
            raise FederationRefusedError("federation_expired") from exc
        except TokenVerificationError as exc:
            raise FederationRefusedError("federation_invalid_token") from exc
        except IdentityFetchError as exc:
            raise FederationUnavailableError from exc
        if identity.expires_at - time.time() > MAX_TOKEN_LIFETIME_SECONDS:
            raise FederationRefusedError("federation_lifetime_too_long")
        email = identity.email.strip().lower() if identity.email and identity.email.strip() else None
        return FederatedIdentity(
            issuer=identity.issuer,
            audiences=identity.audiences,
            subject=identity.subject,
            email=email,
            expires_at=identity.expires_at,
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
