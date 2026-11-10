"""Workload identity: verify an OpenID Connect identity token and find the API key it stands for.

A backend on a cloud platform (Cloud Run, GKE, GCE) can fetch a token its
platform signed, naming the workload and an audience the caller chose. When
the token's issuer is one the deployment lists in ``federation_issuers`` and
the token matches a federation rule on a key, the request authenticates as
that key, so the backend holds no Otari secret. A member's own rule may name
an issuer the deployment does not list; such a rule is checked only in the
token exchange that names it.

Verification is apron-auth's (``apron_auth.workload``): the signature against
the issuer's published keys, public-key algorithms only, and the claims. What
is Otari's is which issuers are trusted, where their keys may be fetched from,
which audiences this deployment answers to, and the rules. Every refusal
carries a reason for the authentication-failure metric and nothing from the
token itself.
"""

import base64
import binascii
import json
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
from apron_auth.errors import (
    ConfigurationError,
    IdentityFetchError,
    TokenExpiredError,
    TokenValidationError,
    TokenValidationReason,
)
from apron_auth.models import WorkloadIssuer
from apron_auth.workload import WorkloadTokenVerifier

from gateway.core.settings.api_keys import FederationIssuer
from gateway.models.api_keys import APIKeyFederationRule
from gateway.services.web_retrieval_network import CheckedFederationTransport

MAX_TOKEN_BYTES = 16 * 1024
# A token with longer than this left before its ``exp`` is refused: platform identity tokens last an hour, so a
# far-off expiry means an issuer minting what is in effect a static secret.
MAX_TOKEN_LIFETIME_SECONDS = 24 * 3600
# How many issuers named only by members' rules keep a verifier, and so their cached signing keys, at once. Members
# choose these, so the cache is bounded; past it the least recently used is dropped and refetches on its next use.
MAX_MEMBER_ISSUERS = 256


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


TransportFactory = Callable[[str], httpx.AsyncBaseTransport]


class FederatedTokenVerifier:
    """Verify identity tokens from the allow-listed issuers, one apron-auth verifier per issuer.

    One per app, because each verifier caches its issuer's signing keys. With ``member_issuers``, a token may also
    come from the issuer a member's rule names, verified by a verifier built on its first use; its keys are always
    fetched under the outbound checks with private hosts refused, whatever ``allow_private_hosts`` says, since a
    member chose the URL. ``transport_factory`` replaces the outbound checks, for tests only.

    ``audiences`` is the set this deployment answers to and it is this verifier's own policy, so no caller passes
    one in. An empty set is refused at construction, so a deployment cannot admit a token minted for another
    service at the same issuer by leaving it out.
    """

    def __init__(
        self,
        issuers: Sequence[FederationIssuer],
        *,
        audiences: Sequence[str],
        allow_private_hosts: bool = False,
        member_issuers: bool = False,
        transport_factory: TransportFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not audiences:
            msg = "A deployment verifying workload identity tokens must answer to at least one audience"
            raise ValueError(msg)
        self.issuers = tuple(issuers)
        self.audiences = tuple(audiences)
        self.allow_private_hosts = allow_private_hosts
        self.member_issuers = member_issuers
        self._clock = clock
        self._member_factory = transport_factory or (lambda _url: CheckedFederationTransport(allow_private_hosts=False))
        factory = transport_factory or (
            lambda _url: CheckedFederationTransport(allow_private_hosts=allow_private_hosts)
        )
        self._by_issuer: dict[str, WorkloadTokenVerifier | None] = {}
        for entry in self.issuers:
            try:
                verifier = WorkloadTokenVerifier(
                    [WorkloadIssuer(issuer=entry.issuer, jwks_url=entry.jwks_uri)],
                    transport_factory=factory,
                    clock=clock,
                )
            except ConfigurationError:
                # Settings validation admits only https issuers, so only an unfetchable URL lands here.
                verifier = None
            self._by_issuer[entry.issuer] = verifier
        self._by_member_issuer: OrderedDict[str, WorkloadTokenVerifier | None] = OrderedDict()

    def _verifier_for_member_issuer(self, issuer: str) -> WorkloadTokenVerifier | None:
        if issuer in self._by_member_issuer:
            self._by_member_issuer.move_to_end(issuer)
            return self._by_member_issuer[issuer]
        try:
            verifier: WorkloadTokenVerifier | None = WorkloadTokenVerifier(
                [WorkloadIssuer(issuer=issuer)], transport_factory=self._member_factory, clock=self._clock
            )
        except ConfigurationError:
            # A member's rule admits only https issuers, so only an unfetchable URL lands here.
            verifier = None
        self._by_member_issuer[issuer] = verifier
        if len(self._by_member_issuer) > MAX_MEMBER_ISSUERS:
            self._by_member_issuer.popitem(last=False)
        return verifier

    async def verify(self, token: str, *, member_issuer: str | None = None) -> FederatedIdentity:
        """Verify ``token`` and return what it says about its bearer.

        The token must name one of the audiences this deployment answers to, which is what proves it was minted
        for this deployment; a rule's own audience still has to match on top of that. ``member_issuer`` is the
        issuer of the member's rule the token is being checked against, which the token must carry; it is trusted
        for this token alone, and only when the verifier admits member issuers. An allow-listed issuer is verified
        as configured either way.

        Raises:
            FederationRefusedError: the token does not verify.
            FederationUnavailableError: the issuer's keys could not be fetched.
        """
        issuer = unverified_issuer(token)
        if issuer is None:
            raise FederationRefusedError("federation_invalid_token")
        if issuer in self._by_issuer:
            verifier = self._by_issuer[issuer]
        elif self.member_issuers and member_issuer is not None and issuer == member_issuer:
            verifier = self._verifier_for_member_issuer(issuer)
        else:
            raise FederationRefusedError("federation_unknown_issuer")
        if verifier is None:
            raise FederationUnavailableError
        try:
            identity = await verifier.verify(token, audience=self.audiences)
        except TokenExpiredError as exc:
            raise FederationRefusedError("federation_expired") from exc
        except TokenValidationError as exc:
            if exc.reason is TokenValidationReason.AUDIENCE:
                raise FederationRefusedError("federation_wrong_audience") from exc
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
