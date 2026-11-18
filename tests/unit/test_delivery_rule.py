"""Whether Otari delivers an invitation email to an address."""

import uuid
from collections.abc import Collection
from datetime import UTC, datetime, timedelta

import pytest

from gateway.models.tenancy import DOMAIN_PROOF_TTL, OrganizationDomain
from gateway.services.organizations import ClaimedDomain
from gateway.services.organizations._delivery_rule import DeliveryRule

pytestmark = pytest.mark.asyncio

_NOW = datetime(2026, 6, 1, 12, tzinfo=UTC)
_INVITING = uuid.uuid4()
_OTHER = uuid.uuid4()


class _Claims:
    def __init__(self, *claims: OrganizationDomain) -> None:
        self._claims = claims

    async def for_organization(self, organization_id: uuid.UUID) -> list[ClaimedDomain]:
        raise NotImplementedError

    async def get_verified_by_domain(self, domain: str) -> OrganizationDomain | None:
        return next((claim for claim in self._claims if claim.domain == domain and claim.verified_at), None)


class _Blocklist:
    def __init__(self, *entries: str) -> None:
        self._entries = set(entries)

    async def blocked_among(self, emails: Collection[str]) -> set[str]:
        return {email for email in emails if email in self._entries}


def _claim(organization_id: uuid.UUID, *, verified_at: datetime | None) -> OrganizationDomain:
    return OrganizationDomain(
        organization_id=organization_id, domain="acme.com", verification_token="token", verified_at=verified_at
    )


def _rule(*claims: OrganizationDomain, blocked: tuple[str, ...] = ()) -> DeliveryRule:
    return DeliveryRule(_Claims(*claims), _Blocklist(*blocked))


async def test_an_address_at_a_domain_another_organization_has_proven_is_not_delivered_to() -> None:
    rule = _rule(_claim(_OTHER, verified_at=_NOW - timedelta(days=1)))

    assert not await rule.allows("ada@acme.com", _INVITING, now=_NOW)


async def test_an_address_at_a_domain_the_inviting_organization_has_proven_is_delivered_to() -> None:
    rule = _rule(_claim(_INVITING, verified_at=_NOW - timedelta(days=1)))

    assert await rule.allows("ada@acme.com", _INVITING, now=_NOW)


async def test_a_proof_older_than_its_lifetime_does_not_stop_delivery() -> None:
    rule = _rule(_claim(_OTHER, verified_at=_NOW - DOMAIN_PROOF_TTL - timedelta(seconds=1)))

    assert await rule.allows("ada@acme.com", _INVITING, now=_NOW)


async def test_a_proof_at_the_end_of_its_lifetime_still_stops_delivery() -> None:
    rule = _rule(_claim(_OTHER, verified_at=_NOW - DOMAIN_PROOF_TTL))

    assert not await rule.allows("ada@acme.com", _INVITING, now=_NOW)


async def test_a_claim_that_was_never_proven_does_not_stop_delivery() -> None:
    rule = _rule(_claim(_OTHER, verified_at=None))

    assert await rule.allows("ada@acme.com", _INVITING, now=_NOW)


async def test_the_domain_is_matched_in_the_spelling_a_claim_is_stored_in() -> None:
    rule = _rule(_claim(_OTHER, verified_at=_NOW - timedelta(days=1)))

    assert not await rule.allows("Ada@ACME.com.", _INVITING, now=_NOW)


async def test_a_blocked_address_is_not_delivered_to() -> None:
    rule = _rule(blocked=("ada@acme.com",))

    assert not await rule.allows("ada@acme.com", _INVITING, now=_NOW)
    assert await rule.allows("bob@acme.com", _INVITING, now=_NOW)


async def test_an_address_at_a_public_provider_is_delivered_to() -> None:
    rule = _rule(_claim(_OTHER, verified_at=_NOW - timedelta(days=1)))

    assert await rule.allows("ada@gmail.com", _INVITING, now=_NOW)


async def test_text_that_is_not_an_address_is_not_delivered_to() -> None:
    assert not await _rule().allows("ada", _INVITING, now=_NOW)


@pytest.mark.parametrize(
    "email",
    [
        "grace@acme.com",
        "grace@münchen.de",
        "grace@xn--mnchen-3ya.de",
        "ivan@пример.рф",
        "ivan@xn--e1afmkfd.xn--p1ai",
    ],
)
async def test_an_address_that_nothing_proves_or_blocks_is_delivered_to(email: str) -> None:
    assert await _rule().allows(email, _INVITING, now=_NOW)


async def test_text_with_a_domain_that_is_not_an_address_is_not_delivered_to() -> None:
    assert not await _rule().allows("ada@bob@acme.com", _INVITING, now=_NOW)
