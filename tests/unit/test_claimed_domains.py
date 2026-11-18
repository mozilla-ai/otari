"""Which of an organization's claimed email domains are still proven."""

import uuid
from datetime import UTC, datetime, timedelta

from gateway.models.tenancy import DOMAIN_PROOF_TTL, OrganizationDomain
from gateway.services.organizations._claimed_domains import ClaimedDomain, proven_domains

_NOW = datetime(2026, 6, 1, 12, tzinfo=UTC)


def _claim(domain: str, *, verified_at: datetime | None) -> ClaimedDomain:
    return ClaimedDomain(domain=domain, verified_at=verified_at, enabled=True)


def test_proven_domains_keeps_a_proof_exactly_the_ttl_old() -> None:
    claim = _claim("acme.test", verified_at=_NOW - DOMAIN_PROOF_TTL)

    assert proven_domains([claim], now=_NOW) == ("acme.test",)


def test_proven_domains_drops_an_expired_proof() -> None:
    claim = _claim("acme.test", verified_at=_NOW - DOMAIN_PROOF_TTL - timedelta(microseconds=1))

    assert proven_domains([claim], now=_NOW) == ()


def test_proven_domains_drops_a_claim_never_verified() -> None:
    claim = _claim("acme.test", verified_at=None)

    assert proven_domains([claim], now=_NOW) == ()


def test_proven_domains_agrees_with_proof_expired_at_the_boundary() -> None:
    verified_at = _NOW - DOMAIN_PROOF_TTL
    claim = _claim("acme.test", verified_at=verified_at)
    row = OrganizationDomain(
        organization_id=uuid.uuid4(), domain="acme.test", verification_token="token", verified_at=verified_at
    )

    assert proven_domains([claim], now=_NOW) == ("acme.test",)
    assert row.proof_expired(now=_NOW) is False
