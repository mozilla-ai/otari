"""Email-domain claims: the management surface, the DNS proof, and auto-join.

Split the way the rest of tenancy is. The API half acts as the one operator
identity a standalone deployment has (owner and superuser), so what a
*non*-manager may not do is exercised at the service layer here rather than
through the routes, matching `test_tenancy_authorization.py`'s reasoning.

Only the resolver is stubbed. It is the single call that leaves the process, and
a test that reached real DNS would pass or fail on somebody else's zone file.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.tenancy import (
    DOMAIN_PROOF_TTL,
    DOMAIN_VERIFICATION_TXT_PREFIX,
    MAX_ORGANIZATION_DOMAINS,
    Organization,
    OrganizationDomain,
    OrganizationDomainCreateRequest,
    OrganizationDomainUpdateRequest,
    User,
)
from gateway.repositories.tenancy import (
    OrganizationDomainRepository,
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
)
from gateway.services.tenancy.errors import (
    NotAuthorizedError,
    OrganizationDomainAlreadyClaimedError,
    OrganizationDomainClaimedHereError,
    OrganizationDomainNotFoundError,
    OrganizationDomainNotVerifiedError,
    PublicEmailDomainError,
    TooManyOrganizationDomainsError,
    UnregistrableDomainError,
)
from gateway.services.tenancy.organization_domain_service import OrganizationDomainService

pytestmark = pytest.mark.asyncio

_SERVICE_MODULE = "gateway.services.tenancy.organization_domain_service"


def _resolver(records: dict[str, list[str]]) -> Any:
    """A stand-in resolver answering from a fixed zone."""

    async def _resolve(domain: str) -> list[str]:
        return records.get(domain, [])

    return _resolve


async def _organization(db: AsyncSession, *, slug: str) -> Organization:
    return await OrganizationRepository(db).create_organization(name=slug.title(), slug=slug, created_by_user_id=None)


async def _identity(
    db: AsyncSession,
    organization: Organization,
    *,
    role: str | None,
    email: str | None = None,
    email_verified: bool = True,
    full_name: str = "Someone",
) -> User:
    user = await UserRepository(db).create_local_identity(
        full_name=full_name,
        active_organization_id=organization.id,
        email=email,
    )
    if email is not None and email_verified:
        user.email_verified_at = datetime.now(UTC)
        db.add(user)
        await db.flush()
    if role is not None:
        await OrganizationMemberRepository(db).create_membership(
            organization_id=organization.id,
            user_id=user.id,
            role=role,
        )
    return user


async def _claim(
    db: AsyncSession,
    organization: Organization,
    *,
    domain: str,
    default_role: str = "member",
    enabled: bool = True,
    verified: bool = True,
) -> OrganizationDomain:
    row = await OrganizationDomainRepository(db).create_domain(
        organization_id=organization.id,
        domain=domain,
        default_role=default_role,
        enabled=enabled,
        verification_token="tok" + uuid.uuid4().hex[:16],
    )
    if verified:
        row.verified_at = datetime.now(UTC)
        db.add(row)
        await db.flush()
    return row


# =============================================================================
# Who may manage a claim
# =============================================================================


@pytest.mark.parametrize("role", ["member", "viewer"])
async def test_a_plain_member_may_not_read_the_claims(async_db: AsyncSession, role: str) -> None:
    """Reading is gated as tightly as writing: a row carries the record to publish."""
    organization = await _organization(async_db, slug="acme")
    caller = await _identity(async_db, organization, role=role, email="member@acme.example")

    with pytest.raises(NotAuthorizedError):
        await OrganizationDomainService(async_db).list_domains_for_user(user=caller)


@pytest.mark.parametrize("role", ["member", "viewer"])
async def test_a_plain_member_may_not_claim_a_domain(async_db: AsyncSession, role: str) -> None:
    organization = await _organization(async_db, slug="acme")
    caller = await _identity(async_db, organization, role=role, email="member@acme.example")

    with pytest.raises(NotAuthorizedError):
        await OrganizationDomainService(async_db).create_domain_for_user(
            user=caller,
            request=OrganizationDomainCreateRequest(domain="acme.example"),
        )


async def test_another_organizations_claim_is_not_found_rather_than_forbidden(async_db: AsyncSession) -> None:
    """A neighbour's claim must not be distinguishable from one that never existed."""
    theirs = await _organization(async_db, slug="theirs")
    mine = await _organization(async_db, slug="mine")
    their_claim = await _claim(async_db, theirs, domain="theirs.example")
    admin = await _identity(async_db, mine, role="admin", email="admin@mine.example")

    with pytest.raises(OrganizationDomainNotFoundError):
        await OrganizationDomainService(async_db).delete_domain_for_user(
            user=admin,
            organization_domain_id=their_claim.id,
        )


# =============================================================================
# What may be claimed
# =============================================================================


@pytest.mark.parametrize("domain", ["gmail.com", "mail.gmail.com", "Yahoo.CO.UK"])
async def test_a_public_provider_cannot_be_claimed(async_db: AsyncSession, domain: str) -> None:
    organization = await _organization(async_db, slug="acme")
    admin = await _identity(async_db, organization, role="admin", email="admin@acme.example")

    with pytest.raises(PublicEmailDomainError):
        await OrganizationDomainService(async_db).create_domain_for_user(
            user=admin,
            request=OrganizationDomainCreateRequest(domain=domain),
        )


@pytest.mark.parametrize("domain", ["not-a-domain", "https://acme.example", "acme.example/path"])
async def test_a_value_that_is_not_a_domain_is_refused(async_db: AsyncSession, domain: str) -> None:
    organization = await _organization(async_db, slug="acme")
    admin = await _identity(async_db, organization, role="admin", email="admin@acme.example")

    with pytest.raises(UnregistrableDomainError):
        await OrganizationDomainService(async_db).create_domain_for_user(
            user=admin,
            request=OrganizationDomainCreateRequest(domain=domain),
        )


async def test_a_domain_another_organization_holds_is_a_conflict(async_db: AsyncSession) -> None:
    theirs = await _organization(async_db, slug="theirs")
    mine = await _organization(async_db, slug="mine")
    await _claim(async_db, theirs, domain="contested.example")
    admin = await _identity(async_db, mine, role="admin", email="admin@mine.example")

    with pytest.raises(OrganizationDomainAlreadyClaimedError) as raised:
        await OrganizationDomainService(async_db).create_domain_for_user(
            user=admin,
            request=OrganizationDomainCreateRequest(domain="contested.example"),
        )
    # The holder is deliberately absent: naming it would make this endpoint a
    # probe for which domains other tenants have registered.
    assert "theirs" not in str(raised.value).lower()


async def test_a_claim_lands_unverified_and_carries_the_record_to_publish(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme")
    admin = await _identity(async_db, organization, role="admin", email="admin@acme.example")

    created = await OrganizationDomainService(async_db).create_domain_for_user(
        user=admin,
        request=OrganizationDomainCreateRequest(domain="  ADMIN@Acme.Example  "),
    )

    assert created.domain == "acme.example"
    assert created.verified_at is None
    assert created.verification_record.startswith(DOMAIN_VERIFICATION_TXT_PREFIX)
    assert created.default_role == "member"
    assert created.enabled is True


# =============================================================================
# The DNS proof
# =============================================================================


async def test_verifying_finds_the_published_record(
    async_db: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    organization = await _organization(async_db, slug="acme")
    admin = await _identity(async_db, organization, role="admin", email="admin@acme.example")
    claim = await _claim(async_db, organization, domain="acme.example", verified=False)

    monkeypatch.setattr(
        f"{_SERVICE_MODULE}.resolve_txt_records",
        _resolver({"acme.example": ["v=spf1 -all", f"{DOMAIN_VERIFICATION_TXT_PREFIX}{claim.verification_token}"]}),
    )
    verified = await OrganizationDomainService(async_db).verify_domain_for_user(
        user=admin,
        organization_domain_id=claim.id,
    )

    assert verified.verified_at is not None


async def test_verifying_refuses_when_the_record_is_absent_or_belongs_to_another_claim(
    async_db: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record with the right prefix but the wrong token proves nothing."""
    organization = await _organization(async_db, slug="acme")
    admin = await _identity(async_db, organization, role="admin", email="admin@acme.example")
    claim = await _claim(async_db, organization, domain="acme.example", verified=False)

    monkeypatch.setattr(
        f"{_SERVICE_MODULE}.resolve_txt_records",
        _resolver({"acme.example": [f"{DOMAIN_VERIFICATION_TXT_PREFIX}someone-elses-token"]}),
    )
    with pytest.raises(OrganizationDomainNotVerifiedError):
        await OrganizationDomainService(async_db).verify_domain_for_user(
            user=admin,
            organization_domain_id=claim.id,
        )

    await async_db.refresh(claim)
    assert claim.verified_at is None


async def test_verifying_an_already_verified_claim_does_not_look_up_again(
    async_db: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Idempotent, and the stamp does not move: a double click costs nothing."""
    organization = await _organization(async_db, slug="acme")
    admin = await _identity(async_db, organization, role="admin", email="admin@acme.example")
    claim = await _claim(async_db, organization, domain="acme.example", verified=True)
    first_verified_at = claim.verified_at

    async def _explode(domain: str) -> list[str]:
        raise AssertionError("an already-verified claim must not be looked up again")

    monkeypatch.setattr(f"{_SERVICE_MODULE}.resolve_txt_records", _explode)
    again = await OrganizationDomainService(async_db).verify_domain_for_user(
        user=admin,
        organization_domain_id=claim.id,
    )

    assert again.verified_at == first_verified_at


async def test_a_resolver_that_cannot_answer_reads_as_proof_not_found(
    async_db: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every DNS failure mode is one answer to the caller."""
    organization = await _organization(async_db, slug="acme")
    admin = await _identity(async_db, organization, role="admin", email="admin@acme.example")
    claim = await _claim(async_db, organization, domain="acme.example", verified=False)

    monkeypatch.setattr(f"{_SERVICE_MODULE}.resolve_txt_records", _resolver({}))
    with pytest.raises(OrganizationDomainNotVerifiedError):
        await OrganizationDomainService(async_db).verify_domain_for_user(
            user=admin,
            organization_domain_id=claim.id,
        )


async def test_the_verification_state_cannot_be_set_through_an_update(async_db: AsyncSession) -> None:
    """The only way to ``verified_at`` is the DNS proof."""
    organization = await _organization(async_db, slug="acme")
    admin = await _identity(async_db, organization, role="admin", email="admin@acme.example")
    claim = await _claim(async_db, organization, domain="acme.example", verified=False)

    updated = await OrganizationDomainService(async_db).update_domain_for_user(
        user=admin,
        organization_domain_id=claim.id,
        request=OrganizationDomainUpdateRequest(default_role="viewer", enabled=False),
    )

    assert updated.default_role == "viewer"
    assert updated.enabled is False
    assert updated.verified_at is None
    assert updated.domain == "acme.example"


# =============================================================================
# Auto-join
# =============================================================================


async def test_a_verified_address_joins_the_organization_that_proved_its_domain(
    async_db: AsyncSession,
) -> None:
    organization = await _organization(async_db, slug="acme")
    await _claim(async_db, organization, domain="acme.example", default_role="viewer")
    newcomer = await _identity(async_db, organization, role=None, email="new@acme.example")

    membership = await OrganizationDomainService(async_db).auto_join_for_user(newcomer)

    assert membership is not None
    assert membership.organization_id == organization.id
    assert membership.role == "viewer"
    assert membership.status == "active"


async def test_an_unverified_address_does_not_join(async_db: AsyncSession) -> None:
    """Otherwise signing up as anyone@theircompany.com would be enough to get in."""
    organization = await _organization(async_db, slug="acme")
    await _claim(async_db, organization, domain="acme.example")
    newcomer = await _identity(async_db, organization, role=None, email="new@acme.example", email_verified=False)

    assert await OrganizationDomainService(async_db).auto_join_for_user(newcomer) is None


async def test_an_unproven_claim_sweeps_in_nobody(async_db: AsyncSession) -> None:
    """The whole point of the DNS proof: a claim on someone else's domain is inert."""
    organization = await _organization(async_db, slug="squatter")
    await _claim(async_db, organization, domain="acme.example", verified=False)
    newcomer = await _identity(async_db, organization, role=None, email="new@acme.example")

    assert await OrganizationDomainService(async_db).auto_join_for_user(newcomer) is None


async def test_a_disabled_claim_sweeps_in_nobody(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme")
    await _claim(async_db, organization, domain="acme.example", enabled=False)
    newcomer = await _identity(async_db, organization, role=None, email="new@acme.example")

    assert await OrganizationDomainService(async_db).auto_join_for_user(newcomer) is None


async def test_an_unclaimed_domain_is_the_ordinary_no_match(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme")
    newcomer = await _identity(async_db, organization, role=None, email="new@elsewhere.example")

    assert await OrganizationDomainService(async_db).auto_join_for_user(newcomer) is None


async def test_an_identity_with_no_address_is_skipped(async_db: AsyncSession) -> None:
    """The deployment's master-key operator signs in through the same route."""
    organization = await _organization(async_db, slug="acme")
    await _claim(async_db, organization, domain="acme.example")
    operator = await _identity(async_db, organization, role=None, email=None)

    assert await OrganizationDomainService(async_db).auto_join_for_user(operator) is None


async def test_a_suspended_member_is_not_re_added_by_signing_in(async_db: AsyncSession) -> None:
    """Somebody was removed on purpose; the next sign-in must not undo that."""
    organization = await _organization(async_db, slug="acme")
    await _claim(async_db, organization, domain="acme.example")
    removed = await _identity(async_db, organization, role=None, email="removed@acme.example")
    members = OrganizationMemberRepository(async_db)
    await members.create_membership(
        organization_id=organization.id,
        user_id=removed.id,
        role="member",
        status="suspended",
    )

    membership = await OrganizationDomainService(async_db).auto_join_for_user(removed)

    assert membership is not None
    assert membership.status == "suspended"


async def test_an_established_role_is_never_overwritten_by_the_claims_default(
    async_db: AsyncSession,
) -> None:
    organization = await _organization(async_db, slug="acme")
    await _claim(async_db, organization, domain="acme.example", default_role="viewer")
    owner = await _identity(async_db, organization, role="owner", email="owner@acme.example")

    membership = await OrganizationDomainService(async_db).auto_join_for_user(owner)

    assert membership is not None
    assert membership.role == "owner"


async def test_auto_join_never_moves_where_the_caller_is_pointed(async_db: AsyncSession) -> None:
    """Adding a membership is one thing; hijacking the active organization is another."""
    home = await _organization(async_db, slug="home")
    claiming = await _organization(async_db, slug="claiming")
    await _claim(async_db, claiming, domain="acme.example")
    person = await _identity(async_db, home, role="member", email="person@acme.example")
    pointed_at = person.active_organization_id

    membership = await OrganizationDomainService(async_db).auto_join_for_user(person)

    assert membership is not None
    assert membership.organization_id == claiming.id
    await async_db.refresh(person)
    assert person.active_organization_id == pointed_at


async def test_verifying_a_domain_later_sweeps_in_the_accounts_that_already_existed(
    async_db: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ordinary case: colleagues already use the deployment when the domain is claimed."""
    organization = await _organization(async_db, slug="acme")
    admin = await _identity(async_db, organization, role="admin", email="admin@acme.example")
    existing = await _identity(async_db, organization, role=None, email="already-here@acme.example")
    claim = await _claim(async_db, organization, domain="acme.example", verified=False)
    service = OrganizationDomainService(async_db)

    assert await service.auto_join_for_user(existing) is None

    monkeypatch.setattr(
        f"{_SERVICE_MODULE}.resolve_txt_records",
        _resolver({"acme.example": [f"{DOMAIN_VERIFICATION_TXT_PREFIX}{claim.verification_token}"]}),
    )
    await service.verify_domain_for_user(user=admin, organization_domain_id=claim.id)

    assert await service.auto_join_for_user(existing) is not None


async def test_dropping_a_claim_leaves_the_members_it_already_admitted(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme")
    claim = await _claim(async_db, organization, domain="acme.example")
    admin = await _identity(async_db, organization, role="admin", email="admin@acme.example")
    joined = await _identity(async_db, organization, role=None, email="joined@acme.example")
    service = OrganizationDomainService(async_db)
    assert await service.auto_join_for_user(joined) is not None

    await service.delete_domain_for_user(user=admin, organization_domain_id=claim.id)

    still_there = await OrganizationMemberRepository(async_db).get_by_organization_and_user(
        organization.id,
        joined.id,
    )
    assert still_there is not None
    assert still_there.status == "active"
    # And the claim no longer admits anyone new.
    latecomer = await _identity(async_db, organization, role=None, email="late@acme.example")
    assert await service.auto_join_for_user(latecomer) is None


# =============================================================================
# The HTTP surface
# =============================================================================


def test_the_claim_lifecycle_over_http(
    client: TestClient,
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = client.post(
        "/v1/organizations/me/domains",
        json={"domain": "acme.example", "default_role": "viewer"},
        headers=master_key_header,
    )
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["verified_at"] is None
    assert body["default_role"] == "viewer"
    record = body["verification_record"]
    assert record.startswith(DOMAIN_VERIFICATION_TXT_PREFIX)
    # The raw token is never on the wire on its own.
    assert "verification_token" not in body

    listed = client.get("/v1/organizations/me/domains", headers=master_key_header)
    assert listed.status_code == 200
    assert [row["domain"] for row in listed.json()["data"]] == ["acme.example"]
    assert listed.json()["count"] == 1

    monkeypatch.setattr(f"{_SERVICE_MODULE}.resolve_txt_records", _resolver({"acme.example": [record]}))
    verified = client.post(
        f"/v1/organizations/me/domains/{body['id']}/verify",
        headers=master_key_header,
    )
    assert verified.status_code == 200, verified.text
    assert verified.json()["verified_at"] is not None

    patched = client.patch(
        f"/v1/organizations/me/domains/{body['id']}",
        json={"enabled": False},
        headers=master_key_header,
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["enabled"] is False

    removed = client.delete(f"/v1/organizations/me/domains/{body['id']}", headers=master_key_header)
    assert removed.status_code == 200, removed.text
    assert client.get("/v1/organizations/me/domains", headers=master_key_header).json()["count"] == 0


def test_a_public_provider_is_refused_over_http(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.post(
        "/v1/organizations/me/domains",
        json={"domain": "gmail.com"},
        headers=master_key_header,
    )
    assert response.status_code == 400, response.text
    assert "public email provider" in response.json()["detail"]


def test_an_unverified_claim_cannot_be_verified_without_the_record(
    client: TestClient,
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = client.post(
        "/v1/organizations/me/domains",
        json={"domain": "acme.example"},
        headers=master_key_header,
    ).json()

    monkeypatch.setattr(f"{_SERVICE_MODULE}.resolve_txt_records", _resolver({}))
    response = client.post(
        f"/v1/organizations/me/domains/{created['id']}/verify",
        headers=master_key_header,
    )
    assert response.status_code == 400, response.text
    assert "TXT record" in response.json()["detail"]


def test_a_management_role_a_domain_may_not_hand_out_is_refused_by_the_schema(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Owner and admin are absent from the request Literal, so this is a 422."""
    response = client.post(
        "/v1/organizations/me/domains",
        json={"domain": "acme.example", "default_role": "admin"},
        headers=master_key_header,
    )
    assert response.status_code == 422, response.text


def test_an_unknown_claim_is_a_404(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.delete(
        f"/v1/organizations/me/domains/{uuid.uuid4()}",
        headers=master_key_header,
    )
    assert response.status_code == 404, response.text


# =============================================================================
# Who a claim belongs to when two organizations want it
# =============================================================================


async def test_an_unproven_claim_does_not_lock_out_the_domains_real_owner(
    async_db: AsyncSession,
) -> None:
    """The trap a plain UNIQUE(domain) sets.

    Creating an organization takes no privilege, so making the *claim*
    exclusive would let anyone claim a domain they do not own and shut its real
    owner out for good, behind a 409 that deliberately cannot say who to ask.
    """
    squatter = await _organization(async_db, slug="squatter")
    real = await _organization(async_db, slug="real")
    await _claim(async_db, squatter, domain="contested.example", verified=False)
    owner = await _identity(async_db, real, role="admin", email="admin@real.example")

    mine = await OrganizationDomainService(async_db).create_domain_for_user(
        user=owner,
        request=OrganizationDomainCreateRequest(domain="contested.example"),
    )

    assert mine.organization_id == real.id
    assert mine.verified_at is None


async def test_proving_a_domain_displaces_the_unproven_claims_on_it(
    async_db: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A beaten claim is removed, not left to sit as un-verifiable forever."""
    squatter = await _organization(async_db, slug="squatter")
    real = await _organization(async_db, slug="real")
    beaten = await _claim(async_db, squatter, domain="contested.example", verified=False)
    owner = await _identity(async_db, real, role="admin", email="admin@real.example")
    service = OrganizationDomainService(async_db)
    mine = await service.create_domain_for_user(
        user=owner,
        request=OrganizationDomainCreateRequest(domain="contested.example"),
    )

    monkeypatch.setattr(
        f"{_SERVICE_MODULE}.resolve_txt_records",
        _resolver({"contested.example": [mine.verification_record]}),
    )
    await service.verify_domain_for_user(user=owner, organization_domain_id=mine.id)

    repository = OrganizationDomainRepository(async_db)
    assert await repository.get(beaten.id) is None
    holder = await repository.get_verified_by_domain("contested.example")
    assert holder is not None
    assert holder.organization_id == real.id


async def test_a_proven_domain_cannot_then_be_claimed_by_anyone_else(
    async_db: AsyncSession,
) -> None:
    real = await _organization(async_db, slug="real")
    latecomer = await _organization(async_db, slug="latecomer")
    await _claim(async_db, real, domain="contested.example", verified=True)
    admin = await _identity(async_db, latecomer, role="admin", email="admin@latecomer.example")

    with pytest.raises(OrganizationDomainAlreadyClaimedError) as raised:
        await OrganizationDomainService(async_db).create_domain_for_user(
            user=admin,
            request=OrganizationDomainCreateRequest(domain="contested.example"),
        )
    assert "real" not in str(raised.value).lower()


async def test_verifying_is_refused_once_another_organization_has_proven_the_domain(
    async_db: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loser of the race gets an answer they can read, not a constraint error."""
    winner = await _organization(async_db, slug="winner")
    loser = await _organization(async_db, slug="loser")
    await _claim(async_db, winner, domain="contested.example", verified=True)
    mine = await _claim(async_db, loser, domain="contested.example", verified=False)
    admin = await _identity(async_db, loser, role="admin", email="admin@loser.example")

    monkeypatch.setattr(
        f"{_SERVICE_MODULE}.resolve_txt_records",
        _resolver({"contested.example": [f"{DOMAIN_VERIFICATION_TXT_PREFIX}{mine.verification_token}"]}),
    )
    with pytest.raises(OrganizationDomainAlreadyClaimedError):
        await OrganizationDomainService(async_db).verify_domain_for_user(
            user=admin,
            organization_domain_id=mine.id,
        )


async def test_the_same_organization_cannot_claim_one_domain_twice(async_db: AsyncSession) -> None:
    """Named plainly, unlike the cross-tenant refusal: this row is the caller's own."""
    organization = await _organization(async_db, slug="acme")
    admin = await _identity(async_db, organization, role="admin", email="admin@acme.example")
    service = OrganizationDomainService(async_db)
    await service.create_domain_for_user(
        user=admin,
        request=OrganizationDomainCreateRequest(domain="acme.example"),
    )

    with pytest.raises(OrganizationDomainClaimedHereError):
        await service.create_domain_for_user(
            user=admin,
            request=OrganizationDomainCreateRequest(domain="acme.example"),
        )


async def test_an_organization_is_capped_on_how_many_domains_it_may_claim(
    async_db: AsyncSession,
) -> None:
    """Every unverified claim is a name this deployment will resolve on demand."""
    organization = await _organization(async_db, slug="acme")
    admin = await _identity(async_db, organization, role="admin", email="admin@acme.example")
    for index in range(MAX_ORGANIZATION_DOMAINS):
        await _claim(async_db, organization, domain=f"claim{index}.example", verified=False)

    with pytest.raises(TooManyOrganizationDomainsError):
        await OrganizationDomainService(async_db).create_domain_for_user(
            user=admin,
            request=OrganizationDomainCreateRequest(domain="one-too-many.example"),
        )


# =============================================================================
# A proof does not last forever
# =============================================================================


async def test_a_proof_that_has_aged_out_admits_nobody(async_db: AsyncSession) -> None:
    """Domains change hands; a stamp kept forever would admit whoever owns it next."""
    organization = await _organization(async_db, slug="acme")
    claim = await _claim(async_db, organization, domain="acme.example")
    claim.verified_at = datetime.now(UTC) - DOMAIN_PROOF_TTL - timedelta(days=1)
    async_db.add(claim)
    await async_db.flush()
    newcomer = await _identity(async_db, organization, role=None, email="new@acme.example")

    assert await OrganizationDomainService(async_db).auto_join_for_user(newcomer) is None


async def test_a_proof_inside_its_window_still_admits(async_db: AsyncSession) -> None:
    """The other side of the boundary, so the TTL is not passing by refusing everything."""
    organization = await _organization(async_db, slug="acme")
    claim = await _claim(async_db, organization, domain="acme.example")
    claim.verified_at = datetime.now(UTC) - DOMAIN_PROOF_TTL + timedelta(days=1)
    async_db.add(claim)
    await async_db.flush()
    newcomer = await _identity(async_db, organization, role=None, email="new@acme.example")

    assert await OrganizationDomainService(async_db).auto_join_for_user(newcomer) is not None


async def test_re_verifying_an_aged_out_proof_runs_the_lookup_again(
    async_db: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """How a claim is renewed: the short circuit applies to a fresh proof only."""
    organization = await _organization(async_db, slug="acme")
    admin = await _identity(async_db, organization, role="admin", email="admin@acme.example")
    claim = await _claim(async_db, organization, domain="acme.example")
    stale = datetime.now(UTC) - DOMAIN_PROOF_TTL - timedelta(days=1)
    claim.verified_at = stale
    async_db.add(claim)
    await async_db.flush()

    monkeypatch.setattr(
        f"{_SERVICE_MODULE}.resolve_txt_records",
        _resolver({"acme.example": [f"{DOMAIN_VERIFICATION_TXT_PREFIX}{claim.verification_token}"]}),
    )
    renewed = await OrganizationDomainService(async_db).verify_domain_for_user(
        user=admin,
        organization_domain_id=claim.id,
    )

    assert renewed.verified_at is not None
    assert renewed.verified_at > stale
    newcomer = await _identity(async_db, organization, role=None, email="new@acme.example")
    assert await OrganizationDomainService(async_db).auto_join_for_user(newcomer) is not None


async def test_a_record_pulled_after_verification_stops_mattering_only_at_the_ttl(
    async_db: AsyncSession,
) -> None:
    """Stated so the window is a decision on the record rather than an accident.

    Nothing re-reads DNS between verifications, so a domain that changes hands
    keeps admitting people until the proof ages out. That is the exposure the
    TTL bounds, and shortening it is the lever if it is ever judged too long.
    """
    organization = await _organization(async_db, slug="acme")
    claim = await _claim(async_db, organization, domain="acme.example")
    claim.verified_at = datetime.now(UTC) - DOMAIN_PROOF_TTL + timedelta(minutes=1)
    async_db.add(claim)
    await async_db.flush()
    newcomer = await _identity(async_db, organization, role=None, email="new@acme.example")

    assert await OrganizationDomainService(async_db).auto_join_for_user(newcomer) is not None
