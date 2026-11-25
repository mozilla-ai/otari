"""What ``seed_member`` writes, held against what adding a member writes.

The fixture exists so a test that merely needs a member does not drive the
endpoint to get one. That is only safe while the two agree, because a fixture
free to invent a state the application never produces would let every test
built on it assert against one.

Compared here: every column that says what the address proved, whether the
account can sign in, which organization it belongs to, what it may do there,
and whether it can own a key. Ids and timestamps differ by construction and
are not compared.

NOTE: Each comparison needs an endpoint that adds a member to compare against.
Where none remains, they go, and what is left is the invariant stated outright
below, which names the seeded shape rather than deriving it from a second
writer.
"""

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.core.config import API_ROOT
from gateway.models.tenancy import OrganizationMember
from gateway.models.tenancy import User as TenancyUser
from gateway.models.users import User as AttributionUser

from .conftest import SeededMember

SEEDED = "seeded@example.com"
ADDED = "added@example.com"


def _identity(db_session: Session, email: str) -> TenancyUser:
    return db_session.scalars(select(TenancyUser).where(col(TenancyUser.email) == email)).one()


def _membership(db_session: Session, user_id: Any) -> OrganizationMember:
    return db_session.scalars(select(OrganizationMember).where(col(OrganizationMember.user_id) == user_id)).one()


@pytest.fixture
def added(client: TestClient, master_key_header: dict[str, str]) -> dict[str, Any]:
    response = client.post(
        f"{API_ROOT}/organizations/me/members",
        json={"email": ADDED, "role": "member"},
        headers=master_key_header,
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def test_a_seeded_identity_matches_one_the_endpoint_adds(
    seed_member: Callable[..., SeededMember], added: dict[str, Any], db_session: Session
) -> None:
    """Every column that says what the address proved, and whether it can sign in."""
    seed_member(email=SEEDED)

    seeded, endpoint = _identity(db_session, SEEDED), _identity(db_session, ADDED)

    assert (seeded.email_verified_at, seeded.email_proven_at) == (endpoint.email_verified_at, endpoint.email_proven_at)
    assert (seeded.hashed_password, seeded.is_active) == (endpoint.hashed_password, endpoint.is_active)
    assert (seeded.full_name, seeded.is_superuser) == (endpoint.full_name, endpoint.is_superuser)


def test_a_seeded_identity_proves_nothing_about_the_address(
    seed_member: Callable[..., SeededMember], db_session: Session
) -> None:
    """Stated outright, so the pair above cannot pass by both sides drifting together."""
    seed_member(email=SEEDED)

    seeded = _identity(db_session, SEEDED)

    assert seeded.email == SEEDED
    assert seeded.email_verified_at is None
    assert seeded.email_proven_at is None
    assert seeded.hashed_password is None


def test_a_seeded_identity_can_own_a_key(seed_member: Callable[..., SeededMember], db_session: Session) -> None:
    """Keys, budgets and usage hang off a request-plane row that nothing joins to the identity.

    Without it the member is listed but cannot be given a key, which is a state
    the endpoint never leaves a member in.
    """
    seeded = seed_member(email=SEEDED)

    owner = db_session.get(AttributionUser, str(seeded.user_id))
    assert owner is not None
    assert owner.alias == SEEDED
    assert owner.deleted_at is None


def test_a_seeded_identity_lands_in_the_same_organization(
    seed_member: Callable[..., SeededMember], added: dict[str, Any], db_session: Session
) -> None:
    """The two organization columns an identity carries, which decide what it sees on sign-in."""
    seed_member(email=SEEDED)

    seeded, endpoint = _identity(db_session, SEEDED), _identity(db_session, ADDED)

    assert seeded.active_organization_id == endpoint.active_organization_id
    assert seeded.default_organization_id == endpoint.default_organization_id


def test_a_seeded_membership_matches_one_the_endpoint_adds(
    seed_member: Callable[..., SeededMember], added: dict[str, Any], db_session: Session
) -> None:
    """The organization a member lands in, the role they hold, and whether they are active."""
    seeded_member: SeededMember = seed_member(email=SEEDED)

    seeded = _membership(db_session, seeded_member.user_id)
    endpoint = _membership(db_session, _identity(db_session, ADDED).id)

    assert (seeded.organization_id, seeded.role, seeded.status) == (
        endpoint.organization_id,
        endpoint.role,
        endpoint.status,
    )


def test_the_ids_it_returns_are_the_rows_it_wrote(
    seed_member: Callable[..., SeededMember], db_session: Session
) -> None:
    """Callers read members back by these, so a wrong one fails somewhere else entirely."""
    seeded_member: SeededMember = seed_member(email=SEEDED, role="admin")

    identity = _identity(db_session, SEEDED)
    membership = _membership(db_session, identity.id)

    assert seeded_member.user_id == identity.id
    assert seeded_member.organization_member_id == membership.id
    assert membership.role == "admin"


def test_seeding_one_address_twice_says_so(seed_member: Callable[..., SeededMember]) -> None:
    """A duplicate seed otherwise fails on the unique index, which leaves the session unusable.

    Every later read in that test then reports the broken session rather than
    the duplicate, so the test that caused it is not the one that reports it.
    """
    seed_member(email=SEEDED)

    with pytest.raises(AssertionError, match="already has an identity"):
        seed_member(email=SEEDED)


def test_a_signed_in_member_has_verified_the_address(
    signed_in_member: Callable[..., TenancyUser], db_session: Session
) -> None:
    """Signing in refuses an unverified address, so a member holding a session has verified.

    Unproved all the same: verifying says the address became usable, not that
    whoever controls it took part.
    """
    signed_in_member(email=SEEDED)

    identity = _identity(db_session, SEEDED)

    assert identity.email_verified_at is not None
    assert identity.email_proven_at is None
