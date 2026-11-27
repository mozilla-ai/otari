"""Writing the rows a member is made of, for a test that needs one before its subject runs.

A test whose subject is something else should not drive an endpoint to get a member, because
that couples the two and the test then fails when the endpoint changes. These functions write
the rows instead.

NOTE: The member holds the address and nothing else, deliberately: no password,
``email_verified_at`` unset and ``email_proven_at`` unset. An address is recorded as proved only
where whoever controls it took part, and nobody does here. Two unset columns are the point rather
than an oversight to fix.

``tests/integration/test_seed_member_helper.py`` holds what is written here against what the
application writes, so neither can drift into a state the other never produces.
"""

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlmodel import col

from gateway.core.config import API_ROOT
from gateway.models.tenancy import OrganizationMember, OrganizationMemberRole
from gateway.models.tenancy import User as TenancyUser
from gateway.models.users import User as AttributionUser


@dataclass(frozen=True)
class SeededMember:
    """The identity and membership rows one seeded member is made of."""

    user_id: uuid.UUID
    organization_member_id: uuid.UUID


def write_member_rows(
    session: Session,
    *,
    email: str,
    organization_id: uuid.UUID,
    role: OrganizationMemberRole = "member",
) -> SeededMember:
    """Write the three rows a member at ``email`` is made of, and commit.

    NOTE: Rows are written on ``session`` and committed, so a caller holding an open write on the
    organization row elsewhere will block against itself.

    NOTE: Each call mints an identity, so one address may be seeded once per test. Seeding it twice
    is a mistake rather than a case to handle, and is refused.
    """
    address = email.strip().lower()
    # Matched the way the application matches an address, because the unique
    # index is case-sensitive and an exact match would miss a row it holds.
    held = session.scalars(select(TenancyUser).where(func.lower(col(TenancyUser.email)) == address)).first()
    assert held is None, f"{address} already has an identity, so seeding it again would conflict"

    user = TenancyUser(
        email=address,
        full_name=None,
        active_organization_id=organization_id,
        default_organization_id=organization_id,
    )
    session.add(user)
    session.flush()
    membership = OrganizationMember(organization_id=organization_id, user_id=user.id, role=role, status="active")
    session.add(membership)
    # Keys, budgets and usage hang off this row rather than off the identity,
    # and nothing joins the two, so a member without one cannot own a key.
    session.add(AttributionUser(user_id=str(user.id), alias=address))
    session.commit()
    return SeededMember(user_id=user.id, organization_member_id=membership.id)


def operator_organization_id(client: TestClient, headers: dict[str, str]) -> uuid.UUID:
    """The organization the deployment's own key acts in.

    NOTE: The request that resolves it also provisions the tenancy root on a deployment that has
    none, so a caller wanting the operator's organization must ask before setting a dashboard
    session cookie, which outranks the master key and would answer for its own organization.
    """
    response = client.get(f"{API_ROOT}/organizations/me", headers=headers)
    assert response.status_code == 200, response.text
    return uuid.UUID(response.json()["organization"]["id"])


@contextmanager
def opened_session(database_url: str) -> Iterator[Session]:
    """A session on the database at ``database_url``, disposing its engine on the way out.

    For a test that drives an application through its own client and holds no session of its own.
    """
    engine = create_engine(database_url)
    try:
        with sessionmaker(bind=engine)() as session:
            yield session
    finally:
        engine.dispose()


def seed_member_for_the_operator(
    client: TestClient,
    headers: dict[str, str],
    database_url: str,
    *,
    email: str,
    role: OrganizationMemberRole = "member",
) -> SeededMember:
    """Put a member at ``email`` into the organization the deployment's own key acts in.

    For a test that drives an application through its own client. A test that already holds a
    session on the same database calls :func:`write_member_rows` instead.
    """
    organization_id = operator_organization_id(client, headers)
    with opened_session(database_url) as session:
        return write_member_rows(session, email=email, organization_id=organization_id, role=role)
