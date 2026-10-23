"""Searching the organization roster.

The roster is what the dashboard's user pickers offer as options. They used to
fetch it whole and match in the browser, which silently offered a subset once
the roster passed a page (otari#1380). The match belongs where every row is, so
the page and the count narrow together.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import API_ROOT
from gateway.models.tenancy import Organization, User
from gateway.repositories.tenancy import OrganizationMemberRepository, OrganizationRepository, UserRepository
from gateway.services.tenancy.organization_service import OrganizationService

_ENDPOINT = f"{API_ROOT}/organizations/me/members"


async def _organization(db: AsyncSession, *, slug: str) -> Organization:
    return await OrganizationRepository(db).create_organization(name=slug.title(), slug=slug, created_by_user_id=None)


async def _member(
    db: AsyncSession,
    organization: Organization,
    *,
    full_name: str,
    email: str | None = None,
    role: str = "member",
) -> User:
    user = await UserRepository(db).create_local_identity(
        full_name=full_name,
        active_organization_id=organization.id,
        is_superuser=False,
    )
    if email is not None:
        user.email = email
        db.add(user)
        await db.flush()
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id,
        user_id=user.id,
        role=role,
    )
    return user


def _service(db: AsyncSession) -> OrganizationService:
    return OrganizationService(db, membership_listener=None)


@pytest.mark.asyncio
async def test_a_term_narrows_on_the_name(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme-search-name")
    caller = await _member(async_db, organization, full_name="Ada Lovelace", role="owner")
    await _member(async_db, organization, full_name="Grace Hopper")
    await _member(async_db, organization, full_name="Alan Turing")

    page = await _service(async_db).list_active_organization_members_for_user(user=caller, search="grace")

    assert [row.full_name for row in page.data] == ["Grace Hopper"]


@pytest.mark.asyncio
async def test_a_term_narrows_on_the_email_too(async_db: AsyncSession) -> None:
    """A picker is typed into by someone who knows an address, not only a name."""

    organization = await _organization(async_db, slug="acme-search-email")
    caller = await _member(async_db, organization, full_name="Owner", role="owner")
    await _member(async_db, organization, full_name="Grace Hopper", email="grace@navy.example")

    page = await _service(async_db).list_active_organization_members_for_user(user=caller, search="navy.example")

    assert [row.full_name for row in page.data] == ["Grace Hopper"]


@pytest.mark.asyncio
async def test_the_match_ignores_case(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme-search-case")
    caller = await _member(async_db, organization, full_name="Owner", role="owner")
    await _member(async_db, organization, full_name="Grace Hopper")

    page = await _service(async_db).list_active_organization_members_for_user(user=caller, search="HOPPER")

    assert [row.full_name for row in page.data] == ["Grace Hopper"]


@pytest.mark.asyncio
async def test_the_count_narrows_with_the_page(async_db: AsyncSession) -> None:
    """The whole point. A count that still reported the roster would put a total
    under a filtered list that describes something else, which is the lie the
    browser-side filter told."""

    organization = await _organization(async_db, slug="acme-search-count")
    caller = await _member(async_db, organization, full_name="Owner", role="owner")
    for index in range(5):
        await _member(async_db, organization, full_name=f"Member {index}")
    await _member(async_db, organization, full_name="Grace Hopper")

    page = await _service(async_db).list_active_organization_members_for_user(user=caller, search="grace")

    assert page.count == 1


@pytest.mark.asyncio
async def test_a_match_past_the_first_page_is_still_found(async_db: AsyncSession) -> None:
    """The failure this exists to stop: filtering a fetched page cannot see a
    member the page did not reach, so a search for somebody real said nobody."""

    organization = await _organization(async_db, slug="acme-search-deep")
    caller = await _member(async_db, organization, full_name="Aaa Owner", role="owner")
    for index in range(30):
        await _member(async_db, organization, full_name=f"Bbb Member {index:02d}")
    await _member(async_db, organization, full_name="Zzz Grace Hopper")

    unsearched = await _service(async_db).list_active_organization_members_for_user(user=caller, limit=10)
    searched = await _service(async_db).list_active_organization_members_for_user(user=caller, limit=10, search="grace")

    assert not any("Grace" in (row.full_name or "") for row in unsearched.data)
    assert [row.full_name for row in searched.data] == ["Zzz Grace Hopper"]


@pytest.mark.asyncio
async def test_a_wildcard_in_the_term_is_matched_literally(async_db: AsyncSession) -> None:
    """A picker takes whatever somebody types, so ``%`` is text and not a pattern."""

    organization = await _organization(async_db, slug="acme-search-wildcard")
    caller = await _member(async_db, organization, full_name="Owner", role="owner")
    await _member(async_db, organization, full_name="100% Cotton")
    await _member(async_db, organization, full_name="Grace Hopper")

    matched = await _service(async_db).list_active_organization_members_for_user(user=caller, search="100%")

    assert [row.full_name for row in matched.data] == ["100% Cotton"]


@pytest.mark.asyncio
async def test_an_underscore_is_matched_literally_too(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme-search-underscore")
    caller = await _member(async_db, organization, full_name="Owner", role="owner")
    await _member(async_db, organization, full_name="ci_bot")
    await _member(async_db, organization, full_name="cixbot")

    matched = await _service(async_db).list_active_organization_members_for_user(user=caller, search="ci_b")

    assert [row.full_name for row in matched.data] == ["ci_bot"]


@pytest.mark.asyncio
async def test_a_blank_term_is_no_filter(async_db: AsyncSession) -> None:
    """A cleared search box is not a search for the empty string."""

    organization = await _organization(async_db, slug="acme-search-blank")
    caller = await _member(async_db, organization, full_name="Owner", role="owner")
    await _member(async_db, organization, full_name="Grace Hopper")

    page = await _service(async_db).list_active_organization_members_for_user(user=caller, search="   ")

    assert page.count == 2


@pytest.mark.asyncio
async def test_the_search_stays_inside_the_callers_organization(async_db: AsyncSession) -> None:
    """Searching is not a way around the tenant boundary."""

    mine = await _organization(async_db, slug="acme-search-mine")
    theirs = await _organization(async_db, slug="acme-search-theirs")
    caller = await _member(async_db, mine, full_name="Owner", role="owner")
    await _member(async_db, theirs, full_name="Grace Hopper")

    page = await _service(async_db).list_active_organization_members_for_user(user=caller, search="grace")

    assert page.count == 0


def test_the_route_takes_the_term(client: TestClient, master_key_header: dict[str, str]) -> None:
    """And bounds it, so the parameter cannot carry an unbounded string."""

    assert client.get(_ENDPOINT, params={"search": "nobody"}, headers=master_key_header).status_code == 200
    over = client.get(_ENDPOINT, params={"search": "x" * 201}, headers=master_key_header)
    assert over.status_code == 422


def test_an_absent_term_lists_the_roster(client: TestClient, master_key_header: dict[str, str]) -> None:
    listed = client.get(_ENDPOINT, headers=master_key_header)

    assert listed.status_code == 200
    assert listed.json()["count"] >= 1


def test_an_unknown_term_is_an_empty_page_rather_than_an_error(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    response = client.get(_ENDPOINT, params={"search": str(uuid.uuid4())}, headers=master_key_header)

    assert response.status_code == 200
    assert response.json() == {"data": [], "count": 0}
