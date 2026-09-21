"""Naming and searching the spend identities a user picker offers.

Keys, budgets and usage attach to the string-keyed ``users`` row; the person is
a UUID identity on the organization roster. The dashboard used to read the whole
roster into the browser to put a name on each id, and then matched what somebody
typed against the page it had fetched, which offered a subset of the options and
said nothing about it (otari#1380). The name comes with the row now, and the
match happens where every row is.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import API_ROOT
from gateway.models.api_keys import APIKey
from gateway.models.tenancy import Organization, User
from gateway.models.users import User as ApiUser
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.repositories.users_repository import list_users_in_organization, roster_names

_ENDPOINT = f"{API_ROOT}/users"


async def _organization(db: AsyncSession, *, slug: str) -> Organization:
    return await OrganizationRepository(db).create_organization(
        name=slug.title(), slug=slug, created_by_user_id=None
    )


async def _member(
    db: AsyncSession,
    organization: Organization,
    *,
    full_name: str | None = None,
    email: str | None = None,
) -> User:
    identity = await UserRepository(db).create_local_identity(
        full_name=full_name,
        active_organization_id=organization.id,
        is_superuser=False,
    )
    if email is not None:
        identity.email = email
        db.add(identity)
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id,
        user_id=identity.id,
        role="member",
    )
    # The spend row the roster identity attributes to, which is what a picker
    # offers and what a key is issued against.
    db.add(ApiUser(user_id=str(identity.id)))
    await db.flush()
    return identity


async def _reachable(db: AsyncSession, organization: Organization, owner: User) -> None:
    """Put the organization in reach of its users, which a key is one way to do."""

    workspace = await WorkspaceRepository(db).create_workspace(
        name="Engineering", organization_id=organization.id, created_by_user_id=owner.id
    )
    await WorkspaceMemberRepository(db).create(workspace_id=workspace.id, user_id=owner.id, role="owner")
    db.add(
        APIKey(
            id=f"sk-reach-{organization.slug}",
            key_hash=f"h-{organization.slug}",
            workspace_id=workspace.id,
            user_id=str(owner.id),
        )
    )
    await db.flush()


@pytest.mark.asyncio
async def test_a_row_carries_the_name_of_the_person_behind_it(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme-names")
    identity = await _member(async_db, organization, full_name="Grace Hopper")

    names = await roster_names(async_db, [str(identity.id)])

    assert names[str(identity.id)] == "Grace Hopper"


@pytest.mark.asyncio
async def test_the_address_names_an_identity_with_no_full_name(async_db: AsyncSession) -> None:
    """The roster's own precedence: a name, else the sign-in address."""

    organization = await _organization(async_db, slug="acme-names-email")
    identity = await _member(async_db, organization, email="grace@navy.example")

    names = await roster_names(async_db, [str(identity.id)])

    assert names[str(identity.id)] == "grace@navy.example"


@pytest.mark.asyncio
async def test_an_identity_with_neither_is_left_out_rather_than_placeheld(async_db: AsyncSession) -> None:
    """A caller with no name to show already knows what to print instead, so a
    placeholder here would only have to be recognized and undone."""

    organization = await _organization(async_db, slug="acme-names-neither")
    identity = await _member(async_db, organization)

    names = await roster_names(async_db, [str(identity.id)])

    assert str(identity.id) not in names


@pytest.mark.asyncio
async def test_an_id_that_is_not_a_member_id_resolves_to_no_name(async_db: AsyncSession) -> None:
    """An id an operator chose, like `ci-bot`, is not a UUID and has no roster
    row. Parsing it as one would raise rather than miss."""

    names = await roster_names(async_db, ["ci-bot", "not-a-uuid"])

    assert names == {}


@pytest.mark.asyncio
async def test_the_search_matches_the_roster_name(async_db: AsyncSession) -> None:
    """The whole point: a picker is typed into with the name it displays, which
    is not a column on the row it offers."""

    organization = await _organization(async_db, slug="acme-search-roster")
    owner = await _member(async_db, organization, full_name="Ada Lovelace")
    await _reachable(async_db, organization, owner)
    await _member(async_db, organization, full_name="Grace Hopper")

    found = await list_users_in_organization(async_db, organization_id=organization.id, search="hopper")
    named = await roster_names(async_db, [row.user_id for row in found])

    assert list(named.values()) == ["Grace Hopper"]


@pytest.mark.asyncio
async def test_the_search_matches_the_id_and_the_alias_too(async_db: AsyncSession) -> None:
    """Somebody who knows the id types the id."""

    organization = await _organization(async_db, slug="acme-search-id")
    owner = await _member(async_db, organization, full_name="Ada Lovelace")
    await _reachable(async_db, organization, owner)
    async_db.add(ApiUser(user_id="ci-bot", alias="Continuous integration"))
    await async_db.flush()

    by_id = await list_users_in_organization(async_db, organization_id=organization.id, search="ci-bot")
    by_alias = await list_users_in_organization(async_db, organization_id=organization.id, search="continuous")

    assert "ci-bot" in {row.user_id for row in by_id}
    assert "ci-bot" in {row.user_id for row in by_alias}


@pytest.mark.asyncio
async def test_a_match_past_the_first_page_is_still_found(async_db: AsyncSession) -> None:
    """The failure this exists to stop: matching a page already fetched cannot
    see somebody the page did not reach, so a search for a real person answered
    that there is none.

    Ids rather than roster names here, because the rows come back ordered by id
    and a member's id is a UUID nobody chooses; the property under test is the
    reach of the match, not the order.
    """

    organization = await _organization(async_db, slug="acme-search-deep")
    owner = await _member(async_db, organization, full_name="Owner")
    await _reachable(async_db, organization, owner)
    for index in range(30):
        async_db.add(ApiUser(user_id=f"aaa-{index:02d}", alias=f"Bot {index:02d}"))
    async_db.add(ApiUser(user_id="zzz-target", alias="Grace Hopper"))
    await async_db.flush()

    unsearched = await list_users_in_organization(async_db, organization_id=organization.id, limit=10)
    searched = await list_users_in_organization(
        async_db, organization_id=organization.id, limit=10, search="grace"
    )

    assert "zzz-target" not in {row.user_id for row in unsearched}
    assert [row.user_id for row in searched] == ["zzz-target"]


@pytest.mark.asyncio
async def test_a_wildcard_in_the_term_is_matched_literally(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme-search-wild")
    owner = await _member(async_db, organization, full_name="Owner")
    await _reachable(async_db, organization, owner)
    async_db.add(ApiUser(user_id="100%-bot"))
    async_db.add(ApiUser(user_id="other-bot"))
    await async_db.flush()

    found = await list_users_in_organization(async_db, organization_id=organization.id, search="100%")

    assert {row.user_id for row in found} == {"100%-bot"}


@pytest.mark.asyncio
async def test_a_blank_term_is_no_filter(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme-search-blank")
    owner = await _member(async_db, organization, full_name="Owner")
    await _reachable(async_db, organization, owner)

    found = await list_users_in_organization(async_db, organization_id=organization.id, search="   ")

    assert found != []


@pytest.mark.asyncio
async def test_the_search_cannot_reach_another_organizations_people(async_db: AsyncSession) -> None:
    """Narrowing only. The organization scope is derived from the joins that
    exist, and a term does not widen it."""

    mine = await _organization(async_db, slug="acme-search-mine")
    theirs = await _organization(async_db, slug="acme-search-theirs")
    owner = await _member(async_db, mine, full_name="Owner")
    await _reachable(async_db, mine, owner)
    their_owner = await _member(async_db, theirs, full_name="Grace Hopper")
    await _reachable(async_db, theirs, their_owner)

    found = await list_users_in_organization(async_db, organization_id=mine.id, search="grace")

    assert found == []


def test_the_route_takes_the_term_and_bounds_it(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    assert client.get(_ENDPOINT, params={"search": "nobody"}, headers=master_key_header).status_code == 200
    over = client.get(_ENDPOINT, params={"search": "x" * 201}, headers=master_key_header)
    assert over.status_code == 422


def test_the_route_publishes_the_display_name(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    listed = client.get(_ENDPOINT, headers=master_key_header)

    assert listed.status_code == 200
    # Present on every row, whatever its value: a picker reads it rather than
    # looking the person up itself.
    assert all("display_name" in row for row in listed.json())
