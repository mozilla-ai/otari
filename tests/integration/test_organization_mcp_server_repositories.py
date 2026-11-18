"""Organization MCP servers at the repository layer: overrides, aliases, and what the schema refuses.

Nothing creates an organization server through a route yet, so the rows are written directly.
"""

import uuid
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.tenancy import Organization, Workspace
from gateway.models.tools import (
    McpServerAlias,
    WorkspaceMcpServer,
    WorkspaceMcpServerGrant,
    WorkspaceMcpServerOverride,
)
from gateway.models.users import User as BillingUser
from gateway.repositories.tenancy import OrganizationRepository, WorkspaceRepository
from gateway.repositories.tools import (
    McpServerAliasRepository,
    WorkspaceMcpServerGrantRepository,
    WorkspaceMcpServerOverrideRepository,
)

pytestmark = pytest.mark.asyncio


async def _organization(db: AsyncSession, slug: str = "acme") -> Organization:
    return await OrganizationRepository(db).create_organization(name=slug.title(), slug=slug, created_by_user_id=None)


async def _workspace(db: AsyncSession, organization: Organization, name: str) -> Workspace:
    return await WorkspaceRepository(db).create_workspace(
        name=name, organization_id=organization.id, created_by_user_id=None
    )


async def _billing_user(db: AsyncSession, user_id: str = "ada") -> str:
    db.add(BillingUser(user_id=user_id))
    await db.flush()
    return user_id


async def _server(
    db: AsyncSession, organization: Organization, name: str = "drive", workspace: Workspace | None = None
) -> WorkspaceMcpServer:
    server = WorkspaceMcpServer(
        organization_id=organization.id,
        workspace_id=workspace.id if workspace is not None else None,
        name=name,
        url=f"https://{name}.example/mcp",
    )
    db.add(server)
    await db.flush()
    return server


async def test_an_override_is_saved_read_and_removed(async_db: AsyncSession) -> None:
    acme = await _organization(async_db)
    marcomms = await _workspace(async_db, acme, "Marcomms")
    sales = await _workspace(async_db, acme, "Sales")
    drive = await _server(async_db, acme)
    overrides = WorkspaceMcpServerOverrideRepository(async_db)

    assert await overrides.find(workspace_id=marcomms.id, server_id=drive.id) is None
    saved = await overrides.save(
        None,
        workspace_id=marcomms.id,
        server=drive,
        organization_tools=["@read"],
        enabled=True,
        everyone_tools=None,
        attach_by_default=None,
    )
    await overrides.save(
        None,
        workspace_id=sales.id,
        server=drive,
        organization_tools=[],
        enabled=True,
        everyone_tools=None,
        attach_by_default=None,
    )
    assert saved is not None
    await overrides.save(
        saved,
        workspace_id=marcomms.id,
        server=drive,
        organization_tools=["@read"],
        enabled=False,
        everyone_tools=["search"],
        attach_by_default=True,
    )
    await async_db.commit()

    found = await overrides.find(workspace_id=marcomms.id, server_id=drive.id)
    assert found is not None
    await async_db.refresh(found)  # read back from the row, not the values just assigned
    assert (found.organization_tools, found.enabled, found.everyone_tools, found.attach_by_default) == (
        ["@read"],
        False,
        ["search"],
        True,
    )
    assert {key: value.id for key, value in (await overrides.for_workspace(marcomms.id, [drive.id])).items()} == {
        drive.id: saved.id
    }
    unconfigured = await _workspace(async_db, acme, "Support")
    assert await overrides.for_workspace(unconfigured.id, [drive.id]) == {}
    assert await overrides.for_workspace(marcomms.id, []) == {}
    assert [override.workspace_id for override in await overrides.for_server(drive.id)] == [marcomms.id, sales.id]

    await overrides.remove(found)
    await async_db.commit()
    assert await overrides.find(workspace_id=marcomms.id, server_id=drive.id) is None


async def test_settings_left_at_their_defaults_are_no_override(async_db: AsyncSession) -> None:
    acme = await _organization(async_db)
    marcomms = await _workspace(async_db, acme, "Marcomms")
    drive = await _server(async_db, acme)
    overrides = WorkspaceMcpServerOverrideRepository(async_db)
    defaults: dict[str, Any] = {
        "organization_tools": None,
        "enabled": True,
        "everyone_tools": None,
        "attach_by_default": None,
    }

    assert await overrides.save(None, workspace_id=marcomms.id, server=drive, **defaults) is None
    limited = await overrides.save(
        None, workspace_id=marcomms.id, server=drive, **{**defaults, "organization_tools": ["@read"]}
    )
    assert limited is not None
    assert await overrides.save(limited, workspace_id=marcomms.id, server=drive, **defaults) is None
    await async_db.commit()

    assert await overrides.find(workspace_id=marcomms.id, server_id=drive.id) is None


async def test_an_override_is_changed_only_for_its_own_workspace_and_server(async_db: AsyncSession) -> None:
    acme = await _organization(async_db)
    marcomms = await _workspace(async_db, acme, "Marcomms")
    sales = await _workspace(async_db, acme, "Sales")
    drive = await _server(async_db, acme)
    overrides = WorkspaceMcpServerOverrideRepository(async_db)
    settings: dict[str, Any] = {
        "organization_tools": ["@read"],
        "enabled": True,
        "everyone_tools": None,
        "attach_by_default": None,
    }
    sales_override = await overrides.save(None, workspace_id=sales.id, server=drive, **settings)

    with pytest.raises(ValueError, match="another workspace"):
        await overrides.save(sales_override, workspace_id=marcomms.id, server=drive, **{**settings, "enabled": False})
    assert sales_override is not None and sales_override.enabled is True


async def test_a_workspace_has_one_override_per_server(async_db: AsyncSession) -> None:
    acme = await _organization(async_db)
    marcomms = await _workspace(async_db, acme, "Marcomms")
    drive = await _server(async_db, acme)
    for _ in range(2):
        async_db.add(WorkspaceMcpServerOverride(workspace_id=marcomms.id, organization_id=acme.id, server_id=drive.id))

    with pytest.raises(IntegrityError):
        await async_db.flush()
    await async_db.rollback()


@pytest.mark.parametrize("claimed", ["acme", "globex"])
async def test_an_override_holds_its_workspace_and_server_to_one_organization(
    async_db: AsyncSession, claimed: str
) -> None:
    acme = await _organization(async_db)
    globex = await _organization(async_db, "globex")
    marcomms = await _workspace(async_db, acme, "Marcomms")
    theirs = await _server(async_db, globex)
    organization_id = acme.id if claimed == "acme" else globex.id
    async_db.add(
        WorkspaceMcpServerOverride(workspace_id=marcomms.id, organization_id=organization_id, server_id=theirs.id)
    )

    with pytest.raises(IntegrityError):
        await async_db.flush()
    await async_db.rollback()


@pytest.mark.parametrize("claimed", ["acme", "globex"])
async def test_an_alias_holds_its_workspace_and_server_to_one_organization(
    async_db: AsyncSession, claimed: str
) -> None:
    acme = await _organization(async_db)
    globex = await _organization(async_db, "globex")
    marcomms = await _workspace(async_db, acme, "Marcomms")
    theirs = await _server(async_db, globex)
    organization_id = acme.id if claimed == "acme" else globex.id
    async_db.add(
        McpServerAlias(
            old_id=uuid.uuid4(), organization_id=organization_id, workspace_id=marcomms.id, server_id=theirs.id
        )
    )

    with pytest.raises(IntegrityError):
        await async_db.flush()
    await async_db.rollback()


async def test_a_workspace_server_s_organization_is_its_workspace_s(async_db: AsyncSession) -> None:
    acme = await _organization(async_db)
    globex = await _organization(async_db, "globex")
    marcomms = await _workspace(async_db, acme, "Marcomms")
    async_db.add(
        WorkspaceMcpServer(organization_id=globex.id, workspace_id=marcomms.id, name="x", url="https://x.example/mcp")
    )

    with pytest.raises(IntegrityError):
        await async_db.flush()
    await async_db.rollback()


async def test_only_an_organization_s_server_takes_overrides_and_aliases(async_db: AsyncSession) -> None:
    acme = await _organization(async_db)
    marcomms = await _workspace(async_db, acme, "Marcomms")
    sales = await _workspace(async_db, acme, "Sales")
    theirs = await _server(async_db, acme, workspace=sales)

    with pytest.raises(ValueError, match="workspace's own"):
        await WorkspaceMcpServerOverrideRepository(async_db).save(
            None,
            workspace_id=marcomms.id,
            server=theirs,
            organization_tools=None,
            enabled=False,
            everyone_tools=None,
            attach_by_default=None,
        )
    with pytest.raises(ValueError, match="workspace's own"):
        await McpServerAliasRepository(async_db).add(uuid.uuid4(), workspace_id=marcomms.id, server=theirs)
    with pytest.raises(ValueError, match="its own workspace"):
        await WorkspaceMcpServerGrantRepository(async_db).put(
            theirs, workspace_id=marcomms.id, api_key_ids=[], user_ids=["ada"], tools=["*"]
        )


async def test_an_organization_server_name_is_unique_within_the_organization(async_db: AsyncSession) -> None:
    acme = await _organization(async_db)
    globex = await _organization(async_db, "globex")
    marcomms = await _workspace(async_db, acme, "Marcomms")
    await _server(async_db, acme)
    # A workspace server and another organization's server may share the name.
    await _server(async_db, acme, workspace=marcomms)
    await _server(async_db, globex)
    await async_db.commit()

    async_db.add(WorkspaceMcpServer(organization_id=acme.id, name="drive", url="https://other.example/mcp"))
    with pytest.raises(IntegrityError):
        await async_db.flush()
    await async_db.rollback()


async def test_a_user_holds_one_grant_per_workspace_on_an_organization_server(async_db: AsyncSession) -> None:
    acme = await _organization(async_db)
    marcomms = await _workspace(async_db, acme, "Marcomms")
    sales = await _workspace(async_db, acme, "Sales")
    drive = await _server(async_db, acme)
    await _billing_user(async_db)
    grants = WorkspaceMcpServerGrantRepository(async_db)

    await grants.put(drive, workspace_id=marcomms.id, api_key_ids=[], user_ids=["ada"], tools=["*"])
    await grants.put(drive, workspace_id=sales.id, api_key_ids=[], user_ids=["ada"], tools=["@read"])
    await async_db.commit()
    [sales_grant] = [grant.id for grant in await grants.for_server(drive.id) if grant.workspace_id == sales.id]
    await grants.put(drive, workspace_id=sales.id, api_key_ids=[], user_ids=["ada"], tools=[])
    await async_db.commit()

    held = {grant.workspace_id: (grant.id, grant.tools) for grant in await grants.for_server(drive.id)}
    assert held[sales.id] == (sales_grant, []), "replaced in place"
    assert held[marcomms.id][1] == ["*"]


async def test_an_alias_names_its_server_in_its_workspace_only(async_db: AsyncSession) -> None:
    acme = await _organization(async_db)
    marcomms = await _workspace(async_db, acme, "Marcomms")
    sales = await _workspace(async_db, acme, "Sales")
    drive = await _server(async_db, acme)
    old_id, unknown = uuid.uuid4(), uuid.uuid4()
    aliases = McpServerAliasRepository(async_db)

    await aliases.add(old_id, workspace_id=marcomms.id, server=drive)
    await async_db.commit()

    assert await aliases.targets(marcomms.id, [old_id, unknown, drive.id]) == {old_id: drive.id}
    assert await aliases.targets(sales.id, [old_id]) == {}
    assert await aliases.targets(marcomms.id, []) == {}


async def test_deleting_a_workspace_takes_its_rows_and_leaves_the_organization_server(async_db: AsyncSession) -> None:
    acme = await _organization(async_db)
    marcomms = await _workspace(async_db, acme, "Marcomms")
    sales = await _workspace(async_db, acme, "Sales")
    drive = await _server(async_db, acme)
    await _billing_user(async_db)
    async_db.add_all(
        [
            WorkspaceMcpServerOverride(workspace_id=marcomms.id, organization_id=acme.id, server_id=drive.id),
            McpServerAlias(old_id=uuid.uuid4(), organization_id=acme.id, workspace_id=marcomms.id, server_id=drive.id),
            WorkspaceMcpServerGrant(server_id=drive.id, workspace_id=marcomms.id, user_id="ada", tools=["*"]),
            WorkspaceMcpServerGrant(server_id=drive.id, workspace_id=sales.id, user_id="ada", tools=["*"]),
        ]
    )
    await async_db.commit()

    drive_id, sales_id = drive.id, sales.id
    await async_db.delete(marcomms)
    await async_db.commit()
    async_db.expire_all()

    assert await async_db.get(WorkspaceMcpServer, drive_id) is not None
    assert (await async_db.execute(select(WorkspaceMcpServerOverride))).scalars().all() == []
    assert (await async_db.execute(select(McpServerAlias))).scalars().all() == []
    remaining = (await async_db.execute(select(WorkspaceMcpServerGrant.workspace_id))).scalars().all()
    assert remaining == [sales_id]


async def test_deleting_an_organization_server_takes_its_overrides_and_aliases(async_db: AsyncSession) -> None:
    acme = await _organization(async_db)
    marcomms = await _workspace(async_db, acme, "Marcomms")
    drive = await _server(async_db, acme)
    async_db.add_all(
        [
            WorkspaceMcpServerOverride(workspace_id=marcomms.id, organization_id=acme.id, server_id=drive.id),
            McpServerAlias(old_id=uuid.uuid4(), organization_id=acme.id, workspace_id=marcomms.id, server_id=drive.id),
        ]
    )
    await async_db.commit()

    await async_db.delete(drive)
    await async_db.commit()

    assert (await async_db.execute(select(WorkspaceMcpServerOverride))).scalars().all() == []
    assert (await async_db.execute(select(McpServerAlias))).scalars().all() == []
