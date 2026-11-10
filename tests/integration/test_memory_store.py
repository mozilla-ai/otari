"""Agent memory at rest: sealed, fenced by workspace, and gone with its workspace or member."""

import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import timedelta

import pytest
import pytest_asyncio
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.adapters.content_key_adapter import build_content_key_port
from gateway.core.config import GatewayConfig
from gateway.core.database import dispose_db, init_db
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.exceptions.memory_exceptions import MemorySealingUnavailableError
from gateway.models.memory import MemoryEntry, MemoryKey, MemoryProposal, MemoryWorkspaceSettings
from gateway.models.tenancy import User as TenancyUser
from gateway.repositories.memory import MemoryRepositories
from gateway.repositories.tenancy import OrganizationRepository, WorkspaceRepository
from gateway.services.memory import EntryDraft, OpenedEntry, ProposalDraft, Provenance, WorkspaceKeys
from gateway.services.memory._store import EntryStore
from gateway.services.secret_box import generate_secret_key

pytestmark = pytest.mark.asyncio

SECRET = "the staging database password rotates on Fridays"
FENCED_TABLES = ("memory_keys", "memory_entries", "memory_proposals", "memory_signals", "memory_reads")


@dataclass(frozen=True)
class Tenants:
    first: uuid.UUID
    second: uuid.UUID
    alice: uuid.UUID
    bob: uuid.UUID


@pytest_asyncio.fixture
async def tenants(async_db: AsyncSession) -> Tenants:
    """Two workspaces of one organization and two members, committed so the service's own sessions see them."""
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme-memory", created_by_user_id=None
    )
    workspaces = WorkspaceRepository(async_db)
    first = await workspaces.create_workspace(name="First", organization_id=organization.id, created_by_user_id=None)
    second = await workspaces.create_workspace(name="Second", organization_id=organization.id, created_by_user_id=None)
    alice = TenancyUser(email="alice@example.test", active_organization_id=organization.id)
    bob = TenancyUser(email="bob@example.test", active_organization_id=organization.id)
    async_db.add_all([alice, bob])
    await async_db.flush()
    for workspace_id in (first.id, second.id):
        async_db.add(MemoryWorkspaceSettings(workspace_id=workspace_id, enabled=True, workspace_scope_enabled=True))
    await async_db.commit()
    return Tenants(first.id, second.id, alice.id, bob.id)


class _Store:
    """The entry store on one Unit of Work, each call its own block, as a service would run it."""

    def __init__(self, uow: UnitOfWork, keys: WorkspaceKeys) -> None:
        self._uow = uow
        self._store = EntryStore(MemoryRepositories.on(uow), keys)

    async def keep(self, draft: EntryDraft) -> uuid.UUID:
        async with self._uow:
            return await self._store.write_entry(draft)

    async def propose(self, draft: ProposalDraft) -> uuid.UUID:
        async with self._uow:
            return await self._store.write_proposal(draft)

    async def recallable(self, workspace_id: uuid.UUID, member_id: uuid.UUID) -> list[OpenedEntry]:
        async with self._uow:
            return await self._store.recallable(
                workspace_id, member_id=member_id, include_workspace=True, include_external=True
            )


@pytest_asyncio.fixture
async def memory(
    test_config: GatewayConfig, clean_database: None, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Callable[[], _Store]]:
    """A factory of stores that share one process's keys, each on a Unit of Work of its own."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    init_db(test_config)
    keys = WorkspaceKeys(
        build_content_key_port(test_config), rotation=timedelta(days=test_config.memory_key_rotation_days)
    )
    try:
        async with create_unit_of_work() as uow:
            yield lambda: _Store(uow, keys)
    finally:
        keys.clear()
        await dispose_db()


def _entry(
    workspace_id: uuid.UUID,
    *,
    text_: str = SECRET,
    scope: str = "workspace",
    owner: uuid.UUID | None = None,
    author: uuid.UUID | None = None,
    trust: str = "agent",
) -> EntryDraft:
    return EntryDraft(
        workspace_id=workspace_id,
        scope=scope,  # type: ignore[arg-type]
        owner_user_id=owner,
        kind="fact",
        trust=trust,  # type: ignore[arg-type]
        text=text_,
        embedding=[0.5, -0.25, 1.0],
        embedding_model="test:embed",
        confidence=0.5,
        provenance=Provenance(author_user_id=author, api_key_id="key-1", session_id="session-1", model="test:model"),
    )


async def _dump(async_db: AsyncSession) -> str:
    """Every memory table's rows, as one text, the way a database dump would show them."""
    rows: list[str] = []
    for table in (*FENCED_TABLES, "memory_workspace_settings"):
        result = await async_db.execute(text(f"SELECT row_to_json(t)::text FROM {table} t"))
        rows.extend(result.scalars().all())
    return "\n".join(rows)


async def test_an_entry_is_stored_sealed_and_recalled_opened(
    memory: Callable[[], _Store], tenants: Tenants, async_db: AsyncSession
) -> None:
    entry_id = await memory().keep(_entry(tenants.first, author=tenants.alice))

    dump = await _dump(async_db)
    assert "staging database password" not in dump and "Fridays" not in dump
    row = (await async_db.execute(select(MemoryEntry).where(MemoryEntry.id == entry_id))).scalar_one()
    assert row.content_sealed.startswith("mem1.") and row.embedding_sealed is not None

    [recalled] = await memory().recallable(tenants.first, tenants.alice)
    assert recalled.id == entry_id and recalled.text == SECRET
    assert recalled.embedding == (0.5, -0.25, 1.0)
    assert recalled.author_user_id == tenants.alice


async def test_a_proposal_is_stored_sealed(
    memory: Callable[[], _Store], tenants: Tenants, async_db: AsyncSession
) -> None:
    draft = ProposalDraft(
        workspace_id=tenants.first,
        scope="workspace",
        owner_user_id=None,
        kind="decision",
        trust="external",
        text=SECRET,
        provenance=Provenance(author_user_id=tenants.alice),
    )
    proposal_id = await memory().propose(draft)

    row = (await async_db.execute(select(MemoryProposal).where(MemoryProposal.id == proposal_id))).scalar_one()
    assert row.state == "pending" and "Fridays" not in row.content_sealed
    assert "Fridays" not in await _dump(async_db)


async def test_a_member_recalls_their_own_entries_and_the_workspaces_never_anothers(
    memory: Callable[[], _Store], tenants: Tenants
) -> None:
    shared = await memory().keep(_entry(tenants.first, text_="shared"))
    alices = await memory().keep(_entry(tenants.first, text_="alice's", scope="member", owner=tenants.alice))
    bobs = await memory().keep(_entry(tenants.first, text_="bob's", scope="member", owner=tenants.bob))

    assert {e.id for e in await memory().recallable(tenants.first, tenants.alice)} == {shared, alices}
    assert {e.id for e in await memory().recallable(tenants.first, tenants.bob)} == {shared, bobs}


async def test_a_workspace_recalls_nothing_of_another(memory: Callable[[], _Store], tenants: Tenants) -> None:
    await memory().keep(_entry(tenants.first))

    assert await memory().recallable(tenants.second, tenants.alice) == []


async def test_a_sealed_value_moved_onto_another_workspaces_row_does_not_open(
    memory: Callable[[], _Store], tenants: Tenants, async_db: AsyncSession
) -> None:
    source = await memory().keep(_entry(tenants.first))
    target = await memory().keep(_entry(tenants.second, text_="harmless"))
    sealed = (await async_db.execute(select(MemoryEntry.content_sealed).where(MemoryEntry.id == source))).scalar_one()
    await async_db.execute(update(MemoryEntry).where(MemoryEntry.id == target).values(content_sealed=sealed))
    await async_db.commit()

    assert await memory().recallable(tenants.second, tenants.alice) == []


async def test_without_a_secret_key_nothing_is_stored(
    memory: Callable[[], _Store], tenants: Tenants, async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OTARI_SECRET_KEY")

    with pytest.raises(MemorySealingUnavailableError):
        await memory().keep(_entry(tenants.first))

    assert (await async_db.execute(select(MemoryEntry))).first() is None
    assert (await async_db.execute(select(MemoryKey))).first() is None


async def test_the_fenced_tables_are_under_forced_row_level_security(async_db: AsyncSession) -> None:
    result = await async_db.execute(
        text("SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class WHERE relname = ANY(:tables)"),
        {"tables": list(FENCED_TABLES)},
    )

    assert {row.relname: (row.relrowsecurity, row.relforcerowsecurity) for row in result} == {
        table: (True, True) for table in FENCED_TABLES
    }


async def test_a_role_held_to_the_policy_reads_only_the_workspace_it_names(
    memory: Callable[[], _Store], tenants: Tenants, async_db: AsyncSession
) -> None:
    await memory().keep(_entry(tenants.first))
    await memory().keep(_entry(tenants.second))
    # The suite connects as a superuser, which no policy binds; a plain role is held to it.
    await async_db.execute(
        text(
            "DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'memory_policy_probe') "
            "THEN CREATE ROLE memory_policy_probe NOLOGIN; END IF; END $$"
        )
    )
    await async_db.execute(text("GRANT SELECT ON memory_entries TO memory_policy_probe"))
    await async_db.commit()

    async def visible(workspace_id: uuid.UUID | None) -> set[uuid.UUID]:
        async with async_db.begin():
            await async_db.execute(text("SET LOCAL ROLE memory_policy_probe"))
            if workspace_id is not None:
                await async_db.execute(
                    text("SELECT set_config('otari.memory_workspace', :w, true)"), {"w": str(workspace_id)}
                )
            rows = await async_db.execute(text("SELECT workspace_id FROM memory_entries"))
            return {row.workspace_id for row in rows}

    await async_db.commit()
    assert await visible(None) == set()
    assert await visible(tenants.first) == {tenants.first}
    assert await visible(tenants.second) == {tenants.second}


async def test_deleting_a_workspace_deletes_its_memory(
    memory: Callable[[], _Store], tenants: Tenants, async_db: AsyncSession
) -> None:
    await memory().keep(_entry(tenants.first))
    await memory().keep(_entry(tenants.second))

    await async_db.execute(text("DELETE FROM workspace WHERE id = :w"), {"w": tenants.first})
    await async_db.commit()

    remaining = (await async_db.execute(select(MemoryEntry.workspace_id))).scalars().all()
    assert remaining == [tenants.second]
    keys = (await async_db.execute(select(MemoryKey.workspace_id))).scalars().all()
    assert keys == [tenants.second]


async def test_deleting_a_member_deletes_their_entries_and_keeps_what_they_shared(
    memory: Callable[[], _Store], tenants: Tenants, async_db: AsyncSession
) -> None:
    shared = await memory().keep(_entry(tenants.first, author=tenants.alice))
    await memory().keep(_entry(tenants.first, scope="member", owner=tenants.alice, author=tenants.alice))

    await async_db.execute(text('DELETE FROM "user" WHERE id = :u'), {"u": tenants.alice})
    await async_db.commit()

    rows = (await async_db.execute(select(MemoryEntry))).scalars().all()
    assert [(row.id, row.author_user_id) for row in rows] == [(shared, None)]


async def test_the_policy_binds_the_tables_owner_and_cascades_still_delete(
    memory: Callable[[], _Store], tenants: Tenants, async_db: AsyncSession
) -> None:
    """As on RDS, where the app owns its tables: forced security holds it, and deletes still cascade."""
    await memory().keep(_entry(tenants.first, scope="member", owner=tenants.alice))
    await memory().keep(_entry(tenants.second))
    await async_db.execute(
        text(
            "DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'memory_owner_probe') "
            "THEN CREATE ROLE memory_owner_probe NOLOGIN; END IF; END $$"
        )
    )
    await async_db.commit()

    # One transaction, rolled back at the end, so the ownership it hands out ends with the test.
    async with async_db.begin():
        for table in FENCED_TABLES:
            await async_db.execute(text(f"ALTER TABLE {table} OWNER TO memory_owner_probe"))
        await async_db.execute(text('GRANT SELECT, DELETE ON workspace, "user" TO memory_owner_probe'))
        await async_db.execute(text("SET LOCAL ROLE memory_owner_probe"))

        async def count(table: str) -> int:
            return int((await async_db.execute(text(f"SELECT count(*) FROM {table}"))).scalar_one())

        assert await count("memory_entries") == 0
        await async_db.execute(text("SELECT set_config('otari.memory_workspace', :w, true)"), {"w": str(tenants.first)})
        assert await count("memory_entries") == 1

        await async_db.execute(text('DELETE FROM "user" WHERE id = :u'), {"u": tenants.alice})
        await async_db.execute(text("DELETE FROM workspace WHERE id = :w"), {"w": tenants.second})
        await async_db.execute(text("RESET ROLE"))
        remaining = await count("memory_entries")
        keys_left = await count("memory_keys")
        # Alice's member entry went with her, and the second workspace's entry and key with it.
        assert (remaining, keys_left) == (0, 1)
        await async_db.rollback()
