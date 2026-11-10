"""Agent memory's use cases: propose, consolidate, search, get, confirm, flag and forget."""

import hashlib
import math
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.adapters.content_key_adapter import build_content_key_port
from gateway.core.config import GatewayConfig
from gateway.core.database import dispose_db, init_db
from gateway.core.unit_of_work import create_unit_of_work
from gateway.exceptions.memory_exceptions import (
    MemoryEntryNotFoundError,
    MemoryModelUnavailableError,
    MemoryOffError,
    MemoryRefusedError,
    MemoryTooLongError,
    NotYoursToForgetError,
    WorkspaceScopeOffError,
)
from gateway.models.memory import MemoryEntry, MemoryProposal, MemoryRead, MemoryWorkspaceSettings
from gateway.models.tenancy import User as TenancyUser
from gateway.repositories.memory import MemoryRepositories
from gateway.repositories.tenancy import OrganizationRepository, WorkspaceRepository
from gateway.services.memory import MemoryCaller, MemoryService, WorkspaceKeys, consolidate_once
from gateway.services.memory._models import Decision, MemoryModelError
from gateway.services.secret_box import generate_secret_key

pytestmark = pytest.mark.asyncio

DEPLOY = "Deploy the api service with make deploy-api from the main branch only"


def _vector(text: str) -> list[float]:
    """A bag of words hashed into 64 dimensions: the same words make the same vector."""
    vector = [0.0] * 64
    for word in text.lower().split():
        vector[int(hashlib.sha256(word.encode()).hexdigest(), 16) % 64] += 1.0
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


@dataclass
class _Models:
    """Stands in for the deployment's memory models."""

    decisions: list[Decision] = field(default_factory=list)
    fail: bool = False
    embedded: int = 0
    embedding_model: str = "test:embed"

    @property
    def can_decide(self) -> bool:
        return bool(self.decisions)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self.fail:
            raise MemoryModelError("down")
        self.embedded += len(texts)
        return [_vector(text) for text in texts]

    async def decide(self, proposal: str, existing: Sequence[str]) -> Decision:
        return self.decisions.pop(0)


@dataclass(frozen=True)
class Tenants:
    workspace: uuid.UUID
    other_workspace: uuid.UUID
    alice: MemoryCaller
    bob: MemoryCaller
    carol: MemoryCaller
    admin: MemoryCaller


@pytest_asyncio.fixture
async def tenants(async_db: AsyncSession) -> Tenants:
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme-memory-service", created_by_user_id=None
    )
    workspaces = WorkspaceRepository(async_db)
    workspace = await workspaces.create_workspace(name="One", organization_id=organization.id, created_by_user_id=None)
    other = await workspaces.create_workspace(name="Two", organization_id=organization.id, created_by_user_id=None)
    people = [TenancyUser(email=f"{name}@example.test", active_organization_id=organization.id) for name in "abcd"]
    async_db.add_all(people)
    await async_db.flush()
    for workspace_id in (workspace.id, other.id):
        async_db.add(
            MemoryWorkspaceSettings(
                workspace_id=workspace_id, enabled=True, workspace_scope_enabled=True, write_policy="auto"
            )
        )
    await async_db.commit()

    def caller(person: TenancyUser, *, manager: bool = False) -> MemoryCaller:
        return MemoryCaller(workspace.id, person.id, is_manager=manager, api_key_id=f"key-{person.email}")

    a, b, c, d = people
    return Tenants(workspace.id, other.id, caller(a), caller(b), caller(c), caller(d, manager=True))


@dataclass
class Memory:
    """Services on one Unit of Work and one process's keys, over models a test controls."""

    build: Callable[[], MemoryService]
    models: _Models
    configure: Callable[..., None]

    def __call__(self) -> MemoryService:
        return self.build()


@pytest_asyncio.fixture
async def memory(
    test_config: GatewayConfig, clean_database: None, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Memory]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    init_db(test_config)
    keys = WorkspaceKeys(build_content_key_port(test_config), rotation=timedelta(days=30))
    models = _Models()
    config = {"value": test_config}

    def configure(**changes: Any) -> None:
        config["value"] = config["value"].model_copy(update=changes)

    try:
        async with create_unit_of_work() as uow:
            yield Memory(
                lambda: MemoryService(uow, MemoryRepositories.on(uow), keys, models, config["value"]),  # type: ignore[arg-type]
                models,
                configure,
            )
    finally:
        keys.clear()
        await dispose_db()


async def _learn(memory: Memory, caller: MemoryCaller, text: str, **fields: Any) -> uuid.UUID:
    """Propose ``text`` and consolidate it; the id of the entry it became."""
    await memory().propose(
        caller,
        text=text,
        kind=fields.get("kind", "procedure"),
        scope=fields.get("scope", "workspace"),
        external=fields.get("external", False),
    )
    await memory().consolidate(caller.workspace_id, limit=20)
    [found] = [r for r in await memory().search(caller, text, limit=10) if r.entry.text == text]
    return found.entry.id


async def _switch(async_db: AsyncSession, workspace_id: uuid.UUID, **values: Any) -> None:
    settings = MemoryWorkspaceSettings
    await async_db.execute(update(settings).where(settings.workspace_id == workspace_id).values(**values))
    await async_db.commit()


async def test_what_one_agent_proposes_another_member_recalls(
    memory: Memory, tenants: Tenants, async_db: AsyncSession
) -> None:
    await memory().propose(tenants.alice, text=DEPLOY, kind="procedure", scope="workspace", external=False)
    assert await memory().search(tenants.bob, "how do I deploy the api", limit=5) == []

    assert await memory().consolidate(tenants.workspace, limit=20) == 1

    [found] = await memory().search(tenants.bob, "how do I deploy the api", limit=5)
    assert (
        found.entry.text == DEPLOY
        and found.entry.trust == "agent"
        and found.entry.author_user_id == tenants.alice.member_id
    )
    assert (await async_db.execute(select(MemoryProposal))).first() is None
    reads = (await async_db.execute(select(MemoryRead))).scalars().all()
    assert [(read.reader_user_id, read.tool) for read in reads] == [(tenants.bob.member_id, "memory_search")] * 2


async def test_the_same_thing_proposed_twice_is_kept_once(memory: Memory, tenants: Tenants) -> None:
    await _learn(memory, tenants.alice, DEPLOY)
    await memory().propose(tenants.bob, text=DEPLOY, kind="procedure", scope="workspace", external=False)
    await memory().consolidate(tenants.workspace, limit=20)

    assert len(await memory().search(tenants.alice, DEPLOY, limit=10)) == 1


async def test_an_update_writes_a_new_version_that_keeps_the_old_one(memory: Memory, tenants: Tenants) -> None:
    old = await _learn(memory, tenants.alice, DEPLOY)
    memory.models.decisions = [Decision("update", 0)]
    newer = DEPLOY + " and tag the release first"
    await memory().propose(tenants.bob, text=newer, kind="procedure", scope="workspace", external=False)
    await memory().consolidate(tenants.workspace, limit=20)

    [found] = await memory().search(tenants.alice, DEPLOY, limit=10)
    assert found.entry.text == newer and found.entry.version == 2
    history = await memory().get(tenants.alice, found.entry.id)
    assert [version.id for version in history.versions] == [old]


async def test_a_conflict_waits_for_an_admin(memory: Memory, tenants: Tenants, async_db: AsyncSession) -> None:
    await _learn(memory, tenants.alice, DEPLOY)
    memory.models.decisions = [Decision("conflict", 0)]
    await memory().propose(tenants.bob, text=DEPLOY + " never", kind="procedure", scope="workspace", external=False)
    await memory().consolidate(tenants.workspace, limit=20)

    proposal = (await async_db.execute(select(MemoryProposal))).scalar_one()
    assert (proposal.state, proposal.review_reason) == ("review", "conflict")
    assert len(await memory().search(tenants.alice, DEPLOY, limit=10)) == 1


async def test_an_agent_cannot_overwrite_a_reviewed_entry(
    memory: Memory, tenants: Tenants, async_db: AsyncSession
) -> None:
    reviewed = await _learn(memory, tenants.alice, DEPLOY)
    await async_db.execute(update(MemoryEntry).where(MemoryEntry.id == reviewed).values(trust="reviewed"))
    await async_db.commit()
    memory.models.decisions = [Decision("update", 0)]
    await memory().propose(tenants.bob, text=DEPLOY + " today", kind="procedure", scope="workspace", external=False)
    await memory().consolidate(tenants.workspace, limit=20)

    proposal = (await async_db.execute(select(MemoryProposal))).scalar_one()
    assert proposal.review_reason == "higher_trust"


async def test_under_a_review_policy_shared_proposals_wait_and_member_ones_do_not(
    memory: Memory, tenants: Tenants, async_db: AsyncSession
) -> None:
    await _switch(async_db, tenants.workspace, write_policy="review")
    await memory().propose(tenants.alice, text=DEPLOY, kind="procedure", scope="workspace", external=False)
    await memory().propose(tenants.alice, text="I prefer short answers", kind="fact", scope="member", external=False)
    await memory().consolidate(tenants.workspace, limit=20)

    proposal = (await async_db.execute(select(MemoryProposal))).scalar_one()
    assert (proposal.scope, proposal.review_reason) == ("workspace", "policy")
    assert [r.entry.scope for r in await memory().search(tenants.alice, "short answers", limit=5)] == ["member"]


async def test_a_member_entry_is_recalled_only_by_its_owner(memory: Memory, tenants: Tenants) -> None:
    mine = await _learn(memory, tenants.alice, "I prefer short answers", scope="member", kind="fact")

    assert await memory().search(tenants.bob, "I prefer short answers", limit=5) == []
    with pytest.raises(MemoryEntryNotFoundError):
        await memory().get(tenants.bob, mine)
    with pytest.raises(MemoryEntryNotFoundError):
        await memory().get(tenants.admin, mine)


async def test_another_workspace_recalls_nothing(memory: Memory, tenants: Tenants) -> None:
    entry = await _learn(memory, tenants.alice, DEPLOY)
    elsewhere = MemoryCaller(tenants.other_workspace, tenants.alice.member_id, is_manager=False)

    assert await memory().search(elsewhere, DEPLOY, limit=5) == []
    with pytest.raises(MemoryEntryNotFoundError):
        await memory().get(elsewhere, entry)


@pytest.mark.parametrize(
    "text", ["use sk-proj-abcdefghijklmnop1234 to deploy", "ask maria.lopez@example.com", "password=correcthorse"]
)
async def test_what_memory_does_not_keep_is_refused_before_anything_is_stored(
    memory: Memory, tenants: Tenants, async_db: AsyncSession, text: str
) -> None:
    with pytest.raises(MemoryRefusedError):
        await memory().propose(tenants.alice, text=text, kind="fact", scope="workspace", external=False)

    assert (await async_db.execute(select(MemoryProposal))).first() is None


async def test_a_proposal_longer_than_memory_keeps_is_refused(memory: Memory, tenants: Tenants) -> None:
    memory.configure(memory_entry_max_length=10)

    with pytest.raises(MemoryTooLongError):
        await memory().propose(tenants.alice, text=DEPLOY, kind="fact", scope="workspace", external=False)


async def test_memory_off_refuses_everything(memory: Memory, tenants: Tenants, async_db: AsyncSession) -> None:
    await _switch(async_db, tenants.workspace, enabled=False)

    with pytest.raises(MemoryOffError):
        await memory().search(tenants.alice, DEPLOY, limit=5)
    with pytest.raises(MemoryOffError):
        await memory().propose(tenants.alice, text=DEPLOY, kind="fact", scope="member", external=False)
    assert memory.models.embedded == 0


async def test_with_the_shared_memory_off_only_member_memory_is_kept_and_recalled(
    memory: Memory, tenants: Tenants, async_db: AsyncSession
) -> None:
    await _learn(memory, tenants.alice, DEPLOY)
    await _switch(async_db, tenants.workspace, workspace_scope_enabled=False)

    with pytest.raises(WorkspaceScopeOffError):
        await memory().propose(tenants.alice, text="Another", kind="fact", scope="workspace", external=False)
    assert await memory().search(tenants.alice, DEPLOY, limit=5) == []
    await _learn(memory, tenants.alice, "I prefer short answers", scope="member", kind="fact")


async def test_external_entries_can_be_kept_out_of_recall(
    memory: Memory, tenants: Tenants, async_db: AsyncSession
) -> None:
    entry = await _learn(memory, tenants.alice, DEPLOY, external=True)
    [found] = await memory().search(tenants.bob, DEPLOY, limit=5)
    assert found.entry.trust == "external"

    await _switch(async_db, tenants.workspace, hide_external=True)

    assert await memory().search(tenants.bob, DEPLOY, limit=5) == []
    with pytest.raises(MemoryEntryNotFoundError):
        await memory().get(tenants.bob, entry)


async def test_each_member_counts_once_and_enough_flags_take_an_entry_out_of_recall(
    memory: Memory, tenants: Tenants
) -> None:
    entry = await _learn(memory, tenants.alice, DEPLOY)

    assert (await memory().confirm(tenants.bob, entry)).confidence == pytest.approx(0.6)
    assert (await memory().confirm(tenants.bob, entry)).confidence == pytest.approx(0.6)
    assert (await memory().flag(tenants.bob, entry, "stale")).confidence == pytest.approx(0.35)
    await memory().flag(tenants.carol, entry, "wrong")
    receipt = await memory().flag(tenants.alice, entry, "wrong")

    assert receipt.status == "quarantined"
    assert await memory().search(tenants.bob, DEPLOY, limit=5) == []


async def test_a_more_trusted_entry_ranks_first(memory: Memory, tenants: Tenants) -> None:
    first = await _learn(memory, tenants.alice, DEPLOY)
    second = await _learn(memory, tenants.alice, DEPLOY.replace("main", "release"))
    await memory().confirm(tenants.bob, second)
    await memory().confirm(tenants.carol, second)

    ranked = [r.entry.id for r in await memory().search(tenants.bob, DEPLOY, limit=5)]
    assert ranked == [second, first]


async def test_only_the_author_or_a_manager_forgets_a_shared_entry_with_its_versions(
    memory: Memory, tenants: Tenants, async_db: AsyncSession
) -> None:
    await _learn(memory, tenants.alice, DEPLOY)
    memory.models.decisions = [Decision("update", 0)]
    await memory().propose(tenants.alice, text=DEPLOY + " now", kind="procedure", scope="workspace", external=False)
    await memory().consolidate(tenants.workspace, limit=20)
    [latest] = await memory().search(tenants.alice, DEPLOY, limit=5)

    with pytest.raises(NotYoursToForgetError):
        await memory().forget(tenants.bob, latest.entry.id)
    assert await memory().forget(tenants.admin, latest.entry.id) == 2
    assert (await async_db.execute(select(MemoryEntry))).first() is None


async def test_at_its_cap_a_workspace_retires_its_least_used_entry(memory: Memory, tenants: Tenants) -> None:
    memory.configure(memory_max_entries_per_workspace=1)
    await _learn(memory, tenants.alice, DEPLOY)
    await _learn(memory, tenants.alice, "Run make test before every push")

    assert [r.entry.text for r in await memory().search(tenants.alice, DEPLOY, limit=10)] == [
        "Run make test before every push"
    ]


async def test_without_the_embedding_model_a_proposal_waits_and_then_goes_to_an_admin(
    memory: Memory, tenants: Tenants, async_db: AsyncSession
) -> None:
    await memory().propose(tenants.alice, text=DEPLOY, kind="procedure", scope="workspace", external=False)
    memory.models.fail = True

    for _ in range(5):
        assert await memory().consolidate(tenants.workspace, limit=20) == 0
    with pytest.raises(MemoryModelUnavailableError):
        await memory().search(tenants.alice, DEPLOY, limit=5)

    proposal = (await async_db.execute(select(MemoryProposal))).scalar_one()
    assert (proposal.attempts, proposal.state, proposal.review_reason) == (5, "review", "model")


async def test_one_pass_consolidates_every_workspace_with_memory_on(memory: Memory, tenants: Tenants) -> None:
    elsewhere = MemoryCaller(tenants.other_workspace, tenants.alice.member_id, is_manager=False)
    await memory().propose(tenants.alice, text=DEPLOY, kind="procedure", scope="workspace", external=False)
    await memory().propose(elsewhere, text=DEPLOY, kind="procedure", scope="workspace", external=False)

    assert await consolidate_once(lambda uow: memory()) == 2
