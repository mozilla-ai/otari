"""A workspace's memory data keys: made when needed, rotated by age, opened once and held."""

import os
import uuid
from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from gateway.exceptions.memory_exceptions import MemorySealingUnavailableError
from gateway.models.memory import MemoryKey
from gateway.ports.content_key_port import ContentKeyError, DataKey
from gateway.repositories.memory import KeyRepository
from gateway.services.memory import WorkspaceKeys

pytestmark = pytest.mark.asyncio

T0 = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


@dataclass
class _Port:
    """A key source that wraps a key by prefixing it with its context, and counts its calls."""

    wrapper: str | None = "sk"
    fail_with: ContentKeyError | None = None
    made: int = 0
    opened: int = 0

    async def new_data_key(self, context: dict[str, str]) -> DataKey:
        if self.fail_with is not None:
            raise self.fail_with
        self.made += 1
        plaintext = os.urandom(32)
        return DataKey(wrapper="sk", plaintext=plaintext, wrapped=repr(sorted(context.items())).encode() + plaintext)

    async def open_data_key(self, wrapper: str, wrapped: bytes, context: dict[str, str]) -> bytes:
        if self.fail_with is not None:
            raise self.fail_with
        self.opened += 1
        prefix = repr(sorted(context.items())).encode()
        if not wrapped.startswith(prefix):
            raise ContentKeyError("rejected", "made for another context")
        return wrapped[len(prefix) :]

    def close(self) -> None:
        pass


@dataclass
class _Keys:
    """An in-memory ``KeyRepository``."""

    now: datetime = T0
    rows: list[MemoryKey] = field(default_factory=list)

    async def newest(self, workspace_id: uuid.UUID) -> MemoryKey | None:
        mine = [row for row in self.rows if row.workspace_id == workspace_id]
        return max(mine, key=lambda row: row.created_at, default=None)

    async def add(self, workspace_id: uuid.UUID, *, wrapper: str, wrapped: str) -> MemoryKey:
        row = MemoryKey(
            id=uuid.uuid4(), workspace_id=workspace_id, wrapper=wrapper, wrapped=wrapped, created_at=self.now
        )
        self.rows.append(row)
        return row

    async def by_ids(self, workspace_id: uuid.UUID, key_ids: Collection[uuid.UUID]) -> list[MemoryKey]:
        return [row for row in self.rows if row.workspace_id == workspace_id and row.id in key_ids]


def _keys(port: _Port | None, repo: _Keys, *, now: list[datetime]) -> WorkspaceKeys:
    return WorkspaceKeys(port, rotation=timedelta(days=30), now=lambda: now[0])


async def test_a_workspace_gets_a_key_on_its_first_seal_and_keeps_it() -> None:
    port, repo, now = _Port(), _Keys(), [T0]
    keys = _keys(port, repo, now=now)
    workspace = uuid.uuid4()

    first = await keys.for_sealing(cast(KeyRepository, repo), workspace)
    again = await keys.for_sealing(cast(KeyRepository, repo), workspace)

    assert first.id == again.id and first.plaintext == again.plaintext
    assert port.made == 1 and len(repo.rows) == 1


async def test_a_key_older_than_the_rotation_is_replaced_and_still_opens() -> None:
    port, repo, now = _Port(), _Keys(), [T0]
    keys = _keys(port, repo, now=now)
    workspace = uuid.uuid4()
    old = await keys.for_sealing(cast(KeyRepository, repo), workspace)

    now[0] = repo.now = T0 + timedelta(days=31)
    new = await keys.for_sealing(cast(KeyRepository, repo), workspace)

    assert new.id != old.id
    keys.clear()
    opened = await keys.for_opening(cast(KeyRepository, repo), workspace, {old.id, new.id})
    assert opened == {old.id: old.plaintext, new.id: new.plaintext}


async def test_an_opened_key_is_held_rather_than_asked_for_again() -> None:
    port, repo, now = _Port(), _Keys(), [T0]
    keys = _keys(port, repo, now=now)
    workspace = uuid.uuid4()
    made = await keys.for_sealing(cast(KeyRepository, repo), workspace)
    keys.clear()

    for _ in range(3):
        await keys.for_opening(cast(KeyRepository, repo), workspace, {made.id})

    assert port.opened == 1


async def test_another_workspaces_key_does_not_open_for_this_one() -> None:
    port, repo, now = _Port(), _Keys(), [T0]
    keys = _keys(port, repo, now=now)
    theirs = await keys.for_sealing(cast(KeyRepository, repo), uuid.uuid4())
    keys.clear()

    assert await keys.for_opening(cast(KeyRepository, repo), uuid.uuid4(), {theirs.id}) == {}


async def test_without_a_key_source_nothing_is_sealed() -> None:
    keys = _keys(None, _Keys(), now=[T0])

    assert not keys.sealing
    with pytest.raises(MemorySealingUnavailableError):
        await keys.for_sealing(cast(KeyRepository, _Keys()), uuid.uuid4())


@pytest.mark.parametrize("reason", ["unavailable", "denied", "rejected"])
async def test_a_key_source_that_fails_seals_nothing(reason: str) -> None:
    port = _Port(fail_with=ContentKeyError(reason, "no"))  # type: ignore[arg-type]
    keys = _keys(port, _Keys(), now=[T0])

    with pytest.raises(MemorySealingUnavailableError):
        await keys.for_sealing(cast(KeyRepository, _Keys()), uuid.uuid4())


async def test_a_key_that_no_longer_opens_is_left_out() -> None:
    port, repo, now = _Port(), _Keys(), [T0]
    keys = _keys(port, repo, now=now)
    workspace = uuid.uuid4()
    made = await keys.for_sealing(cast(KeyRepository, repo), workspace)
    keys.clear()
    port.fail_with = ContentKeyError("denied", "revoked")

    assert await keys.for_opening(cast(KeyRepository, repo), workspace, {made.id}) == {}
