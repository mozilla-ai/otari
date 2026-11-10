"""Each workspace's memory data keys: made and opened through :class:`ContentKeyPort`, held a few minutes.

A workspace seals new rows under its newest key until that key is older than
``memory_key_rotation_days``, then makes another. A search opens every row of a
workspace, so a workspace has few keys rather than one per row, and an opened
key is held for ``KEY_MAX_AGE_S``: a busy workspace costs its key source one
call every few minutes, not one a row.

Opened keys are dropped once they reach their age, and when the process stops.
Dropping a reference cannot make Python overwrite the bytes, which stay in
memory until reused.
"""

import asyncio
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from time import monotonic

from gateway.exceptions.memory_exceptions import MemorySealingUnavailableError
from gateway.log_config import logger
from gateway.metrics import REGISTRY, Counter
from gateway.models.memory import MemoryKey
from gateway.ports.content_key_port import ContentKeyError, ContentKeyPort
from gateway.repositories.memory import KeyRepository
from gateway.services.memory._sealing import SealedValueError, decode_wrapped, encode_wrapped, key_context

KEY_MAX_AGE_S = 300.0
KEY_CACHE_SIZE = 4_096
# Longest one call to the key source may take, so a slow one fails a step rather than holding it.
KEY_SOURCE_BUDGET_S = 5.0

UNOPENED_KEYS = Counter(
    "gateway_memory_keys_unopened",
    "Memory data keys that could not be opened, so their rows were left out, by reason",
    ["reason"],
    registry=REGISTRY,
)


@dataclass(frozen=True, repr=False)
class OpenKey:
    """A workspace data key in plaintext, and the id rows sealed under it carry. Its ``repr`` names only the id."""

    id: uuid.UUID
    plaintext: bytes

    def __repr__(self) -> str:
        return f"OpenKey(id={self.id})"


class WorkspaceKeys:
    """Makes, opens and holds the workspaces' data keys. One per process, so its cache serves every request."""

    def __init__(
        self,
        port: ContentKeyPort | None,
        *,
        rotation: timedelta,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._port = port
        self._rotation = rotation
        self._now = now
        self._opened: OrderedDict[uuid.UUID, tuple[bytes, float]] = OrderedDict()

    @property
    def sealing(self) -> bool:
        """Whether a key source to make new keys with is configured."""
        return self._port is not None and self._port.wrapper is not None

    def clear(self) -> None:
        """Drop every opened key, as the process stops."""
        self._opened.clear()

    async def for_sealing(self, keys: KeyRepository, workspace_id: uuid.UUID) -> OpenKey:
        """The key new rows of the workspace are sealed under, made first when it has none young enough.

        Raises:
            MemorySealingUnavailableError: no key source is configured, or it could not make or open the key.
        """
        newest = await keys.newest(workspace_id)
        if newest is not None and self._now() - newest.created_at < self._rotation:
            return OpenKey(newest.id, await self._open_or_refuse(newest))
        if self._port is None or self._port.wrapper is None:
            raise MemorySealingUnavailableError
        try:
            made = await _within_budget(self._port.new_data_key(key_context(workspace_id)))
        except ContentKeyError as exc:
            logger.warning("A memory data key could not be made (%s)", exc.reason)
            raise MemorySealingUnavailableError from None
        row = await keys.add(workspace_id, wrapper=made.wrapper, wrapped=encode_wrapped(made.wrapped))
        self._hold(row.id, made.plaintext)
        return OpenKey(row.id, made.plaintext)

    async def for_opening(
        self, keys: KeyRepository, workspace_id: uuid.UUID, key_ids: Collection[uuid.UUID]
    ) -> dict[uuid.UUID, bytes]:
        """The workspace's keys among ``key_ids`` that open; a key that does not is left out, and counted."""
        self._prune()
        opened: dict[uuid.UUID, bytes] = {}
        missing: list[uuid.UUID] = []
        for key_id in set(key_ids):
            held = self._opened.get(key_id)
            if held is None:
                missing.append(key_id)
            else:
                self._opened.move_to_end(key_id)
                opened[key_id] = held[0]
        if missing:
            # Fetched by the workspace, so an id of another workspace's key finds no row to open.
            for row in await keys.by_ids(workspace_id, missing):
                plaintext = await self._open(row)
                if plaintext is not None:
                    opened[row.id] = plaintext
        return opened

    async def _open_or_refuse(self, row: MemoryKey) -> bytes:
        plaintext = await self._open(row)
        if plaintext is None:
            raise MemorySealingUnavailableError
        return plaintext

    async def _open(self, row: MemoryKey) -> bytes | None:
        """The key's plaintext, from the cache or the key source; None when it does not open."""
        held = self._opened.get(row.id)
        if held is not None and monotonic() - held[1] < KEY_MAX_AGE_S:
            return held[0]
        if self._port is None:
            UNOPENED_KEYS.labels(reason="unavailable").inc()
            return None
        try:
            wrapped = decode_wrapped(row.wrapped)
            plaintext = await _within_budget(
                self._port.open_data_key(row.wrapper, wrapped, key_context(row.workspace_id))
            )
        except SealedValueError:
            UNOPENED_KEYS.labels(reason="rejected").inc()
            return None
        except ContentKeyError as exc:
            UNOPENED_KEYS.labels(reason=exc.reason).inc()
            logger.warning("A memory data key could not be opened (%s)", exc.reason)
            return None
        self._hold(row.id, plaintext)
        return plaintext

    def _hold(self, key_id: uuid.UUID, plaintext: bytes) -> None:
        self._opened[key_id] = (plaintext, monotonic())
        self._opened.move_to_end(key_id)
        while len(self._opened) > KEY_CACHE_SIZE:
            self._opened.popitem(last=False)

    def _prune(self) -> None:
        now = monotonic()
        for key_id in [key_id for key_id, (_, at) in self._opened.items() if now - at >= KEY_MAX_AGE_S]:
            del self._opened[key_id]


async def _within_budget[T](call: Awaitable[T]) -> T:
    """A key source's answer, or ``unavailable`` once it has taken longer than a step can spare."""
    try:
        return await asyncio.wait_for(call, timeout=KEY_SOURCE_BUDGET_S)
    except TimeoutError:
        raise ContentKeyError("unavailable", "the content key source did not answer in time") from None
