"""The entries' embeddings automatic recall has opened, held so the next turn does not read and open them again.

Recall compares the turn with every entry a member may recall, on each turn. Opening each entry's embedding is
most of that turn's cost, and an embedding never changes once stored: updating an entry stores a new version
under a new id. So what was opened is held, by workspace and entry id, and only entries not held yet are read
and opened.

What is held is never what decides what is recalled. Each turn still lists, from the database and under its
row-level security, the entries the member may recall, and only those are looked up here: an entry retired,
flagged, deleted or erased is never handed over from what is held.

Plaintext lives in this process longer than a request, so it is bounded: an embedding is dropped
``ttl_seconds`` after it was opened, however often it is used, and the least recently used workspaces are
dropped past ``max_vectors``. Erasing a workspace drops what is held of it here at once; on other processes it
lasts until it expires. Embeddings only, never an entry's text.
"""

import uuid
from array import array
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from time import monotonic

TTL_SECONDS = 600.0
MAX_VECTORS = 20_000


class RecallVectors:
    """Opened embeddings by workspace and entry id, each for ``ttl_seconds``. One per process."""

    def __init__(
        self,
        *,
        ttl_seconds: float = TTL_SECONDS,
        max_vectors: int = MAX_VECTORS,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._max = max_vectors
        self._clock = clock
        # Least recently used workspace first; each maps an entry id to its embedding and when it was opened.
        self._held: OrderedDict[uuid.UUID, dict[uuid.UUID, tuple[array[float], float]]] = OrderedDict()
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def take(self, workspace_id: uuid.UUID, entry_ids: Iterable[uuid.UUID]) -> dict[uuid.UUID, array[float]]:
        """Those of ``entry_ids`` held for the workspace, dropping first what of it has expired."""
        held = self._held.get(workspace_id)
        if held is None:
            return {}
        self._held.move_to_end(workspace_id)
        expired = [entry_id for entry_id, (_, opened_at) in held.items() if self._clock() - opened_at >= self._ttl]
        for entry_id in expired:
            del held[entry_id]
        self._count -= len(expired)
        return {entry_id: held[entry_id][0] for entry_id in entry_ids if entry_id in held}

    def keep(self, workspace_id: uuid.UUID, opened: Mapping[uuid.UUID, Sequence[float]]) -> None:
        """Hold embeddings just opened for the workspace, dropping the least recently used workspaces past the cap."""
        if not opened:
            return
        held = self._held.setdefault(workspace_id, {})
        self._held.move_to_end(workspace_id)
        now = self._clock()
        for entry_id, vector in opened.items():
            if entry_id not in held:
                self._count += 1
            held[entry_id] = (array("f", vector), now)
        while self._count > self._max and len(self._held) > 1:
            _, dropped = self._held.popitem(last=False)
            self._count -= len(dropped)

    def forget(self, workspace_id: uuid.UUID) -> None:
        """Drop everything held of the workspace, as its memory is erased."""
        self._count -= len(self._held.pop(workspace_id, {}))
