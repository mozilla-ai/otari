"""The entries' embeddings automatic recall has opened, held so the next turn does not read and open them again.

Recall compares the turn with every entry a member may recall, on each turn. Opening each entry's embedding is
most of a cold turn's cost, and an embedding never changes once stored: updating an entry stores a new version
under a new id. So what was opened is held, by workspace and entry id, and only entries not held yet are read
and opened.

Each workspace's embeddings are held as one matrix, each row scaled to length one, so ranking a turn is one
matrix product. A matrix is never changed in place: keeping or dropping rows makes a new one, so a ranking
running on a worker thread reads a consistent :class:`HeldView` while the event loop keeps more.

What is held is never what decides what is recalled. Each turn still lists, from the database and under its
row-level security, the entries the member may recall, and only those are looked up here: an entry retired,
flagged, deleted or erased is never handed over from what is held.

Plaintext lives in this process longer than a request, so it is bounded: an embedding is dropped
``ttl_seconds`` after it was opened, however often it is used, and the least recently used workspaces are
dropped past ``max_vectors``. Erasing a workspace drops what is held of it here at once; on other processes it
lasts until it expires. Embeddings only, never an entry's text.
"""

import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from time import monotonic

import numpy as np
import numpy.typing as npt

TTL_SECONDS = 600.0
MAX_VECTORS = 20_000

Matrix = npt.NDArray[np.float32]


def unit_rows(vectors: Sequence[Sequence[float] | npt.NDArray[np.float32]]) -> Matrix:
    """``vectors`` as the rows of a float32 matrix, each scaled to length one; an all-zero row stays zero."""
    matrix = np.array(vectors, dtype=np.float32, ndmin=2)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.divide(matrix, norms, out=matrix, where=norms > 0)
    return matrix


@dataclass(frozen=True)
class HeldView:
    """Some of a workspace's held embeddings: ``matrix`` rows ``rows`` are the embeddings of ``ids``, in order."""

    matrix: Matrix
    rows: npt.NDArray[np.intp]
    ids: list[uuid.UUID]


class _Workspace:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self.matrix: Matrix = np.empty((0, dimensions), dtype=np.float32)
        self.opened_at: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        # Live rows only; a row whose id is gone is dead until the matrix is next compacted.
        self.rows: dict[uuid.UUID, int] = {}
        self.oldest = float("inf")


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
        # Least recently used workspace first.
        self._held: OrderedDict[uuid.UUID, _Workspace] = OrderedDict()
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def missing(self, workspace_id: uuid.UUID, entry_ids: Iterable[uuid.UUID]) -> list[uuid.UUID]:
        """Those of ``entry_ids`` not held for the workspace, dropping first what of it has expired."""
        held = self._fresh(workspace_id)
        if held is None:
            return list(entry_ids)
        return [entry_id for entry_id in entry_ids if entry_id not in held.rows]

    def view(self, workspace_id: uuid.UUID, entry_ids: Iterable[uuid.UUID]) -> HeldView | None:
        """Those of ``entry_ids`` held for the workspace, in order; None when it holds none of them."""
        held = self._fresh(workspace_id)
        if held is None:
            return None
        ids = [entry_id for entry_id in entry_ids if entry_id in held.rows]
        if not ids:
            return None
        rows = np.fromiter((held.rows[entry_id] for entry_id in ids), dtype=np.intp, count=len(ids))
        return HeldView(held.matrix, rows, ids)

    def keep(
        self, workspace_id: uuid.UUID, opened: Mapping[uuid.UUID, Sequence[float] | npt.NDArray[np.float32]]
    ) -> None:
        """Hold embeddings just opened for the workspace, dropping the least recently used workspaces past the cap.

        An embedding of another size than those held, as after the deployment changed embedding model, replaces
        what the workspace held.
        """
        held = self._fresh(workspace_id)
        new = {entry_id: vector for entry_id, vector in opened.items() if held is None or entry_id not in held.rows}
        if not new:
            return
        dimensions = len(next(iter(new.values())))
        new = {entry_id: vector for entry_id, vector in new.items() if len(vector) == dimensions}
        if held is None or held.dimensions != dimensions:
            self.forget(workspace_id)
            held = self._held[workspace_id] = _Workspace(dimensions)
        self._held.move_to_end(workspace_id)
        if len(held.rows) * 2 < len(held.matrix):
            self._compact(held)
        now = self._clock()
        start = len(held.matrix)
        held.matrix = np.concatenate([held.matrix, unit_rows(list(new.values()))])
        held.opened_at = np.concatenate([held.opened_at, np.full(len(new), now)])
        held.rows.update({entry_id: start + offset for offset, entry_id in enumerate(new)})
        held.oldest = min(held.oldest, now)
        self._count += len(new)
        while self._count > self._max and len(self._held) > 1:
            _, dropped = self._held.popitem(last=False)
            self._count -= len(dropped.rows)

    def forget(self, workspace_id: uuid.UUID) -> None:
        """Drop everything held of the workspace, as its memory is erased."""
        dropped = self._held.pop(workspace_id, None)
        if dropped is not None:
            self._count -= len(dropped.rows)

    def _fresh(self, workspace_id: uuid.UUID) -> _Workspace | None:
        """The workspace's held embeddings, marked as just used, with what expired dropped."""
        held = self._held.get(workspace_id)
        if held is None:
            return None
        self._held.move_to_end(workspace_id)
        now = self._clock()
        if now - held.oldest >= self._ttl:
            expired = [entry_id for entry_id, row in held.rows.items() if now - held.opened_at[row] >= self._ttl]
            for entry_id in expired:
                del held.rows[entry_id]
            self._count -= len(expired)
            live = np.fromiter(held.rows.values(), dtype=np.intp, count=len(held.rows))
            held.oldest = float(held.opened_at[live].min()) if len(live) else float("inf")
        return held

    @staticmethod
    def _compact(held: _Workspace) -> None:
        """Keep only the live rows, in a new matrix."""
        order = list(held.rows.items())
        rows = np.fromiter((row for _, row in order), dtype=np.intp, count=len(order))
        held.matrix = held.matrix[rows]
        held.opened_at = held.opened_at[rows]
        held.rows = {entry_id: position for position, (entry_id, _) in enumerate(order)}
