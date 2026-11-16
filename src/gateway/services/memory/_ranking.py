"""Recall's ranking in this process: open the candidates' embeddings, compare them with the turn, best first.

The candidates' embeddings are compared with the turn as one matrix product, on a worker thread, so a workspace
with thousands of entries holds up no other request the process serves. The opened embeddings are kept in
``held`` where the process keeps any; otherwise they are opened for the turn and dropped after it.

A memory service runs this same ranking for the gateways that hand theirs off.
"""

import asyncio
import uuid
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from gateway.core.unit_of_work import UnitOfWork
from gateway.ports.memory_ranking_port import MemoryRankingPort, RankCandidate, RankedCandidate, RankOrder
from gateway.services.memory._recall_vectors import HeldView, Matrix, RecallVectors, unit_rows
from gateway.services.memory._store import EntryStore
from gateway.services.memory._timing import timed


class LocalMemoryRanking(MemoryRankingPort):
    """Ranks on the caller's Unit of Work, opening embeddings through its store."""

    def __init__(self, uow: UnitOfWork, store: EntryStore, held: RecallVectors | None) -> None:
        self._uow = uow
        self._store = store
        self._held = held

    async def rank(
        self,
        workspace_id: uuid.UUID,
        candidates: Sequence[RankCandidate],
        queries: Sequence[Sequence[float]],
        *,
        order: RankOrder,
        keep_top: int,
        keep_above: float | None = None,
    ) -> list[RankedCandidate]:
        if not candidates or not queries:
            return []
        ids = [candidate.id for candidate in candidates]
        with timed("open"):
            missing = self._held.missing(workspace_id, ids) if self._held is not None else ids
            opened = {}
            if missing:
                async with self._uow:
                    opened = await self._store.open_embeddings(workspace_id, missing)
        with timed("score"):
            if self._held is None:
                return await asyncio.to_thread(
                    rank_vectors, candidates, opened, queries, order=order, keep_top=keep_top, keep_above=keep_above
                )
            self._held.keep(workspace_id, opened)
            view = self._held.view(workspace_id, ids)
            return await asyncio.to_thread(
                rank_view, candidates, view, queries, order=order, keep_top=keep_top, keep_above=keep_above
            )

    async def forget(self, workspace_id: uuid.UUID) -> None:
        if self._held is not None:
            self._held.forget(workspace_id)


def rank_vectors(
    candidates: Sequence[RankCandidate],
    vectors: Mapping[uuid.UUID, Sequence[float] | npt.NDArray[np.float32]],
    queries: Sequence[Sequence[float]],
    *,
    order: RankOrder,
    keep_top: int,
    keep_above: float | None,
) -> list[RankedCandidate]:
    """What :meth:`LocalMemoryRanking.rank` returns, from embeddings opened for this turn. Pure, for a thread.

    A candidate with no vector in ``vectors`` is left out, and one of another size than the turn's is as unlike
    it as can be (similarity 0).
    """
    present = [candidate.id for candidate in candidates if candidate.id in vectors]
    if not present:
        return []
    sizes = {len(vectors[entry_id]) for entry_id in present}
    if len(sizes) == 1:
        view = HeldView(unit_rows([vectors[entry_id] for entry_id in present]), np.arange(len(present)), present)
        return rank_view(candidates, view, queries, order=order, keep_top=keep_top, keep_above=keep_above)
    # Embeddings of more than one size, as while a deployment changes embedding model: each size apart.
    similarity: dict[uuid.UUID, float] = {}
    for size in sizes:
        alike = [entry_id for entry_id in present if len(vectors[entry_id]) == size]
        matrix = unit_rows([vectors[entry_id] for entry_id in alike])
        similarity.update(zip(alike, _similarities(matrix, np.arange(len(alike)), queries).tolist(), strict=True))
    similarities = np.array([similarity[entry_id] for entry_id in present], dtype=np.float32)
    return _ordered(candidates, present, similarities, order=order, keep_top=keep_top, keep_above=keep_above)


def rank_view(
    candidates: Sequence[RankCandidate],
    view: HeldView | None,
    queries: Sequence[Sequence[float]],
    *,
    order: RankOrder,
    keep_top: int,
    keep_above: float | None,
) -> list[RankedCandidate]:
    """What :meth:`LocalMemoryRanking.rank` returns, from the embeddings a process holds. Pure, for a thread."""
    if view is None:
        return []
    similarities = _similarities(view.matrix, view.rows, queries)
    return _ordered(candidates, view.ids, similarities, order=order, keep_top=keep_top, keep_above=keep_above)


def _similarities(
    matrix: Matrix, rows: npt.NDArray[np.intp], queries: Sequence[Sequence[float]]
) -> npt.NDArray[np.float32]:
    """Each of ``rows``' best cosine similarity to any of ``queries``; a query of another size counts as 0."""
    dimensions = matrix.shape[1]
    fitting = [query for query in queries if len(query) == dimensions]
    if not fitting:
        return np.zeros(len(rows), dtype=np.float32)
    # The whole matrix by the queries, then the rows asked: no copy of the rows themselves.
    best = (matrix @ unit_rows(fitting).T)[rows].max(axis=1)
    if len(fitting) < len(queries):
        best = np.maximum(best, 0.0)
    return best


def _ordered(
    candidates: Sequence[RankCandidate],
    ids: Sequence[uuid.UUID],
    similarities: npt.NDArray[np.float32],
    *,
    order: RankOrder,
    keep_top: int,
    keep_above: float | None,
) -> list[RankedCandidate]:
    """``ids``, with their similarities, best first by ``order``: the first ``keep_top``, and any alike enough.

    ``ids`` are in the order of ``candidates``, and ties keep it. Only what is kept becomes a result.
    """
    confidence = {candidate.id: candidate.confidence for candidate in candidates}
    confidences = np.fromiter((confidence[entry_id] for entry_id in ids), dtype=np.float32, count=len(ids))
    key = similarities if order == "similarity" else np.maximum(similarities, 0.0) * confidences
    best_first = np.argsort(-key, kind="stable")
    kept = np.arange(len(best_first)) < keep_top
    if keep_above is not None:
        kept |= similarities[best_first] >= keep_above
    return [
        RankedCandidate(ids[index], confidence[ids[index]], float(similarities[index]))
        for index in best_first[kept].tolist()
    ]
