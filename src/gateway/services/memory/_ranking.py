"""Recall's ranking in this process: open the candidates' embeddings, compare them with the turn, best first.

Comparing is pure CPU over every candidate of a workspace, so it runs on a worker
thread, never on the event loop: a workspace with thousands of entries would
otherwise hold up every other request the process serves for the length of it.
The opened embeddings are kept in ``held`` where the process keeps any.

A memory service runs this same ranking for the gateways that hand theirs off.
"""

import asyncio
import math
import uuid
from collections.abc import Mapping, Sequence

from gateway.core.unit_of_work import UnitOfWork
from gateway.ports.memory_ranking_port import MemoryRankingPort, RankCandidate, RankedCandidate, RankOrder
from gateway.services.memory._recall_vectors import RecallVectors
from gateway.services.memory._scoring import relevance
from gateway.services.memory._store import EntryStore


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
        async with self._uow:
            vectors = await self._store.embeddings(workspace_id, [candidate.id for candidate in candidates], self._held)
        return await asyncio.to_thread(
            rank_vectors, candidates, vectors, queries, order=order, keep_top=keep_top, keep_above=keep_above
        )

    async def forget(self, workspace_id: uuid.UUID) -> None:
        if self._held is not None:
            self._held.forget(workspace_id)


def rank_vectors(
    candidates: Sequence[RankCandidate],
    vectors: Mapping[uuid.UUID, Sequence[float]],
    queries: Sequence[Sequence[float]],
    *,
    order: RankOrder,
    keep_top: int,
    keep_above: float | None,
) -> list[RankedCandidate]:
    """What :meth:`LocalMemoryRanking.rank` returns, from embeddings already opened. Pure, so a thread may run it.

    A candidate with no vector in ``vectors`` is left out. Ties keep the order of ``candidates``.
    """
    # Each query's norm once, not once per candidate.
    normed = [(query, math.sqrt(math.sumprod(query, query))) for query in queries]
    scored: list[RankedCandidate] = []
    for candidate in candidates:
        vector = vectors.get(candidate.id)
        if vector is None:
            continue
        norm = math.sqrt(math.sumprod(vector, vector))
        similarity = max(
            (
                math.sumprod(query, vector) / (query_norm * norm)
                if len(query) == len(vector) and query_norm and norm
                else 0.0
            )
            for query, query_norm in normed
        )
        scored.append(RankedCandidate(candidate.id, candidate.confidence, similarity))
    if order == "similarity":
        scored.sort(key=lambda ranked: ranked.similarity, reverse=True)
    else:
        scored.sort(key=lambda ranked: relevance(ranked.similarity, ranked.confidence), reverse=True)
    return [
        ranked
        for position, ranked in enumerate(scored)
        if position < keep_top or (keep_above is not None and ranked.similarity >= keep_above)
    ]
