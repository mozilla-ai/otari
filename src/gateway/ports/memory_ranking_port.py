"""Where agent memory's recall ranks the entries a member may recall against a turn.

Ranking opens each candidate's embedding, compares it with the turn, and keeps
the opened embeddings for the next turn. That is the costliest part of a recall,
in CPU and in memory. One implementation ranks in the gateway's own process;
another hands it to a memory service the deployment runs apart, so a busy
memory cannot slow the requests the gateway serves.

The caller always decides which entries are candidates, from the database and
under its row-level security, and passes them by id. A ranker only orders what
it is handed: it never widens what a member may recall. Embeddings and texts
never cross this port, only ids, confidences and the turn's own vectors.

Failures cross as :class:`MemoryRankingError`. A caller serves the request
without memory rather than fail it.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

RankOrder = Literal["similarity", "relevance"]


class MemoryRankingError(Exception):
    """The ranker could not answer: unreachable, too slow, or it refused the call."""


@dataclass(frozen=True)
class RankCandidate:
    """An entry a member may recall, by id, with the confidence relevance weighs it by."""

    id: uuid.UUID
    confidence: float


@dataclass(frozen=True)
class RankedCandidate:
    """A candidate with how alike its embedding is to the turn: the best of the turn's vectors."""

    id: uuid.UUID
    confidence: float
    similarity: float


class MemoryRankingPort(Protocol):
    """Order a workspace's candidates by how alike they are to a turn."""

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
        """The candidates best first by ``order``: the first ``keep_top``, and any other alike at least ``keep_above``.

        ``similarity`` orders by the best cosine similarity to any of ``queries``; ``relevance`` by that similarity
        weighed by the candidate's confidence. A candidate whose embedding does not open is left out.

        Raises:
            MemoryRankingError: the ranker could not answer.
        """
        ...

    async def forget(self, workspace_id: uuid.UUID) -> None:
        """Drop every embedding held for the workspace, as its memory is erased. Best effort: never raises."""
        ...
