"""Recall's ranking handed to a memory service the deployment runs apart: ``otari memory-service``.

The gateway sends the candidates it read from the database, by id and confidence,
and the turn's vectors; the service opens their embeddings, holds them, ranks,
and answers with ids and scores. No text and no stored embedding crosses.

Every call carries ``memory_service_token``. One client per process, so calls
share its connections. A call that does not answer within ``timeout`` fails as
:class:`MemoryRankingError`, so automatic recall goes on without memory rather
than hold up the request.
"""

import uuid
from collections.abc import Sequence
from typing import Any

import httpx

from gateway.core.config import API_ROOT
from gateway.log_config import logger
from gateway.ports.memory_ranking_port import (
    MemoryRankingError,
    MemoryRankingPort,
    RankCandidate,
    RankedCandidate,
    RankOrder,
)

RANK_PATH = f"{API_ROOT}/memory/engine/rank"
FORGET_PATH = f"{API_ROOT}/memory/engine/forget"
# Below automatic recall's own budget, so a slow service fails the ranking, not the whole recall.
TIMEOUT_S = 2.0


class HttpMemoryRanking(MemoryRankingPort):
    """Ranks on the memory service at ``url``."""

    def __init__(
        self,
        url: str,
        token: str,
        *,
        timeout: float = TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
            follow_redirects=False,
            transport=transport,
        )

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
        body = {
            "workspace_id": str(workspace_id),
            "candidates": [[str(candidate.id), candidate.confidence] for candidate in candidates],
            "queries": [list(query) for query in queries],
            "order": order,
            "keep_top": keep_top,
            "keep_above": keep_above,
        }
        answer = await self._post(RANK_PATH, body)
        try:
            return [
                RankedCandidate(uuid.UUID(entry_id), float(confidence), float(similarity))
                for entry_id, confidence, similarity in answer["ranked"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            msg = "the memory service answered in a shape this gateway does not read"
            raise MemoryRankingError(msg) from exc

    async def forget(self, workspace_id: uuid.UUID) -> None:
        try:
            await self._post(FORGET_PATH, {"workspace_id": str(workspace_id)})
        except MemoryRankingError as exc:
            # What it holds of the workspace expires on its own within minutes; the erase itself stands.
            logger.warning("The memory service did not drop what it holds of an erased workspace: %s", exc)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, path: str, body: dict[str, Any]) -> Any:
        try:
            response = await self._client.post(path, json=body)
        except httpx.TimeoutException as exc:
            msg = "the memory service did not answer in time"
            raise MemoryRankingError(msg) from exc
        except httpx.HTTPError as exc:
            msg = f"the memory service could not be reached ({type(exc).__name__})"
            raise MemoryRankingError(msg) from exc
        if response.status_code != httpx.codes.OK:
            msg = f"the memory service answered {response.status_code}"
            raise MemoryRankingError(msg)
        try:
            return response.json()
        except ValueError as exc:
            msg = "the memory service answered with something other than JSON"
            raise MemoryRankingError(msg) from exc
