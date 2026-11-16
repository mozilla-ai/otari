"""Recall's ranking: in this process, handed to a memory service, and the settings that choose between them."""

import asyncio
import json
import uuid
from typing import Any

import httpx
import pytest

from gateway.adapters.memory_ranking_adapter import FORGET_PATH, RANK_PATH, HttpMemoryRanking
from gateway.api.routes.memory import FEATURE
from gateway.core.config import GatewayConfig
from gateway.main import _validate_memory_service
from gateway.ports.memory_ranking_port import MemoryRankingError, RankCandidate, RankedCandidate
from gateway.services.memory._ranking import rank_vectors

A, B, C, D = (uuid.UUID(int=n) for n in range(1, 5))
TOKEN = "t" * 40


def _ids(ranked: list[RankedCandidate]) -> list[uuid.UUID]:
    return [entry.id for entry in ranked]


# -- Ranking in this process -------------------------------------------------


def test_similarity_keeps_the_top_and_whatever_else_clears_the_floor() -> None:
    candidates = [RankCandidate(A, 0.5), RankCandidate(B, 0.5), RankCandidate(C, 0.5), RankCandidate(D, 0.5)]
    vectors = {A: [1.0, 0.0], B: [0.9, 0.1], C: [0.0, 1.0], D: [0.7, 0.7]}

    top_one = rank_vectors(candidates, vectors, [[1.0, 0.0]], order="similarity", keep_top=1, keep_above=None)
    with_floor = rank_vectors(candidates, vectors, [[1.0, 0.0]], order="similarity", keep_top=1, keep_above=0.7)

    assert _ids(top_one) == [A]
    assert _ids(with_floor) == [A, B, D], "C is below the floor; D clears it at 0.707"
    assert with_floor[0].similarity == pytest.approx(1.0)


def test_relevance_weighs_similarity_by_confidence() -> None:
    candidates = [RankCandidate(A, 0.1), RankCandidate(B, 0.9)]
    vectors = {A: [1.0, 0.0], B: [0.8, 0.6]}

    by_similarity = rank_vectors(candidates, vectors, [[1.0, 0.0]], order="similarity", keep_top=2, keep_above=None)
    by_relevance = rank_vectors(candidates, vectors, [[1.0, 0.0]], order="relevance", keep_top=2, keep_above=None)

    assert _ids(by_similarity) == [A, B]
    assert _ids(by_relevance) == [B, A]


def test_the_best_of_several_queries_counts_and_what_did_not_open_is_left_out() -> None:
    candidates = [RankCandidate(A, 0.5), RankCandidate(B, 0.5), RankCandidate(C, 0.5)]
    vectors = {A: [1.0, 0.0], B: [0.0, 1.0], C: [1.0, 0.0, 0.0]}

    ranked = rank_vectors(
        candidates, vectors | {}, [[1.0, 0.0], [0.0, 1.0]], order="similarity", keep_top=5, keep_above=None
    )
    missing = rank_vectors(
        [RankCandidate(D, 0.5)], vectors, [[1.0, 0.0]], order="similarity", keep_top=5, keep_above=None
    )

    assert {entry.id: entry.similarity for entry in ranked} == {A: 1.0, B: 1.0, C: 0.0}, "C's size differs"
    assert missing == []


# -- Ranking handed to a memory service ----------------------------------------


def _service(handler: Any) -> HttpMemoryRanking:
    return HttpMemoryRanking("http://memory-service/", TOKEN, transport=httpx.MockTransport(handler))


def test_the_gateway_sends_ids_confidences_and_the_turn_and_reads_back_the_order() -> None:
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ranked": [[str(B), 0.9, 0.8], [str(A), 0.5, 0.7]]})

    ranked = asyncio.run(
        _service(answer).rank(
            C, [RankCandidate(A, 0.5), RankCandidate(B, 0.9)], [[1.0, 0.0]], order="similarity", keep_top=10
        )
    )

    assert ranked == [RankedCandidate(B, 0.9, 0.8), RankedCandidate(A, 0.5, 0.7)]
    [request] = seen
    assert request.url.path == RANK_PATH
    assert request.headers["Authorization"] == f"Bearer {TOKEN}"
    assert json.loads(request.content) == {
        "workspace_id": str(C),
        "candidates": [[str(A), 0.5], [str(B), 0.9]],
        "queries": [[1.0, 0.0]],
        "order": "similarity",
        "keep_top": 10,
        "keep_above": None,
    }


def test_nothing_to_rank_makes_no_call() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no call expected")

    assert asyncio.run(_service(fail).rank(C, [], [[1.0]], order="similarity", keep_top=3)) == []


@pytest.mark.parametrize(
    "handler",
    [
        lambda request: httpx.Response(401, json={"detail": "no"}),
        lambda request: httpx.Response(200, text="not json"),
        lambda request: httpx.Response(200, json={"ranked": [["not-a-uuid", 1, 1]]}),
        lambda request: (_ for _ in ()).throw(httpx.ConnectError("down", request=request)),
        lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("slow", request=request)),
    ],
    ids=["refused", "not-json", "wrong-shape", "unreachable", "slow"],
)
def test_a_service_that_does_not_answer_well_fails_as_a_ranking_error(handler: Any) -> None:
    with pytest.raises(MemoryRankingError) as raised:
        asyncio.run(_service(handler).rank(C, [RankCandidate(A, 0.5)], [[1.0]], order="similarity", keep_top=3))

    assert TOKEN not in str(raised.value)


def test_forgetting_on_a_service_that_is_down_does_not_fail_the_erase() -> None:
    seen: list[str] = []

    def down(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        raise httpx.ConnectError("down", request=request)

    asyncio.run(_service(down).forget(C))

    assert seen == [FORGET_PATH]


# -- The settings that split a deployment ---------------------------------------


def _config(**values: Any) -> GatewayConfig:
    return GatewayConfig.model_validate(
        {"database_url": "sqlite+aiosqlite:///:memory:", "memory_enabled": True} | values
    )


@pytest.mark.parametrize(
    ("values", "refused"),
    [
        ({"memory_service_url": "http://m:8011"}, "memory_service_token"),
        ({"memory_engine": True}, "memory_service_token"),
        ({"memory_engine": True, "memory_service_token": "short"}, "at least"),
        ({"memory_engine": True, "memory_service_url": "http://m:8011", "memory_service_token": TOKEN}, "unset"),
        (
            {"memory_enabled": False, "memory_service_url": "http://m:8011", "memory_service_token": TOKEN},
            "memory_enabled",
        ),
        ({"memory_engine": True, "memory_service_token": TOKEN, "master_key": TOKEN}, "master key"),
    ],
)
def test_a_split_that_could_not_tell_its_gateways_from_anyone_refuses_to_start(
    values: dict[str, Any], refused: str
) -> None:
    with pytest.raises(ValueError, match=refused):
        _validate_memory_service(_config(**values))


def test_a_gateway_with_a_memory_service_leaves_the_consolidator_to_it() -> None:
    config = _config(memory_service_url="http://m:8011", memory_service_token=TOKEN)
    assert FEATURE.worker is not None

    asyncio.run(asyncio.wait_for(FEATURE.worker(config), timeout=1))


def test_only_the_memory_service_mounts_its_routes() -> None:
    gateway = FEATURE.routers(_config(memory_service_url="http://m:8011", memory_service_token=TOKEN))
    service = FEATURE.routers(_config(memory_engine=True, memory_service_token=TOKEN))

    assert len(service) == len(gateway) + 1


def test_the_matrix_ranking_orders_as_the_cosine_of_each_entry_does() -> None:
    import random

    from gateway.services.memory._scoring import cosine, relevance

    rng = random.Random(7)
    candidates = [RankCandidate(uuid.UUID(int=n + 10), rng.uniform(0.1, 1.0)) for n in range(300)]
    vectors = {candidate.id: [rng.gauss(0, 1) for _ in range(64)] for candidate in candidates}
    queries = [[rng.gauss(0, 1) for _ in range(64)] for _ in range(2)]

    for order in ("similarity", "relevance"):
        ranked = rank_vectors(candidates, vectors, queries, order=order, keep_top=25, keep_above=None)
        expected = sorted(
            candidates,
            key=lambda candidate: (
                max(cosine(query, vectors[candidate.id]) for query in queries)
                if order == "similarity"
                else relevance(max(cosine(query, vectors[candidate.id]) for query in queries), candidate.confidence)
            ),
            reverse=True,
        )[:25]
        assert _ids(ranked) == [candidate.id for candidate in expected]
        for entry in ranked:
            assert entry.similarity == pytest.approx(
                max(cosine(query, vectors[entry.id]) for query in queries), abs=1e-5
            )
