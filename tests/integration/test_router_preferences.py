"""Integration tests for the kNN router: preference endpoints + e2e routing.

Drives the real FastAPI app with ``router_backend=knn`` over a temporary SQLite
database. Only two things are faked: the embedding provider (a deterministic
2-D embedder that separates "easy" from "hard" prompts) and the chat provider
call (``acompletion``). Everything else runs for real: auth, the preference
endpoints, the routing-memory store, the kNN vote, and the chat dispatch seam.

What this proves end to end:
- compare fans a prompt out to every candidate model;
- rank persists rank-to-scalar routing-memory records and warms the tenant;
- a cold tenant passes the requested model through, and once warm the router
  sends easy prompts to the cheap model and hard prompts to the strong one;
- rank is rejected when the backend is not the kNN backend.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
)
from fastapi.testclient import TestClient

from gateway.api.deps import reset_config
from gateway.core.config import API_KEY_HEADER, GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app

CHEAP = "openai:gpt-3.5-turbo"
STRONG = "openai:gpt-4o"

_EASY = "easy"
_HARD = "hard"


def _kind(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ("plus", "sum", "add", "+")):
        return _EASY
    return _HARD


def _fake_vec(text: str) -> list[float]:
    # 2-D: axis 0 = "easy" mass, axis 1 = "hard" mass. Same-kind prompts cluster
    # tightly in cosine space; opposite kinds are orthogonal.
    return [1.0, 0.01] if _kind(text) == _EASY else [0.01, 1.0]


class _Emb:
    def __init__(self, vec: list[float]) -> None:
        self.embedding = vec


class _EmbResult:
    def __init__(self, vec: list[float]) -> None:
        self.data = [_Emb(vec)]


async def fake_aembedding(**kwargs: Any) -> _EmbResult:
    text = kwargs["inputs"]
    if isinstance(text, list):
        text = text[0]
    return _EmbResult(_fake_vec(str(text)))


def _completion(model: str) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-knn",
        object="chat.completion",
        created=1700000000,
        model=model,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="ok"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _build_client(backend: str, *, fake_embeddings: bool = True) -> Generator[TestClient]:
    cfg = GatewayConfig(
        database_url="sqlite:///./_knn_test.db",
        master_key="test-master-key",
        auto_migrate=True,
        require_pricing=False,
        router_backend=backend,
        router_candidates=f"{CHEAP},{STRONG}",
        router_k=2,
        router_seed_count=4,
        router_alpha=0.3,
    )
    app = create_app(cfg)
    if fake_embeddings:
        with patch("gateway.services.knn_router.aembedding", new=fake_aembedding):
            with TestClient(app) as client:
                yield client
    else:
        with TestClient(app) as client:
            yield client
    reset_config()
    reset_db()


@pytest.fixture
def knn_client(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    monkeypatch.chdir(tmp_path)
    yield from _build_client("knn")


@pytest.fixture
def noop_client(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    monkeypatch.chdir(tmp_path)
    yield from _build_client("noop")


@pytest.fixture
def live_knn_client(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    """A kNN client using the real embedding provider (no fake)."""
    monkeypatch.chdir(tmp_path)
    yield from _build_client("knn", fake_embeddings=False)


def _headers(token: str) -> dict[str, str]:
    return {API_KEY_HEADER: f"Bearer {token}"}


def _master(client: TestClient) -> dict[str, str]:
    return _headers("test-master-key")


def _make_key(client: TestClient, user_id: str | None = None) -> str:
    body: dict[str, Any] = {"key_name": "k"}
    if user_id is not None:
        body["user_id"] = user_id
    resp = client.post("/v1/keys", json=body, headers=_master(client))
    assert resp.status_code == 200, resp.text
    return str(resp.json()["key"])


def _scores_from_ranking(ranking: list[str]) -> dict[str, float]:
    """Best-to-worst order -> linear quality scores (rank-1 = 1.0, rank-N = 0.0)."""
    n = len(ranking)
    return {model: (1.0 if n == 1 else 1.0 - i / (n - 1)) for i, model in enumerate(ranking)}


def _rank(
    client: TestClient,
    headers: dict[str, str],
    prompt: str,
    ranking: list[str],
    *,
    task_id: str | None = None,
) -> None:
    body: dict[str, Any] = {"prompt": prompt, "scores": _scores_from_ranking(ranking)}
    if task_id is not None:
        body["task_id"] = task_id
    resp = client.post("/v1/router/preferences/rank", json=body, headers=headers)
    assert resp.status_code == 200, resp.text


def _seed(client: TestClient, headers: dict[str, str], *, task_id: str | None = None) -> None:
    # Easy region: cheap is preferred. Hard region: strong is preferred.
    _rank(client, headers, "what is 2 plus 2", [CHEAP, STRONG], task_id=task_id)
    _rank(client, headers, "compute the sum of 3 and 4", [CHEAP, STRONG], task_id=task_id)
    _rank(client, headers, "prove the reason the sky appears blue", [STRONG, CHEAP], task_id=task_id)
    _rank(client, headers, "explain why entropy increases", [STRONG, CHEAP], task_id=task_id)


def _chat_model(client: TestClient, headers: dict[str, str], prompt: str, *, task: str | None = None) -> str:
    captured: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        captured["model"] = kwargs["model"]
        return _completion(model=kwargs["model"])

    request_headers = dict(headers)
    if task is not None:
        request_headers["Otari-Router-Task"] = task
    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": STRONG, "messages": [{"role": "user", "content": prompt}]},
            headers=request_headers,
        )
    assert resp.status_code == 200, resp.text
    return str(captured["model"])


# ---------------------------------------------------------------------------
# Preference endpoints
# ---------------------------------------------------------------------------


def test_compare_fans_out_to_all_candidate_models(knn_client: TestClient) -> None:
    token = _make_key(knn_client, "tenant-compare")

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        return _completion(model=kwargs["model"])

    with patch("gateway.api.routes.router.acompletion", new=mock_acompletion):
        resp = knn_client.post(
            "/v1/router/preferences/compare",
            json={"prompt": "what is 2 plus 2", "models": [CHEAP, STRONG]},
            headers=_headers(token),
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["prompt"] == "what is 2 plus 2"
    returned = {r["model"]: r["content"] for r in body["responses"]}
    assert returned[CHEAP] == "ok"
    assert returned[STRONG] == "ok"


def test_compare_isolates_a_single_model_failure(knn_client: TestClient) -> None:
    token = _make_key(knn_client, "tenant-compare-fail")

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        if kwargs["model"] == "gpt-3.5-turbo":
            raise RuntimeError("cheap model down")
        return _completion(model=kwargs["model"])

    with patch("gateway.api.routes.router.acompletion", new=mock_acompletion):
        resp = knn_client.post(
            "/v1/router/preferences/compare",
            json={"prompt": "hi", "models": [CHEAP, STRONG]},
            headers=_headers(token),
        )
    assert resp.status_code == 200, resp.text
    by_model = {r["model"]: r for r in resp.json()["responses"]}
    assert by_model[CHEAP]["error"] is not None
    assert by_model[CHEAP]["content"] is None
    assert by_model[STRONG]["content"] == "ok"


def test_rank_records_and_warms_tenant(knn_client: TestClient) -> None:
    token = _make_key(knn_client, "tenant-rank")
    headers = _headers(token)

    status = knn_client.get("/v1/router/status", headers=headers).json()
    assert status["backend"] == "knn"
    assert status["default_pool"] == {"records": 0, "warm": False}
    assert status["tasks"] == []

    _seed(knn_client, headers)

    status = knn_client.get("/v1/router/status", headers=headers).json()
    # 4 examples = 4 records (one per example); seed_count is 4, so the pool is warm.
    assert status["default_pool"] == {"records": 4, "warm": True}


def test_rank_accepts_tied_scores(knn_client: TestClient) -> None:
    token = _make_key(knn_client, "tenant-ties")
    headers = _headers(token)
    resp = knn_client.post(
        "/v1/router/preferences/rank",
        json={"prompt": "rate both equally", "scores": {CHEAP: 0.7, STRONG: 0.7}},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    # One example -> one record, even though it scored two models (tied here).
    assert resp.json()["recorded"] == 1
    assert knn_client.get("/v1/router/status", headers=headers).json()["default_pool"]["records"] == 1


def test_rank_rejects_out_of_range_or_empty_scores(knn_client: TestClient) -> None:
    headers = _headers(_make_key(knn_client, "tenant-bad-scores"))
    for bad in ({CHEAP: 1.5}, {CHEAP: -0.1}, {}):
        resp = knn_client.post(
            "/v1/router/preferences/rank",
            json={"prompt": "x", "scores": bad},
            headers=headers,
        )
        assert resp.status_code == 422, resp.text


def test_rank_rejected_when_backend_not_knn(noop_client: TestClient) -> None:
    token = _make_key(noop_client, "tenant-noop")
    resp = noop_client.post(
        "/v1/router/preferences/rank",
        json={"prompt": "x", "scores": {CHEAP: 1.0, STRONG: 0.0}},
        headers=_headers(token),
    )
    assert resp.status_code == 400
    assert "knn" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# End-to-end routing through the chat seam
# ---------------------------------------------------------------------------


def test_cold_tenant_passes_requested_model_through(knn_client: TestClient) -> None:
    token = _make_key(knn_client, "tenant-cold")
    # No seed data: the router must not reroute; the requested (strong) model runs.
    served = _chat_model(knn_client, _headers(token), "what is 2 plus 2")
    assert served == STRONG


def test_warm_tenant_routes_easy_to_cheap_and_hard_to_strong(knn_client: TestClient) -> None:
    token = _make_key(knn_client, "tenant-warm")
    headers = _headers(token)
    _seed(knn_client, headers)

    # Easy prompt, similar to the cheap-preferred seeds: routed to the cheap model.
    assert _chat_model(knn_client, headers, "what is 7 plus 8") == CHEAP
    # Hard prompt, similar to the strong-preferred seeds: stays on the strong model.
    assert _chat_model(knn_client, headers, "prove why the ocean is salty") == STRONG


def test_routing_memory_is_isolated_per_tenant(knn_client: TestClient) -> None:
    warm = _make_key(knn_client, "tenant-isolated-warm")
    cold = _make_key(knn_client, "tenant-isolated-cold")
    _seed(knn_client, _headers(warm))

    # The warm tenant reroutes easy prompts; a different, unseeded tenant does
    # not (its store is empty), proving per-tenant scoping.
    assert _chat_model(knn_client, _headers(warm), "what is 1 plus 1") == CHEAP
    assert _chat_model(knn_client, _headers(cold), "what is 1 plus 1") == STRONG


def test_task_partitions_route_independently(knn_client: TestClient) -> None:
    # One tenant, two task partitions with OPPOSITE preferences in the easy
    # region. The same prompt must route by the partition the request selects,
    # proving a task only ever votes over its own records.
    headers = _headers(_make_key(knn_client, "tenant-tasks"))
    _seed(knn_client, headers, task_id="alpha")  # easy -> cheap
    # "beta": flip the easy region so the cheap model is the bad choice there.
    _rank(knn_client, headers, "what is 2 plus 2", [STRONG, CHEAP], task_id="beta")
    _rank(knn_client, headers, "compute the sum of 3 and 4", [STRONG, CHEAP], task_id="beta")
    _rank(knn_client, headers, "prove the reason the sky appears blue", [CHEAP, STRONG], task_id="beta")
    _rank(knn_client, headers, "explain why entropy increases", [CHEAP, STRONG], task_id="beta")

    assert _chat_model(knn_client, headers, "what is 5 plus 5", task="alpha") == CHEAP
    assert _chat_model(knn_client, headers, "what is 5 plus 5", task="beta") == STRONG


def test_unseeded_task_partition_passes_through(knn_client: TestClient) -> None:
    # Hard isolation: a warm task does not lend its records to a cold sibling.
    headers = _headers(_make_key(knn_client, "tenant-task-cold"))
    _seed(knn_client, headers, task_id="alpha")

    # The selected partition is warm and reroutes; an unseeded partition is cold
    # and passes the requested model through even though "alpha" is warm.
    assert _chat_model(knn_client, headers, "what is 5 plus 5", task="alpha") == CHEAP
    assert _chat_model(knn_client, headers, "what is 5 plus 5", task="beta") == STRONG
    # No task selector falls back to the tenant's whole pool, which here is all
    # "alpha" records, so it reroutes too.
    assert _chat_model(knn_client, headers, "what is 5 plus 5") == CHEAP


def test_status_reports_pools_with_independent_warmth(knn_client: TestClient) -> None:
    headers = _headers(_make_key(knn_client, "tenant-task-status"))
    _seed(knn_client, headers, task_id="alpha")  # 4 examples -> warm (seed_count 4)
    _rank(knn_client, headers, "what is 2 plus 2", [CHEAP, STRONG], task_id="beta")  # 1 example -> cold
    _rank(knn_client, headers, "untagged prompt", [CHEAP, STRONG])  # 1 untagged, not a partition

    body = knn_client.get("/v1/router/status", headers=headers).json()
    assert body["backend"] == "knn"
    # The default pool is every record the tenant has (4 + 1 + 1); warms on its own.
    assert body["default_pool"] == {"records": 6, "warm": True}
    tasks = {t["task_id"]: t for t in body["tasks"]}
    # Only tagged partitions are listed; untagged records are not a partition.
    assert set(tasks) == {"alpha", "beta"}
    assert tasks["alpha"]["records"] == 4 and tasks["alpha"]["warm"] is True
    assert tasks["beta"]["records"] == 1 and tasks["beta"]["warm"] is False


def test_onboarding_with_a_plain_api_key_no_explicit_user(knn_client: TestClient) -> None:
    """The common first-run path: create one API key (no user_id), rank a few
    prompts with it, then serve traffic with the same key. The auto-created
    virtual user is the tenant for both, so routing warms and reroutes.
    """
    token = _make_key(knn_client)  # no explicit user_id -> auto virtual user
    headers = _headers(token)

    assert knn_client.get("/v1/router/status", headers=headers).json()["default_pool"]["warm"] is False
    _seed(knn_client, headers)
    assert knn_client.get("/v1/router/status", headers=headers).json()["default_pool"]["warm"] is True

    assert _chat_model(knn_client, headers, "what is 4 plus 4") == CHEAP


def test_trace_sticky_holds_model_across_conversation_turns(knn_client: TestClient) -> None:
    """Trace-sticky must persist across separate requests of one conversation.

    Regression guard: the chat handler resolves the backend once per request, so
    the trace-decision cache only survives if the backend instance is reused. A
    follow-up turn that is itself "hard" must still be served by the model the
    trace started on, not re-routed.
    """
    token = _make_key(knn_client, "tenant-trace")
    headers = _headers(token)
    _seed(knn_client, headers)

    opener = "what is 5 plus 5"
    # Turn 1: easy opener routes to the cheap model and anchors the trace.
    assert _chat_model(knn_client, headers, opener) == CHEAP

    # Turn 2: same conversation (client resends history), but the new turn is a
    # hard prompt that on its own would route to the strong model. Stickiness
    # keeps it on the cheap model the trace started with.
    captured: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        captured["model"] = kwargs["model"]
        return _completion(model=kwargs["model"])

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        resp = knn_client.post(
            "/v1/chat/completions",
            json={
                "model": STRONG,
                "messages": [
                    {"role": "user", "content": opener},
                    {"role": "assistant", "content": "10"},
                    {"role": "user", "content": "now prove why entropy always increases"},
                ],
            },
            headers=headers,
        )
    assert resp.status_code == 200, resp.text
    assert captured["model"] == CHEAP


# ---------------------------------------------------------------------------
# Live end to end: real embeddings + real provider call (skipped without a key)
# ---------------------------------------------------------------------------

_NO_KEY = not os.getenv("OPENAI_API_KEY")


@pytest.mark.skipif(_NO_KEY, reason="needs OPENAI_API_KEY")
def test_live_knn_routes_easy_prompt_to_cheaper_model(live_knn_client: TestClient) -> None:
    """Full stack with live OpenAI models: real text-embedding-3-small builds the
    routing memory and embeds the query, and a request for the strong model is
    actually answered by the cheap one for an easy, well-clustered prompt.
    """
    token = _make_key(live_knn_client, "tenant-live")
    headers = _headers(token)
    # Seed with real embeddings: arithmetic prompts prefer cheap; open-ended
    # reasoning prompts prefer strong.
    _rank(live_knn_client, headers, "what is 12 plus 7", [CHEAP, STRONG])
    _rank(live_knn_client, headers, "add 21 and 34", [CHEAP, STRONG])
    _rank(live_knn_client, headers, "compose a short poem about the sea", [STRONG, CHEAP])
    _rank(live_knn_client, headers, "explain the causes of the french revolution", [STRONG, CHEAP])

    status = live_knn_client.get("/v1/router/status", headers=headers).json()
    assert status["default_pool"]["warm"] is True

    # Real chat call. The caller asks for the strong model; the router should send
    # this arithmetic prompt to the cheap model, and the provider response reports
    # the model that actually ran.
    resp = live_knn_client.post(
        "/v1/chat/completions",
        json={
            "model": STRONG,
            "messages": [{"role": "user", "content": "what is 9 plus 6"}],
            "max_tokens": 5,
            "temperature": 0,
        },
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    served = resp.json()["model"]
    assert "3.5" in served, f"expected the cheap gpt-3.5 model to serve an easy prompt, got {served}"
