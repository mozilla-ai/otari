"""End-to-end behavior for a learned routing policy (a `router:` select entry).

A learned policy is an ordinary routing policy whose ordering comes from a router
backend instead of from static rules, so the invariants it has to keep are the
policy invariants: the caller sees the policy name, billing keys on the model that
actually served, the caller's allow-list still wins, and a failure falls over to
the next candidate. What is new is *where* the head candidate comes from.

The embedding provider is faked (two orthogonal regions, "easy" and "hard") so the
kNN vote is deterministic. Everything else runs for real: auth, the preference
API, the routing-memory table, the compiler, the attempt walker, and settlement.
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
)
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from gateway.core.config import API_KEY_HEADER, GatewayConfig, PricingConfig
from gateway.db import Base, get_db
from gateway.main import create_app
from gateway.models.routing import RoutingConfig

from .conftest import _run_alembic_migrations, build_async_session_override

HEADERS = {API_KEY_HEADER: "Bearer test-master-key"}

CHEAP = "openai:gpt-5-nano"
STRONG = "openai:gpt-5"
OTHER = "anthropic:claude-haiku-4-5"

USER = "test-user"


# -- deterministic embeddings ----------------------------------------------


def _is_easy(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in ("plus", "sum", "add", "capital of"))


def _fake_vector(text: str) -> list[float]:
    # 2-D: axis 0 is "easy" mass, axis 1 is "hard" mass. Same-kind prompts cluster
    # tightly in cosine space; opposite kinds are near-orthogonal.
    return [1.0, 0.01] if _is_easy(text) else [0.01, 1.0]


class _Embedding:
    def __init__(self, vector: list[float]) -> None:
        self.embedding = vector


class _EmbeddingResult:
    def __init__(self, vector: list[float]) -> None:
        self.data = [_Embedding(vector)]


async def _fake_aembedding(**kwargs: Any) -> _EmbeddingResult:
    text = kwargs["inputs"]
    if isinstance(text, list):
        text = text[0]
    return _EmbeddingResult(_fake_vector(str(text)))


def _completion(model: str) -> ChatCompletion:
    return ChatCompletion(
        id="cmpl-1",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hello"),
            )
        ],
        created=0,
        model=model,
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://upstream")
    return httpx.HTTPStatusError(str(status), request=request, response=httpx.Response(status, request=request))


# -- fixtures ---------------------------------------------------------------


@pytest.fixture
def learned_config(postgres_url: str) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        providers={"openai": {"api_key": "sk-openai"}, "anthropic": {"api_key": "sk-ant"}},
        # k=2 and seed=4 keep the teaching loop in these tests short while still
        # exercising both gates.
        router_k=2,
        router_seed_count=4,
        router_alpha=0.3,
        pricing={
            "openai:gpt-5-nano": PricingConfig(input_price_per_million=0.5, output_price_per_million=1.5),
            "openai:gpt-5": PricingConfig(input_price_per_million=2.5, output_price_per_million=10.0),
            "anthropic:claude-haiku-4-5": PricingConfig(
                input_price_per_million=1.0, output_price_per_million=4.0
            ),
        },
        routing=RoutingConfig.model_validate(
            {
                "policies": {
                    # The shape the docs recommend: the router ranks a cheap/strong
                    # pool, and the strong model is the default that serves when it
                    # declines.
                    "smart": {
                        "select": [
                            {"router": "knn", "candidates": [CHEAP, STRONG]},
                            {"default": STRONG},
                        ]
                    },
                    # Same, with an operator failure chain after the routed pool.
                    "smart-failover": {
                        "select": [
                            {"router": "knn", "candidates": [CHEAP, STRONG]},
                            {"default": STRONG},
                        ],
                        "on_failure": [OTHER],
                    },
                    # A router this build does not have: must serve the default.
                    "unknown-router": {
                        "select": [
                            {"router": "cheapest", "candidates": [CHEAP, STRONG]},
                            {"default": STRONG},
                        ]
                    },
                }
            }
        ),
    )


@pytest.fixture
def client(learned_config: GatewayConfig) -> Generator[TestClient]:
    _run_alembic_migrations(learned_config.database_url)
    engine = create_engine(learned_config.database_url, pool_pre_ping=True)
    app = create_app(learned_config)
    override_get_db, dispose_override = build_async_session_override(learned_config.database_url)
    app.dependency_overrides[get_db] = override_get_db
    try:
        with patch("gateway.services.routing.knn.aembedding", new=_fake_aembedding), TestClient(app) as test_client:
            _create_user(test_client)
            yield test_client
    finally:
        dispose_override()
        Base.metadata.drop_all(bind=engine)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version CASCADE"))
            conn.commit()
        engine.dispose()


# -- helpers ---------------------------------------------------------------


def _create_user(client: TestClient, user_id: str = USER) -> None:
    resp = client.post("/v1/users", json={"user_id": user_id}, headers=HEADERS)
    assert resp.status_code == 200, resp.text


def _rank(client: TestClient, prompt: str, scores: dict[str, float], **extra: Any) -> Any:
    """One example, through the batch endpoint."""
    example: dict[str, Any] = {"prompt": prompt, "scores": scores, **extra}
    return client.post(
        "/v1/routing/preferences/rank", json={"user_id": USER, "examples": [example]}, headers=HEADERS
    )


def _teach(client: TestClient, **extra: Any) -> None:
    """Four examples: cheap is good enough on the easy region, not on the hard one."""
    for prompt in ("what is 2 plus 2", "add 3 and 4"):
        assert _rank(client, prompt, {CHEAP: 1.0, STRONG: 1.0}, **extra).status_code == 200
    for prompt in ("prove why the sky appears blue", "explain why entropy increases"):
        assert _rank(client, prompt, {CHEAP: 0.0, STRONG: 1.0}, **extra).status_code == 200


def _chat(
    client: TestClient,
    model: str,
    prompt: str = "what is 2 plus 2",
    *,
    headers: dict[str, str] | None = None,
    **extra: Any,
) -> tuple[Any, list[str]]:
    """POST a chat request, returning the response and the models the provider saw."""
    calls: list[str] = []

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs["model"])
        return _completion(kwargs["model"])

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "user": USER, **extra},
            headers={**HEADERS, **(headers or {})},
        )
    return resp, calls


def _usage_rows(client: TestClient) -> list[dict[str, Any]]:
    resp = client.get("/v1/usage", headers=HEADERS)
    assert resp.status_code == 200, resp.text
    payload: Any = resp.json()
    rows: list[dict[str, Any]] = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    return rows


def _status(client: TestClient, user_id: str = USER) -> dict[str, Any]:
    resp = client.get("/v1/routing/status", params={"user_id": user_id}, headers=HEADERS)
    assert resp.status_code == 200, resp.text
    status: dict[str, Any] = resp.json()
    return status


# -- the decline paths, which are the safe default -------------------------


def test_a_cold_policy_serves_its_default_target(client: TestClient) -> None:
    # Nothing has been taught, so the router declines and the policy behaves
    # exactly like a plain one-target policy.
    resp, calls = _chat(client, "smart")

    assert resp.status_code == 200, resp.text
    assert calls == [STRONG]
    assert resp.json()["model"] == "smart"
    rows = _usage_rows(client)
    assert rows[0]["selection_reason"] == "default"


def test_a_policy_naming_an_unknown_router_serves_its_default(client: TestClient) -> None:
    # A build without the named backend must not fail requests through the policy.
    resp, calls = _chat(client, "unknown-router")

    assert resp.status_code == 200, resp.text
    assert calls == [STRONG]


# -- the routed path -------------------------------------------------------


def test_a_warm_policy_routes_an_easy_prompt_to_the_cheap_candidate(client: TestClient) -> None:
    _teach(client)

    resp, calls = _chat(client, "smart", "what is the capital of France")

    assert resp.status_code == 200, resp.text
    assert calls == [CHEAP]
    # The caller still sees the policy name, so routing is invisible to their code.
    assert resp.json()["model"] == "smart"

    # Billing keys on the model that actually served, attributed to the router.
    served = [row for row in _usage_rows(client) if row["status"] == "success"]
    assert served[0]["model"] == "gpt-5-nano"
    assert served[0]["policy_name"] == "smart"
    assert served[0]["selection_reason"] == "router:knn"


def test_a_score_key_in_another_spelling_still_routes(client: TestClient) -> None:
    """A score key `/rank` accepted must be a key the router can match.

    `provider/model` and `instance:model` name the same model and `/rank` accepts
    either, but the router looks `qualities` up by the policy's own spelling. Stored
    verbatim, an accepted alternate spelling makes that candidate invisible: the pool
    reports warm, the router scores only the models whose spelling happened to match,
    and it confidently serves the strong one forever.
    """
    slash_cheap = CHEAP.replace(":", "/")
    for prompt in ("what is 2 plus 2", "add 3 and 4"):
        assert _rank(client, prompt, {slash_cheap: 1.0, STRONG: 1.0}).status_code == 200
    for prompt in ("prove why the sky appears blue", "explain why entropy increases"):
        assert _rank(client, prompt, {slash_cheap: 0.0, STRONG: 1.0}).status_code == 200

    resp, calls = _chat(client, "smart", "what is the capital of France")

    assert resp.status_code == 200, resp.text
    assert calls == [CHEAP]


def test_two_spellings_of_one_candidate_in_one_example_are_refused(client: TestClient) -> None:
    # Both keys would collapse onto the policy's spelling, so one score would be
    # dropped silently. Saying so beats picking a winner.
    resp = _rank(client, "what is 2 plus 2", {CHEAP: 1.0, CHEAP.replace(":", "/"): 0.0})

    assert resp.status_code == 400, resp.text
    assert "name the same model" in resp.json()["detail"]


def test_a_warm_policy_keeps_a_hard_prompt_on_the_strong_candidate(client: TestClient) -> None:
    _teach(client)

    resp, calls = _chat(client, "smart", "prove that the halting problem is undecidable")

    assert resp.status_code == 200, resp.text
    assert calls == [STRONG]


def test_the_ranking_is_the_failover_chain(client: TestClient) -> None:
    """A routed request that fails lands on the router's next choice.

    This is what makes routing safe to switch on: the cheap model going down costs
    a retry, not the request. Standalone had no failover at all before policies.
    """
    _teach(client)
    calls: list[str] = []

    async def flaky(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs["model"])
        if kwargs["model"] == CHEAP:
            raise _http_error(503)
        return _completion(kwargs["model"])

    with patch("gateway.api.routes.chat.acompletion", new=flaky):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "smart-failover",
                "messages": [{"role": "user", "content": "what is 2 plus 2"}],
                "user": USER,
            },
            headers=HEADERS,
        )

    assert resp.status_code == 200, resp.text
    # The router's ranking first, then the operator's own chain.
    assert calls == [CHEAP, STRONG]
    rows = _usage_rows(client)
    served = [row for row in rows if row["status"] == "success"]
    absorbed = [row for row in rows if row["status"] == "absorbed"]
    assert served[0]["model"] == "gpt-5"
    assert served[0]["selection_reason"] == "router:knn"
    assert absorbed[0]["model"] == "gpt-5-nano"
    assert absorbed[0]["counts_toward_budget"] is False


def test_the_router_never_dispatches_a_candidate_the_key_forbids(client: TestClient) -> None:
    # The allow-list outranks the router: a cheap pick the key may not use must not
    # be served, and the request must still succeed on an allowed candidate.
    _teach(client)
    key = client.post(
        "/v1/keys",
        json={"key_name": "restricted", "user_id": USER, "allowed_models": ["openai:gpt-5"]},
        headers=HEADERS,
    )
    assert key.status_code == 200, key.text
    token = key.json()["key"]

    resp, calls = _chat(
        client, "smart", "what is 2 plus 2", headers={API_KEY_HEADER: f"Bearer {token}"}
    )

    assert resp.status_code == 200, resp.text
    assert calls == [STRONG]


def test_routing_applies_on_the_streaming_path(client: TestClient) -> None:
    _teach(client)
    calls: list[str] = []

    async def mock_stream(**kwargs: Any) -> Any:
        calls.append(kwargs["model"])

        async def _chunks() -> Any:
            from any_llm.types.completion import ChatCompletionChunk

            yield ChatCompletionChunk(
                id="c1", choices=[], created=0, model="gpt-5-nano", object="chat.completion.chunk", usage=None
            )
            yield ChatCompletionChunk(
                id="c2",
                choices=[],
                created=0,
                model="gpt-5-nano",
                object="chat.completion.chunk",
                usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        return _chunks()

    with patch("gateway.api.routes.chat.acompletion", new=mock_stream):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "smart",
                "messages": [{"role": "user", "content": "what is 2 plus 2"}],
                "user": USER,
                "stream": True,
            },
            headers=HEADERS,
        )
        body = resp.text

    assert resp.status_code == 200
    assert "data:" in body
    assert calls == [CHEAP]


# -- per-request control ---------------------------------------------------


def test_otari_router_off_serves_the_default_target(client: TestClient) -> None:
    _teach(client)

    resp, calls = _chat(client, "smart", "what is 2 plus 2", headers={"Otari-Router": "off"})

    assert resp.status_code == 200, resp.text
    assert calls == [STRONG]


def test_an_invalid_otari_router_value_is_a_400(client: TestClient) -> None:
    resp, _ = _chat(client, "smart", headers={"Otari-Router": "maybe"})

    assert resp.status_code == 400
    assert "Otari-Router" in resp.json()["detail"]


def test_trace_stickiness_holds_the_decision_across_turns(client: TestClient) -> None:
    """A conversation decides once. Later turns reuse it even when they look hard.

    Without this, an agent loop would drift between models mid-conversation, which
    both breaks prompt caching and makes the transcript inconsistent.
    """
    _teach(client)
    conversation = {"Otari-Conversation-Id": "conv-1"}

    first, first_calls = _chat(client, "smart", "what is 2 plus 2", headers=conversation)
    assert first.status_code == 200
    assert first_calls == [CHEAP]

    # Turn two of the same conversation: the assistant turn marks it a continuation,
    # and its own content would have routed to the strong model on its own.
    calls: list[str] = []

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs["model"])
        return _completion(kwargs["model"])

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        followup = client.post(
            "/v1/chat/completions",
            json={
                "model": "smart",
                "messages": [
                    {"role": "user", "content": "what is 2 plus 2"},
                    {"role": "assistant", "content": "4"},
                    {"role": "user", "content": "prove that the halting problem is undecidable"},
                ],
                "user": USER,
            },
            headers={**HEADERS, **conversation},
        )

    assert followup.status_code == 200, followup.text
    assert calls == [CHEAP]


def test_a_task_partition_warms_on_its_own(client: TestClient) -> None:
    # Examples filed under a task do not warm the default pool, and a request
    # carrying that task votes only over its own partition.
    _teach(client, task_id="math")

    default_pool, default_calls = _chat(client, "smart", "what is 2 plus 2")
    assert default_calls == [CHEAP]  # untagged examples are in the default pool too
    assert default_pool.status_code == 200

    other_task, other_calls = _chat(
        client, "smart", "what is 2 plus 2", headers={"Otari-Router-Task": "support"}
    )
    assert other_calls == [STRONG]  # the "support" partition is empty, so it declines
    assert other_task.status_code == 200

    same_task, same_calls = _chat(
        client, "smart", "what is 2 plus 2", headers={"Otari-Router-Task": "math"}
    )
    assert same_calls == [CHEAP]
    assert same_task.status_code == 200


def test_routing_memory_is_isolated_per_user(client: TestClient) -> None:
    # One user's examples must never steer another's traffic: the records hold the
    # prompts they send. A global policy therefore warms once per user.
    _teach(client)
    _create_user(client, "other-user")
    calls: list[str] = []

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs["model"])
        return _completion(kwargs["model"])

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "smart",
                "messages": [{"role": "user", "content": "what is 2 plus 2"}],
                "user": "other-user",
            },
            headers=HEADERS,
        )

    assert resp.status_code == 200, resp.text
    # The second user's pool is empty, so their router declines.
    assert calls == [STRONG]
    assert _status(client, "other-user")["default_pool"]["records"] == 0


# -- the teaching loop -----------------------------------------------------


def test_rank_reports_progress_toward_the_seed_count(client: TestClient) -> None:
    first = _rank(client, "what is 2 plus 2", {CHEAP: 1.0, STRONG: 1.0})

    assert first.status_code == 200, first.text
    assert first.json() == {
        "recorded": 1,
        "seed_count": 4,
        "pools": [{"task_id": None, "records": 1, "warm": False}],
    }


def test_rank_records_a_whole_batch_in_one_call(client: TestClient) -> None:
    """A pool needs `seed_count` examples, so teaching is a batch, not a loop.

    One call per example was the shape that made API-only teaching a 20-round-trip
    job; this is what makes the documented recipe a single command.
    """
    resp = client.post(
        "/v1/routing/preferences/rank",
        json={
            "user_id": USER,
            "examples": [
                {"prompt": "what is 2 plus 2", "scores": {CHEAP: 1.0, STRONG: 1.0}},
                {"prompt": "add 3 and 4", "scores": {CHEAP: 1.0, STRONG: 1.0}},
                {"prompt": "prove why the sky appears blue", "scores": {CHEAP: 0.0, STRONG: 1.0}},
                {"prompt": "explain why entropy increases", "scores": {CHEAP: 0.0, STRONG: 1.0}},
            ],
        },
        headers=HEADERS,
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["recorded"] == 4
    # Warm after one call, which is the point of the batch.
    assert body["pools"] == [{"task_id": None, "records": 4, "warm": True}]

    # And the router routes on the strength of it.
    served, calls = _chat(client, "smart", "what is the capital of France")
    assert served.status_code == 200, served.text
    assert calls == [CHEAP]


def test_rank_reports_every_pool_the_batch_touched(client: TestClient) -> None:
    resp = client.post(
        "/v1/routing/preferences/rank",
        json={
            "user_id": USER,
            "examples": [
                {"prompt": "what is 2 plus 2", "scores": {CHEAP: 1.0}},
                {"prompt": "add 3 and 4", "scores": {CHEAP: 1.0}, "task_id": "math"},
            ],
        },
        headers=HEADERS,
    )

    assert resp.status_code == 200, resp.text
    pools = {pool["task_id"]: pool for pool in resp.json()["pools"]}
    # The default pool counts both, because an untagged request votes over everything.
    assert pools[None]["records"] == 2
    assert pools["math"]["records"] == 1


def test_rank_refuses_a_score_key_that_names_no_model(client: TestClient) -> None:
    """The failure this prevents is the worst one the feature has.

    A mistyped selector is otherwise accepted, counts toward the seed count, and
    produces records nothing can match: `/status` reports the pool warm and every
    request declines with "no neighbor scored any candidate", visible only in a log
    line. There is no route that can delete the bad records afterwards.
    """
    resp = client.post(
        "/v1/routing/preferences/rank",
        json={
            "user_id": USER,
            "examples": [{"prompt": "x", "scores": {"openai:gpt-5-nano-typo-xyz": 1.0}}],
        },
        headers=HEADERS,
    )

    assert resp.status_code == 400
    assert "openai:gpt-5-nano-typo-xyz" in resp.json()["detail"]
    # Nothing was written, so the pool cannot look warm on the strength of a typo.
    assert _status(client)["default_pool"]["records"] == 0


def test_rank_allows_teaching_before_any_learned_policy_exists(client: TestClient) -> None:
    # Only resolvability is enforced when the user resolves no learned policy:
    # teaching a pool before writing the policy that reads it is a legitimate order
    # of operations, and refusing it would make the API demand a sequence.
    scoped = client.post(
        "/v1/users", json={"user_id": "policyless"}, headers=HEADERS
    )
    assert scoped.status_code == 200, scoped.text
    with patch("gateway.services.policy_store.effective_policies", return_value={}):
        resp = client.post(
            "/v1/routing/preferences/rank",
            json={"user_id": "policyless", "examples": [{"prompt": "x", "scores": {CHEAP: 1.0}}]},
            headers=HEADERS,
        )
    assert resp.status_code == 200, resp.text


def test_rank_refuses_a_selector_that_resolves_to_no_provider(client: TestClient) -> None:
    resp = client.post(
        "/v1/routing/preferences/rank",
        json={"user_id": USER, "examples": [{"prompt": "x", "scores": {"nope:whatever": 1.0}}]},
        headers=HEADERS,
    )
    assert resp.status_code == 400
    assert "nope:whatever" in resp.json()["detail"]


def test_rank_reports_an_unreachable_embedding_model_as_a_502(client: TestClient) -> None:
    # The read path degrades to the policy default when embedding fails; the write
    # path has nothing safe to do, so it must say what broke rather than 500.
    with patch(
        "gateway.services.routing.knn.aembedding",
        new=AsyncMock(side_effect=RuntimeError("no credentials for embeddings")),
    ):
        resp = _rank(client, "what is 2 plus 2", {CHEAP: 1.0, STRONG: 1.0})

    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert "openai:text-embedding-3-small" in detail
    assert "OTARI_ROUTER_EMBEDDING_MODEL" in detail


def test_rank_accepts_tied_scores(client: TestClient) -> None:
    # A tie is the whole point: two models that both answered well is exactly when
    # the router should take the cheaper one.
    assert _rank(client, "what is 2 plus 2", {CHEAP: 1.0, STRONG: 1.0}).status_code == 200


@pytest.mark.parametrize("scores", [{}, {CHEAP: 1.5}, {CHEAP: -0.1}])
def test_rank_rejects_out_of_range_or_empty_scores(client: TestClient, scores: dict[str, float]) -> None:
    assert _rank(client, "prompt", scores).status_code == 422


def test_rank_trims_the_task_label_so_the_header_can_reach_the_pool(client: TestClient) -> None:
    # The request side trims `Otari-Router-Task` and treats blank as absent, so a
    # label stored verbatim as " math " created a partition no request could reach
    # and `/status` listed it as real.
    assert _rank(client, "what is 2 plus 2", {CHEAP: 1.0}, task_id="  math  ").status_code == 200
    assert _rank(client, "add 3 and 4", {CHEAP: 1.0}, task_id="   ").status_code == 200

    status = _status(client)
    pools = {pool["task_id"]: pool["records"] for pool in status["tasks"]}
    assert pools == {"math": 1}
    # The blank label filed into the default pool rather than a phantom partition.
    assert status["default_pool"]["records"] == 2


@pytest.mark.parametrize("prompt", ["   ", "\n\t"])
def test_rank_refuses_a_blank_prompt(client: TestClient, prompt: str) -> None:
    # It used to write an audit row, no memory row, and a `recorded` count that
    # disagreed with both.
    assert _rank(client, prompt, {CHEAP: 1.0}).status_code == 422


def test_eviction_keeps_the_store_bounded_without_a_giant_in_list(
    learned_config: GatewayConfig, postgres_url: str
) -> None:
    """Eviction has to work at the *default* cap, not just at a tiny one.

    It used to delete with `id NOT IN (<max_records ids>)`, which binds one host
    parameter per kept row. SQLite caps those at 999 on builds before 3.32, so with
    the default cap of 5000 the query that keeps the store bounded would fail. This
    drives it through the real endpoint and asserts the store is actually trimmed.
    """
    learned_config.router_max_records_per_user = 3
    _run_alembic_migrations(learned_config.database_url)
    engine = create_engine(learned_config.database_url, pool_pre_ping=True)
    app = create_app(learned_config)
    override_get_db, dispose_override = build_async_session_override(learned_config.database_url)
    app.dependency_overrides[get_db] = override_get_db
    try:
        with patch("gateway.services.routing.knn.aembedding", new=_fake_aembedding), TestClient(app) as client:
            _create_user(client)
            resp = client.post(
                "/v1/routing/preferences/rank",
                json={
                    "user_id": USER,
                    "examples": [
                        {"prompt": f"what is {n} plus {n}", "scores": {CHEAP: 1.0, STRONG: 1.0}}
                        for n in range(6)
                    ],
                },
                headers=HEADERS,
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["recorded"] == 6
            # Six written, cap of three: the oldest are gone rather than the write
            # failing or the store growing unbounded.
            assert _status(client)["default_pool"]["records"] <= 3
    finally:
        dispose_override()
        Base.metadata.drop_all(bind=engine)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version CASCADE"))
            conn.commit()
        engine.dispose()


def test_rank_rejects_an_empty_batch(client: TestClient) -> None:
    resp = client.post(
        "/v1/routing/preferences/rank", json={"user_id": USER, "examples": []}, headers=HEADERS
    )
    assert resp.status_code == 422


def test_rank_rejects_an_unknown_user(client: TestClient) -> None:
    resp = client.post(
        "/v1/routing/preferences/rank",
        json={"user_id": "nobody", "examples": [{"prompt": "x", "scores": {CHEAP: 1.0}}]},
        headers=HEADERS,
    )

    assert resp.status_code == 404


def test_status_reports_each_pool_independently(client: TestClient) -> None:
    _teach(client)
    _teach(client, task_id="math")
    assert _rank(client, "one more", {CHEAP: 1.0}, task_id="support").status_code == 200

    status = _status(client)

    # The default pool counts every record the user has, labelled or not.
    assert status["default_pool"] == {"records": 9, "warm": True}
    pools = {pool["task_id"]: pool for pool in status["tasks"]}
    assert pools["math"] == {"task_id": "math", "records": 4, "warm": True}
    assert pools["support"] == {"task_id": "support", "records": 1, "warm": False}
    # And it names the policies that depend on this memory.
    learned = {policy["name"] for policy in status["policies"]}
    assert {"smart", "smart-failover"} <= learned
    assert status["seed_count"] == 4


def test_the_preference_surfaces_require_the_master_key(client: TestClient) -> None:
    # Which model serves a caller is an operator decision, so a user key must not
    # reach the teaching surfaces at all.
    key = client.post("/v1/keys", json={"key_name": "plain", "user_id": USER}, headers=HEADERS)
    assert key.status_code == 200, key.text
    user_headers = {API_KEY_HEADER: f"Bearer {key.json()['key']}"}

    assert client.get("/v1/routing/status", params={"user_id": USER}, headers=user_headers).status_code == 401
    assert (
        client.post(
            "/v1/routing/preferences/rank",
            json={"user_id": USER, "examples": [{"prompt": "x", "scores": {CHEAP: 1.0}}]},
            headers=user_headers,
        ).status_code
        == 401
    )


# -- authoring guardrails --------------------------------------------------


def test_a_stored_learned_policy_needs_priced_candidates(client: TestClient) -> None:
    # A router scores by cost, so an unpriced candidate makes it decline forever.
    resp = client.post(
        "/v1/routing/policies",
        json={
            "name": "unpriced",
            "spec": {
                "select": [
                    {"router": "knn", "candidates": ["openai:gpt-4.1-nano-unpriced", STRONG]},
                    {"default": STRONG},
                ]
            },
        },
        headers=HEADERS,
    )

    assert resp.status_code == 400
    assert "no pricing" in resp.json()["detail"]


def test_a_stored_learned_policy_routes_like_a_configured_one(client: TestClient) -> None:
    saved = client.post(
        "/v1/routing/policies",
        json={
            "name": "stored-smart",
            "spec": {
                "select": [{"router": "knn", "candidates": [CHEAP, STRONG]}, {"default": STRONG}],
            },
        },
        headers=HEADERS,
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["is_dynamic"] is True
    _teach(client)

    resp, calls = _chat(client, "stored-smart", "what is 2 plus 2")

    assert resp.status_code == 200, resp.text
    assert calls == [CHEAP]


def test_explain_says_the_router_decides_at_request_time(client: TestClient) -> None:
    # Explain dispatches nothing, so it cannot rank. Showing the decline path plus
    # the pool beats showing a one-candidate plan that looks like a broken router.
    resp = client.post("/v1/routing/policies/explain", json={"name": "smart"}, headers=HEADERS)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["router_backend"] == "knn"
    assert body["router_candidates"] == [CHEAP, STRONG]
    assert body["selection_reason"] == "default"
    assert body["is_dynamic"] is True


def test_a_learned_policy_is_refused_in_hybrid_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # Hybrid mode resolves models upstream, so a local policy name is not a model
    # the platform knows. Guarded here because the router is the newest reason
    # someone might reach for a policy.
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    config = GatewayConfig(
        mode="hybrid",
        platform={"base_url": "http://platform.test/api/v1"},
        routing=RoutingConfig.model_validate(
            {
                "policies": {
                    "smart": {
                        "select": [{"router": "knn", "candidates": [CHEAP, STRONG]}, {"default": STRONG}]
                    }
                }
            }
        ),
    )
    app = create_app(config)
    with TestClient(app) as hybrid_client:
        resp = hybrid_client.post(
            "/v1/chat/completions",
            json={"model": "smart", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer user_token"},
        )

    assert resp.status_code == 400
    assert "hybrid mode" in resp.json()["detail"]


def test_a_policy_with_no_learned_pool_reports_zero(client: TestClient) -> None:
    status = _status(client)

    assert status["default_pool"] == {"records": 0, "warm": False}
    assert status["tasks"] == []
    assert status["embedding_model"]
    assert status["granularity"] == "trace_sticky"


def test_declining_is_recorded_on_the_usage_row_as_default(client: TestClient) -> None:
    # "The router chose the strong model" and "the router did not run" have to stay
    # distinguishable after the fact, which is what selection_reason carries.
    _teach(client)
    hard, _ = _chat(client, "smart", "prove why entropy increases")
    assert hard.status_code == 200
    routed = [row for row in _usage_rows(client) if row["status"] == "success"]
    assert routed[0]["selection_reason"] == "router:knn"

    off, _ = _chat(client, "smart", "what is 2 plus 2", headers={"Otari-Router": "off"})
    assert off.status_code == 200
    rows = [row for row in _usage_rows(client) if row["status"] == "success"]
    assert rows[0]["selection_reason"] == "default"


def test_a_static_policy_is_unaffected_by_the_router_step(client: TestClient) -> None:
    # The router step must cost a policy with no router nothing at all: same plan,
    # same reason, no embedding call (the fake would answer, but nothing asks).
    saved = client.post(
        "/v1/routing/policies",
        json={"name": "plain", "spec": {"select": [{"default": STRONG}], "on_failure": [OTHER]}},
        headers=HEADERS,
    )
    assert saved.status_code == 200, saved.text

    with patch("gateway.services.routing.knn.aembedding", new=AsyncMock(side_effect=AssertionError("no embed"))):
        resp, calls = _chat(client, "plain")

    assert resp.status_code == 200, resp.text
    assert calls == [STRONG]
