"""End-to-end behavior for routing policies (mozilla-ai/otari#463).

A policy is a model name callers use like any other. It decides which real model
serves the request (``select``), what is tried after a retryable failure
(``on_failure``), and which guardrails always run.

The invariants worth defending, and why:

* A one-candidate policy answers exactly as naming its target directly does. That
  is what makes "an alias is a one-target policy" a true statement rather than a
  slogan, and it keeps every existing behavior reachable through the new surface.
* Failover writes **one** usage row, for the candidate that actually served, and
  settles the reservation once. Settling per attempt would double-charge.
* Billing keys on the resolved target while the response says the policy name.
* A policy never routes a caller to a model their key is not allowed to use.
"""

from collections.abc import Generator
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
    CreateEmbeddingResponse,
    Embedding,
    Usage,
)
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from gateway.core.config import API_KEY_HEADER, GatewayConfig
from gateway.db import Base, get_db
from gateway.main import create_app
from gateway.models.routing import RoutingConfig

from .conftest import _run_alembic_migrations, build_async_session_override

HEADERS = {API_KEY_HEADER: "Bearer test-master-key"}


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


def _message_response() -> Any:
    """A minimal valid Anthropic Message, for the /v1/messages dispatch path."""
    from any_llm.types.messages import MessageResponse, MessageUsage, TextBlock

    return MessageResponse(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-haiku-4-5",
        content=[TextBlock(type="text", text="ok", citations=None)],
        stop_reason=cast(Any, "end_turn"),
        stop_sequence=None,
        usage=MessageUsage(
            input_tokens=5,
            output_tokens=2,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            cache_creation=None,
            server_tool_use=None,
            service_tier=None,
        ),
        container=None,
    )


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://upstream")
    return httpx.HTTPStatusError(str(status), request=request, response=httpx.Response(status, request=request))


@pytest.fixture
def routing_config(postgres_url: str) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        providers={
            "openai": {"api_key": "sk-openai"},
            "anthropic": {"api_key": "sk-ant"},
        },
        routing=RoutingConfig.model_validate(
            {
                "policies": {
                    # Plain failover: no conditions, so no budget read is needed.
                    "fast": {
                        "select": [{"default": "openai:gpt-5-mini"}],
                        "on_failure": ["anthropic:claude-haiku-4-5"],
                    },
                    # A one-candidate policy: the alias-equivalent shape.
                    "solo": {"select": [{"default": "openai:gpt-5-mini"}]},
                    # Budget tier-down.
                    "thrifty": {
                        "select": [
                            {"when": {"budget_used_pct": {"gte": 80}}, "target": "openai:gpt-5-nano"},
                            {"default": "openai:gpt-5-mini"},
                        ],
                    },
                }
            }
        ),
    )


def _build_client(config: GatewayConfig) -> Generator[TestClient]:
    _run_alembic_migrations(config.database_url)
    engine = create_engine(config.database_url, pool_pre_ping=True)
    app = create_app(config)
    override_get_db, dispose_override = build_async_session_override(config.database_url)
    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        dispose_override()
        Base.metadata.drop_all(bind=engine)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version CASCADE"))
            conn.commit()
        engine.dispose()


@pytest.fixture
def client(routing_config: GatewayConfig) -> Generator[TestClient]:
    yield from _build_client(routing_config)


def _create_user(client: TestClient, user_id: str = "test-user", **extra: Any) -> None:
    resp = client.post("/v1/users", json={"user_id": user_id, **extra}, headers=HEADERS)
    assert resp.status_code == 200, resp.text


def _chat(client: TestClient, model: str, **extra: Any) -> Any:
    return client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "hi"}], "user": "test-user", **extra},
        headers=HEADERS,
    )


def _awaited_model(mock: AsyncMock) -> str:
    """The `model` kwarg of the single provider call, asserting there was one."""
    assert mock.await_args is not None, "the provider was never called"
    model: str = mock.await_args.kwargs["model"]
    return model


def _usage_rows(client: TestClient) -> list[dict[str, Any]]:
    resp = client.get("/v1/usage", headers=HEADERS)
    assert resp.status_code == 200, resp.text
    payload: Any = resp.json()
    rows: list[dict[str, Any]] = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    return rows


# ---------------------------------------------------------------------------
# The happy path, and the alias-equivalence invariant
# ---------------------------------------------------------------------------


def test_policy_routes_to_its_default_target(client: TestClient) -> None:
    _create_user(client)
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion("gpt-5-mini"))) as mock:
        resp = _chat(client, "fast")

    assert resp.status_code == 200, resp.text
    # The caller sees the policy name, never the underlying model.
    assert resp.json()["model"] == "fast"
    assert _awaited_model(mock) == "openai:gpt-5-mini"
    # Billing keys on the resolved target.
    rows = _usage_rows(client)
    assert len(rows) == 1
    assert rows[0]["model"] == "gpt-5-mini"
    assert rows[0]["provider"] == "openai"


def test_single_candidate_policy_matches_naming_the_model_directly(client: TestClient) -> None:
    """The compatibility claim, made testable: one candidate must behave exactly
    like the plain model, including the failure status.
    """
    _create_user(client)
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=_http_error(429))):
        via_policy = _chat(client, "solo")
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=_http_error(429))):
        direct = _chat(client, "openai:gpt-5-mini")

    assert via_policy.status_code == direct.status_code == 429


# ---------------------------------------------------------------------------
# Failover
# ---------------------------------------------------------------------------


def test_failover_serves_the_next_candidate_and_bills_it_once(client: TestClient) -> None:
    _create_user(client)
    calls: list[str] = []

    async def flaky(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs["model"])
        if kwargs["model"] == "openai:gpt-5-mini":
            raise _http_error(503)
        return _completion("claude-haiku-4-5")

    with patch("gateway.api.routes.chat.acompletion", new=flaky):
        resp = _chat(client, "fast")

    assert resp.status_code == 200, resp.text
    assert calls == ["openai:gpt-5-mini", "anthropic:claude-haiku-4-5"]
    # Relabeled to the policy, so a fallover is invisible to the caller's code.
    assert resp.json()["model"] == "fast"

    # Two rows: the attempt that served, plus the failure the policy absorbed.
    rows = _usage_rows(client)
    served = [r for r in rows if r["status"] == "success"]
    absorbed = [r for r in rows if r["status"] == "absorbed"]
    assert len(served) == 1
    assert len(absorbed) == 1

    # The serving row is billed, and keyed on the model that actually served.
    assert served[0]["model"] == "claude-haiku-4-5"
    assert served[0]["provider"] == "anthropic"
    assert served[0]["attempt_position"] == 2
    assert served[0]["attempt_count"] == 2
    assert served[0]["policy_name"] == "fast"
    assert served[0]["selection_reason"] == "on_failure"

    # The absorbed row records what was tried first and why it went, and is
    # deliberately not an error: the request was served.
    assert absorbed[0]["model"] == "gpt-5-mini"
    assert absorbed[0]["attempt_position"] == 1
    assert absorbed[0]["counts_toward_budget"] is False

    # Both belong to one request, which is what makes the history reconstructable.
    assert served[0]["request_group_id"] == absorbed[0]["request_group_id"]
    assert served[0]["request_group_id"] is not None


def test_all_candidates_failing_is_a_generic_502(client: TestClient) -> None:
    """A multi-candidate fallthrough must not attribute one provider's status to
    the whole plan.
    """
    _create_user(client)
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=_http_error(503))):
        resp = _chat(client, "fast")

    assert resp.status_code == 502


def test_a_malformed_request_does_not_burn_the_chain(client: TestClient) -> None:
    """A 400 would be rejected by every provider, so only one call should happen."""
    _create_user(client)
    calls: list[str] = []

    async def bad_request(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs["model"])
        raise _http_error(400)

    with patch("gateway.api.routes.chat.acompletion", new=bad_request):
        resp = _chat(client, "fast")

    assert resp.status_code == 400
    assert calls == ["openai:gpt-5-mini"]


def test_an_auth_failure_does_not_fail_over(client: TestClient) -> None:
    """The operator owns both keys here, so a 401 is their misconfiguration.
    Failing over would move the spend to another provider and hide it.
    """
    _create_user(client)
    calls: list[str] = []

    async def unauthorized(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs["model"])
        raise _http_error(401)

    with patch("gateway.api.routes.chat.acompletion", new=unauthorized):
        resp = _chat(client, "fast")
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=_http_error(401))):
        direct = _chat(client, "openai:gpt-5-mini")

    assert calls == ["openai:gpt-5-mini"]
    # Whatever a provider 401 maps to for a plain model, it maps to the same
    # through a policy. (It is deliberately not surfaced as a 401: the caller's
    # own key was fine, and saying "unauthorized" would point them at the wrong
    # thing.)
    assert resp.status_code == direct.status_code


def test_streaming_fails_over_before_any_bytes_are_flushed(client: TestClient) -> None:
    _create_user(client)
    calls: list[str] = []

    async def flaky_stream(**kwargs: Any) -> Any:
        calls.append(kwargs["model"])
        if kwargs["model"] == "openai:gpt-5-mini":
            raise _http_error(503)

        async def chunks() -> Any:
            from any_llm.types.completion import ChatCompletionChunk, ChoiceDelta, ChunkChoice

            yield ChatCompletionChunk(
                id="c1",
                choices=[ChunkChoice(delta=ChoiceDelta(content="hi"), index=0, finish_reason=None)],
                created=0,
                model="claude-haiku-4-5",
                object="chat.completion.chunk",
                usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        return chunks()

    with patch("gateway.api.routes.chat.acompletion", new=flaky_stream):
        resp = _chat(client, "fast", stream=True)
        body = resp.text

    assert resp.status_code == 200
    assert calls == ["openai:gpt-5-mini", "anthropic:claude-haiku-4-5"]
    # Chunks are relabeled to the policy name too.
    assert '"model":"fast"' in body.replace(" ", "")


# ---------------------------------------------------------------------------
# Model access: a policy must not become a way around an allow-list
# ---------------------------------------------------------------------------


def test_a_policy_cannot_route_to_a_model_the_key_may_not_use(client: TestClient) -> None:
    """The chain is filtered by the caller's allow-list at compile time, so a
    fallover can never reach a forbidden model.
    """
    _create_user(client)
    key_resp = client.post(
        "/v1/keys",
        json={"user_id": "test-user", "allowed_models": ["openai:gpt-5-mini"]},
        headers=HEADERS,
    )
    assert key_resp.status_code == 200, key_resp.text
    scoped = {API_KEY_HEADER: f"Bearer {key_resp.json()['key']}"}

    calls: list[str] = []

    async def flaky(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs["model"])
        raise _http_error(503)

    with patch("gateway.api.routes.chat.acompletion", new=flaky):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "fast", "messages": [{"role": "user", "content": "hi"}]},
            headers=scoped,
        )

    # The anthropic candidate was dropped before dispatch, so only the permitted
    # one was tried. That is the security-relevant assertion: a chain must never
    # reach a model the key may not use.
    assert calls == ["openai:gpt-5-mini"]
    # And with one candidate left, the answer matches naming that model directly.
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=_http_error(503))):
        direct = client.post(
            "/v1/chat/completions",
            json={"model": "openai:gpt-5-mini", "messages": [{"role": "user", "content": "hi"}]},
            headers=scoped,
        )
    assert resp.status_code == direct.status_code


def test_a_policy_with_no_permitted_candidate_is_refused_without_naming_its_targets(
    client: TestClient,
) -> None:
    _create_user(client)
    key_resp = client.post(
        "/v1/keys",
        json={"user_id": "test-user", "allowed_models": ["openai:some-other-model"]},
        headers=HEADERS,
    )
    assert key_resp.status_code == 200, key_resp.text
    scoped = {API_KEY_HEADER: f"Bearer {key_resp.json()['key']}"}

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "fast", "messages": [{"role": "user", "content": "hi"}]},
        headers=scoped,
    )

    assert resp.status_code == 403
    detail = resp.json()["detail"]
    assert "fast" in detail
    # A policy exists partly to keep its targets off the wire, so the caller-facing
    # message must not enumerate them even while refusing.
    assert "gpt-5-mini" not in detail
    assert "claude-haiku" not in detail


# ---------------------------------------------------------------------------
# Budget-conditional selection
# ---------------------------------------------------------------------------


def test_tier_down_fires_once_the_budget_threshold_is_crossed(
    client: TestClient, routing_config: GatewayConfig
) -> None:
    budget = client.post("/v1/budgets", json={"max_budget": 1.0}, headers=HEADERS)
    assert budget.status_code == 200, budget.text
    budget_id = budget.json()["budget_id"]
    _create_user(client, budget_id=budget_id)

    # Under the threshold: the default candidate serves.
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion("gpt-5-mini"))) as mock:
        assert _chat(client, "thrifty").status_code == 200
    assert _awaited_model(mock) == "openai:gpt-5-mini"

    # Push committed spend past 80% of the cap. There is no API to set spend
    # (correctly, it is an accounting field), so this writes it directly.
    engine = create_engine(routing_config.database_url, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("UPDATE users SET spend = 0.9 WHERE user_id = 'test-user'"))
        conn.commit()
    engine.dispose()
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion("gpt-5-nano"))) as mock:
        assert _chat(client, "thrifty").status_code == 200
    assert _awaited_model(mock) == "openai:gpt-5-nano"


def test_a_user_without_a_budget_uses_the_default_candidate(client: TestClient) -> None:
    """An undefined percentage must never match, and must never raise: "no budget
    configured" cannot become a 500 on every request through the policy.
    """
    _create_user(client)
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion("gpt-5-mini"))) as mock:
        resp = _chat(client, "thrifty")

    assert resp.status_code == 200
    assert _awaited_model(mock) == "openai:gpt-5-mini"


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_policies_are_listed_as_models(client: TestClient) -> None:
    resp = client.get("/v1/models", headers=HEADERS)
    assert resp.status_code == 200
    ids = {model["id"] for model in resp.json()["data"]}
    assert {"fast", "solo", "thrifty"} <= ids


# ---------------------------------------------------------------------------
# Reach on the other model-taking endpoints
# ---------------------------------------------------------------------------


def test_a_static_policy_resolves_on_a_non_completion_endpoint(client: TestClient) -> None:
    """"An alias is a one-target policy" has to be true everywhere, not just on
    the completion routes, or the two concepts are not actually the same thing.
    """
    _create_user(client)
    embedding_response = CreateEmbeddingResponse(
        data=[Embedding(embedding=[0.1, 0.2], index=0, object="embedding")],
        model="gpt-5-mini",
        object="list",
        usage=Usage(prompt_tokens=5, total_tokens=5),
    )
    with patch(
        "gateway.api.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=embedding_response
    ) as mock:
        resp = client.post(
            "/v1/embeddings",
            json={"model": "solo", "input": "hello", "user": "test-user"},
            headers=HEADERS,
        )

    assert resp.status_code == 200, resp.text
    # It resolved to the target and dispatched there, rather than being rejected
    # as an unknown model...
    assert mock.await_args is not None
    assert mock.await_args.kwargs["model"] == "gpt-5-mini"
    assert mock.await_args.kwargs["provider"] == "openai"
    # ...and the caller still sees the name they sent.
    assert resp.json()["model"] == "solo"


def test_a_dynamic_policy_is_not_a_model_name_outside_the_completion_routes(client: TestClient) -> None:
    """Its candidate depends on request state the synchronous resolution path
    cannot see, so there is no honest single target to hand back. Refusing beats
    silently serving the default and calling it the policy.
    """
    _create_user(client)
    resp = client.post(
        "/v1/embeddings",
        json={"model": "thrifty", "input": "hello", "user": "test-user"},
        headers=HEADERS,
    )

    assert resp.status_code == 400
    assert "thrifty" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Stored policies: the management API the dashboard drives
# ---------------------------------------------------------------------------


def _spec(default: str, on_failure: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"select": [{"default": default}]}
    if on_failure:
        body["on_failure"] = on_failure
    body.update(extra)
    return body


def test_stored_policy_takes_effect_without_a_restart(client: TestClient) -> None:
    """The whole point of storing policies: an operator creates one through the API
    and the very next request can use it.
    """
    _create_user(client)
    created = client.post(
        "/v1/routing/policies",
        json={"name": "runtime", "spec": _spec("openai:gpt-5-mini", ["anthropic:claude-haiku-4-5"])},
        headers=HEADERS,
    )
    assert created.status_code == 200, created.text
    assert created.json()["source"] == "stored"
    assert created.json()["is_dynamic"] is False

    calls: list[str] = []

    async def flaky(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs["model"])
        if kwargs["model"] == "openai:gpt-5-mini":
            raise _http_error(503)
        return _completion("claude-haiku-4-5")

    with patch("gateway.api.routes.chat.acompletion", new=flaky):
        resp = _chat(client, "runtime")

    assert resp.status_code == 200, resp.text
    assert calls == ["openai:gpt-5-mini", "anthropic:claude-haiku-4-5"]
    assert resp.json()["model"] == "runtime"


def test_listing_shows_stored_and_config_policies_together(client: TestClient) -> None:
    client.post("/v1/routing/policies", json={"name": "runtime", "spec": _spec("openai:gpt-5-mini")}, headers=HEADERS)
    resp = client.get("/v1/routing/policies", headers=HEADERS)

    assert resp.status_code == 200, resp.text
    by_name = {item["name"]: item for item in resp.json()}
    assert by_name["runtime"]["source"] == "stored"
    assert by_name["fast"]["source"] == "config"
    # `thrifty` has a condition, so it has no single target.
    assert by_name["thrifty"]["is_dynamic"] is True


def test_a_stored_policy_can_be_deleted_and_stops_resolving(client: TestClient) -> None:
    _create_user(client)
    client.post("/v1/routing/policies", json={"name": "temp", "spec": _spec("openai:gpt-5-mini")}, headers=HEADERS)
    assert client.delete("/v1/routing/policies/temp", headers=HEADERS).status_code == 204

    resp = _chat(client, "temp")
    assert resp.status_code == 400
    assert "temp" in resp.json()["detail"]


def test_a_config_policy_cannot_be_deleted_through_the_api(client: TestClient) -> None:
    resp = client.delete("/v1/routing/policies/fast", headers=HEADERS)
    assert resp.status_code == 404
    assert "config.yml" in resp.json()["detail"]


def test_a_global_stored_policy_may_not_shadow_a_config_one(client: TestClient) -> None:
    """Config wins during resolution, so storing this would be dead config. Saying
    no is the only answer that does not lie about what the gateway will do.
    """
    resp = client.post(
        "/v1/routing/policies", json={"name": "fast", "spec": _spec("openai:gpt-5-mini")}, headers=HEADERS
    )
    assert resp.status_code == 400
    assert "config.yml" in resp.json()["detail"]


def test_a_user_scoped_policy_overrides_a_config_one_for_that_user_only(client: TestClient) -> None:
    _create_user(client)
    _create_user(client, "other-user")
    scoped = client.post(
        "/v1/routing/policies",
        json={"name": "fast", "spec": _spec("anthropic:claude-haiku-4-5"), "user_id": "test-user"},
        headers=HEADERS,
    )
    assert scoped.status_code == 200, scoped.text

    with patch(
        "gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion("claude-haiku-4-5"))
    ) as mock:
        assert _chat(client, "fast").status_code == 200
    assert _awaited_model(mock) == "anthropic:claude-haiku-4-5"

    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion("gpt-5-mini"))) as mock:
        other = client.post(
            "/v1/chat/completions",
            json={"model": "fast", "messages": [{"role": "user", "content": "hi"}], "user": "other-user"},
            headers=HEADERS,
        )
        assert other.status_code == 200, other.text
    assert _awaited_model(mock) == "openai:gpt-5-mini"


def test_an_invalid_spec_is_refused_with_field_level_errors(client: TestClient) -> None:
    """A form has to be able to point at the field that is wrong, so the schema's
    own messages are surfaced rather than flattened into one string.
    """
    resp = client.post(
        "/v1/routing/policies",
        json={"name": "broken", "spec": {"select": [{"target": "openai:gpt-5-mini"}]}},
        headers=HEADERS,
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["message"].startswith("routing policy 'broken'")
    assert detail["errors"], "the pydantic errors must be surfaced for the form to bind"


def test_a_stored_policy_may_not_point_at_an_alias(client: TestClient) -> None:
    alias = client.post("/v1/aliases", json={"name": "cheap", "target": "openai:gpt-5-nano"}, headers=HEADERS)
    assert alias.status_code == 200, alias.text
    resp = client.post(
        "/v1/routing/policies", json={"name": "chained", "spec": _spec("cheap:x")}, headers=HEADERS
    )
    assert resp.status_code == 400
    assert "chaining" in resp.json()["detail"]


def test_a_policy_may_not_reuse_an_alias_name(client: TestClient) -> None:
    alias = client.post("/v1/aliases", json={"name": "taken", "target": "openai:gpt-5-nano"}, headers=HEADERS)
    assert alias.status_code == 200, alias.text
    resp = client.post(
        "/v1/routing/policies", json={"name": "taken", "spec": _spec("openai:gpt-5-mini")}, headers=HEADERS
    )
    assert resp.status_code == 400
    assert "alias" in resp.json()["detail"]


def test_writing_a_policy_for_an_unknown_user_is_a_404(client: TestClient) -> None:
    resp = client.post(
        "/v1/routing/policies",
        json={"name": "scoped", "spec": _spec("openai:gpt-5-mini"), "user_id": "nobody"},
        headers=HEADERS,
    )
    assert resp.status_code == 404


def test_policy_management_requires_the_master_key(client: TestClient) -> None:
    _create_user(client)
    key_resp = client.post("/v1/keys", json={"user_id": "test-user"}, headers=HEADERS)
    assert key_resp.status_code == 200, key_resp.text
    caller = {API_KEY_HEADER: f"Bearer {key_resp.json()['key']}"}

    # Only an operator may decide which models a name reaches; otherwise a caller
    # could widen their own access by writing a policy.
    assert client.get("/v1/routing/policies", headers=caller).status_code in (401, 403)
    assert (
        client.post(
            "/v1/routing/policies", json={"name": "sneaky", "spec": _spec("openai:gpt-5-mini")}, headers=caller
        ).status_code
        in (401, 403)
    )


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------


def test_explain_returns_the_plan_for_a_saved_policy(client: TestClient) -> None:
    resp = client.post("/v1/routing/policies/explain", json={"name": "fast"}, headers=HEADERS)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [c["dispatch_model"] for c in body["candidates"]] == [
        "openai:gpt-5-mini",
        "anthropic:claude-haiku-4-5",
    ]
    assert body["selection_reason"] == "default"


def test_explain_checks_an_unsaved_draft(client: TestClient) -> None:
    """Authoring-time validation is only possible against a draft, so the endpoint
    accepts a spec that has not been stored.
    """
    resp = client.post(
        "/v1/routing/policies/explain",
        json={"name": "draft", "spec": _spec("openai:gpt-5-mini", ["anthropic:claude-haiku-4-5"])},
        headers=HEADERS,
    )
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["candidates"]) == 2


def test_explain_reports_dropped_candidates_with_reasons(client: TestClient) -> None:
    """The reason this endpoint exists: a three-model chain can compile down to one
    attempt, and an author needs to see that before an outage does.
    """
    resp = client.post(
        "/v1/routing/policies/explain",
        json={"name": "fast", "allowed_models": ["anthropic:claude-haiku-4-5"]},
        headers=HEADERS,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["candidates"]) == 1
    assert body["dropped"][0]["selector"] == "openai:gpt-5-mini"
    assert body["dropped"][0]["reason"] == "not_allowed"


def test_explain_simulates_a_budget_threshold(client: TestClient) -> None:
    resp = client.post(
        "/v1/routing/policies/explain",
        json={"name": "thrifty", "budget_used_pct": 85},
        headers=HEADERS,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["candidates"][0]["dispatch_model"] == "openai:gpt-5-nano"
    assert body["selection_reason"] == "condition:budget_used_pct"


def test_explain_needs_at_least_a_name_or_a_spec(client: TestClient) -> None:
    assert client.post("/v1/routing/policies/explain", json={}, headers=HEADERS).status_code == 400


def test_explain_prefers_a_draft_over_the_saved_policy_of_the_same_name(client: TestClient) -> None:
    """The normal editing flow: the operator is looking at `fast` and wants to know
    what their unsaved edit would do, so both name and spec are sent.
    """
    resp = client.post(
        "/v1/routing/policies/explain",
        json={"name": "fast", "spec": _spec("anthropic:claude-haiku-4-5")},
        headers=HEADERS,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["name"] == "fast"
    # The draft's single candidate, not the saved policy's two.
    assert [c["dispatch_model"] for c in body["candidates"]] == ["anthropic:claude-haiku-4-5"]


def test_an_absorbed_failure_does_not_count_as_an_error_or_an_extra_request(client: TestClient) -> None:
    """The regression this guards: every error metric counts `status == "error"`,
    and request volume is the denominator of all of them. If a recovered attempt
    counted as either, a working fallback chain would report an outage, which is
    the exact inverse of what the feature is for.
    """
    _create_user(client)

    async def flaky(**kwargs: Any) -> ChatCompletion:
        if kwargs["model"] == "openai:gpt-5-mini":
            raise _http_error(503)
        return _completion("claude-haiku-4-5")

    with patch("gateway.api.routes.chat.acompletion", new=flaky):
        assert _chat(client, "fast").status_code == 200

    summary = client.get("/v1/usage/summary", headers=HEADERS)
    assert summary.status_code == 200, summary.text
    totals = summary.json()["totals"]

    assert totals["error_count"] == 0
    assert totals["request_count"] == 1


def test_a_failed_request_still_says_which_policy_it_went_through(client: TestClient) -> None:
    """A failure is when an operator most needs the attribution, so the error row
    carries it too, attributed to the last candidate the walk reached.
    """
    _create_user(client)
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=_http_error(503))):
        assert _chat(client, "fast").status_code == 502

    rows = _usage_rows(client)
    errors = [r for r in rows if r["status"] == "error"]
    assert len(errors) == 1
    assert errors[0]["policy_name"] == "fast"
    assert errors[0]["attempt_position"] == 2
    assert errors[0]["attempt_count"] == 2


def test_a_stored_policy_is_listed_in_the_model_catalog(client: TestClient) -> None:
    """A policy created through the API has to appear as a model, or one made in
    the dashboard would work when called and be invisible in the catalog.
    """
    created = client.post(
        "/v1/routing/policies",
        json={"name": "listed", "spec": _spec("openai:gpt-5-mini")},
        headers=HEADERS,
    )
    assert created.status_code == 200, created.text

    resp = client.get("/v1/models", headers=HEADERS)
    assert resp.status_code == 200
    entries = {model["id"]: model for model in resp.json()["data"]}
    assert "listed" in entries
    assert entries["listed"]["owned_by"] == "otari"


def test_a_dynamic_stored_policy_reports_no_single_price(client: TestClient) -> None:
    created = client.post(
        "/v1/routing/policies",
        json={
            "name": "listed-dynamic",
            "spec": {
                "select": [
                    {"when": {"budget_used_pct": {"gte": 70}}, "target": "openai:gpt-5-nano"},
                    {"default": "openai:gpt-5-mini"},
                ]
            },
        },
        headers=HEADERS,
    )
    assert created.status_code == 200, created.text

    resp = client.get("/v1/models", headers=HEADERS)
    entries = {model["id"]: model for model in resp.json()["data"]}
    assert entries["listed-dynamic"]["pricing"] is None
    # Reuses the existing field rather than inventing a second way to say it.
    assert entries["listed-dynamic"]["pricing_source"] == "dynamic"


# ---------------------------------------------------------------------------
# Mandated guardrails actually reach the guardrail runner
#
# The regression this guards is that the whole feature was once a no-op: the
# compiler built the guardrail list and no route ever read it, so the schema, the
# CLI, the dashboard, and the docs all described enforcement that never happened.
# Asserting a 200 is not enough, because a guardrail that never runs also returns
# 200. These assert the runner was actually handed the mandate.
# ---------------------------------------------------------------------------


@pytest.fixture
def guarded_client(routing_config: GatewayConfig) -> Generator[TestClient]:
    guarded = routing_config.model_copy(
        update={
            "routing": RoutingConfig.model_validate(
                {
                    "policies": {
                        "guarded": {
                            "select": [{"default": "openai:gpt-5-mini"}],
                            "guardrails": [
                                {"profile": "prompt-injection", "mode": "block", "on_unavailable": "block"}
                            ],
                        }
                    }
                }
            )
        }
    )
    yield from _build_client(guarded)


def test_a_policy_guardrail_is_handed_to_the_guardrail_runner(guarded_client: TestClient) -> None:
    _create_user(guarded_client)
    seen: list[Any] = []

    async def capture(guardrails: Any, text: str, **kwargs: Any) -> None:
        seen.append(guardrails)

    with (
        patch("gateway.api.routes._pipeline.apply_input_guardrails", new=capture),
        patch("gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion("gpt-5-mini"))),
    ):
        resp = guarded_client.post(
            "/v1/chat/completions",
            json={"model": "guarded", "messages": [{"role": "user", "content": "hi"}], "user": "test-user"},
            headers=HEADERS,
        )

    assert resp.status_code == 200, resp.text
    assert len(seen) == 1
    assert seen[0] is not None, "the policy's guardrail never reached the runner"
    assert [g.profile for g in seen[0]] == ["prompt-injection"]
    assert seen[0][0].mode == "block"


def test_a_caller_cannot_weaken_a_policy_guardrail_over_the_wire(guarded_client: TestClient) -> None:
    _create_user(guarded_client)
    seen: list[Any] = []

    async def capture(guardrails: Any, text: str, **kwargs: Any) -> None:
        seen.append(guardrails)

    with (
        patch("gateway.api.routes._pipeline.apply_input_guardrails", new=capture),
        patch("gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion("gpt-5-mini"))),
    ):
        resp = guarded_client.post(
            "/v1/chat/completions",
            json={
                "model": "guarded",
                "messages": [{"role": "user", "content": "hi"}],
                "user": "test-user",
                "guardrails": [{"profile": "prompt-injection", "mode": "monitor", "on_unavailable": "monitor"}],
            },
            headers=HEADERS,
        )

    assert resp.status_code == 200, resp.text
    assert [g.profile for g in seen[0]] == ["prompt-injection"]
    assert seen[0][0].mode == "block"
    assert seen[0][0].on_unavailable == "block"


def test_a_blocking_policy_guardrail_refuses_the_request(guarded_client: TestClient) -> None:
    """End to end: a mandated block guardrail that flags stops the provider call."""
    _create_user(guarded_client)
    provider = AsyncMock(return_value=_completion("gpt-5-mini"))

    async def flagged(guardrails: Any, input_text: str, *, default_url: str | None) -> Any:
        from gateway.services.guardrails import GuardrailResult, GuardrailVerdict

        return GuardrailVerdict(
            results=[GuardrailResult(profile=g.profile, mode=g.mode, valid=False) for g in guardrails]
        )

    with (
        patch("gateway.api.routes._helpers.run_input_guardrails", new=flagged),
        patch("gateway.api.routes.chat.acompletion", new=provider),
    ):
        resp = guarded_client.post(
            "/v1/chat/completions",
            json={"model": "guarded", "messages": [{"role": "user", "content": "hi"}], "user": "test-user"},
            headers=HEADERS,
        )

    assert resp.status_code == 403, resp.text
    provider.assert_not_awaited()


# ---------------------------------------------------------------------------
# Reach and attribution on every completion endpoint
# ---------------------------------------------------------------------------


def test_a_terminal_failure_names_the_candidate_that_actually_failed(client: TestClient) -> None:
    """A 401 does not fail over, so the request stops on candidate 1. Attributing
    it to the end of the plan would blame a provider that was never called, in a
    row that feeds the by-provider breakdown and the error taxonomy.
    """
    _create_user(client)
    calls: list[str] = []

    async def unauthorized(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs["model"])
        raise _http_error(401)

    with patch("gateway.api.routes.chat.acompletion", new=unauthorized):
        _chat(client, "fast")

    assert calls == ["openai:gpt-5-mini"]
    errors = [r for r in _usage_rows(client) if r["status"] == "error"]
    assert len(errors) == 1
    assert errors[0]["provider"] == "openai"
    assert errors[0]["model"] == "gpt-5-mini"
    assert errors[0]["attempt_position"] == 1


def test_a_streamed_fallover_correlates_both_of_its_rows(client: TestClient) -> None:
    """request_group_id exists to tie an absorbed attempt to the row that served.
    On the streaming path the serving row once carried no attribution at all, so
    the correlation the migration is for did not happen.
    """
    _create_user(client)

    async def flaky_stream(**kwargs: Any) -> Any:
        if kwargs["model"] == "openai:gpt-5-mini":
            raise _http_error(503)

        async def chunks() -> Any:
            from any_llm.types.completion import ChatCompletionChunk, ChoiceDelta, ChunkChoice

            yield ChatCompletionChunk(
                id="c1",
                choices=[ChunkChoice(delta=ChoiceDelta(content="hi"), index=0, finish_reason=None)],
                created=0,
                model="claude-haiku-4-5",
                object="chat.completion.chunk",
                usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        return chunks()

    with patch("gateway.api.routes.chat.acompletion", new=flaky_stream):
        assert _chat(client, "fast", stream=True).status_code == 200

    rows = _usage_rows(client)
    served = [r for r in rows if r["status"] == "success"]
    absorbed = [r for r in rows if r["status"] == "absorbed"]
    assert len(served) == 1 and len(absorbed) == 1
    assert served[0]["policy_name"] == "fast"
    assert served[0]["attempt_position"] == 2
    assert served[0]["request_group_id"] is not None
    assert served[0]["request_group_id"] == absorbed[0]["request_group_id"]


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
def test_messages_endpoint_fails_over(client: TestClient, stream: bool) -> None:
    _create_user(client)
    calls: list[str] = []

    async def flaky(**kwargs: Any) -> Any:
        calls.append(kwargs["model"])
        if kwargs["model"] == "openai:gpt-5-mini":
            raise _http_error(503)
        if not stream:
            return _message_response()

        async def chunks() -> Any:
            from any_llm.types.completion import ChatCompletionChunk, ChoiceDelta, ChunkChoice

            yield ChatCompletionChunk(
                id="c1",
                choices=[ChunkChoice(delta=ChoiceDelta(content="hi"), index=0, finish_reason=None)],
                created=0,
                model="claude-haiku-4-5",
                object="chat.completion.chunk",
                usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        return chunks()

    with patch("gateway.api.routes.messages.amessages", new=flaky):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "fast",
                "max_tokens": 16,
                "stream": stream,
                "messages": [{"role": "user", "content": "hi"}],
                # /v1/messages takes the billed user in metadata, not a top-level field.
                "metadata": {"user_id": "test-user"},
            },
            headers=HEADERS,
        )

    assert resp.status_code == 200, resp.text
    assert calls == ["openai:gpt-5-mini", "anthropic:claude-haiku-4-5"]


def test_responses_endpoint_fails_over(client: TestClient) -> None:
    _create_user(client)
    calls: list[str] = []

    async def flaky(**kwargs: Any) -> Any:
        calls.append(f"{kwargs['provider'].value}:{kwargs['model']}")
        if kwargs["model"] == "gpt-5-mini":
            raise _http_error(503)
        from any_llm.types.responses import Response as ProviderResponse

        return ProviderResponse.model_construct(
            id="resp-1", object="response", created_at=0, model="claude-haiku-4-5", output=[], usage=None
        )

    with patch("gateway.api.routes.responses.aresponses", new=flaky):
        resp = client.post(
            "/v1/responses",
            json={"model": "fast", "input": "hi", "user": "test-user"},
            headers=HEADERS,
        )

    assert resp.status_code == 200, resp.text
    assert calls == ["openai:gpt-5-mini", "anthropic:claude-haiku-4-5"]
