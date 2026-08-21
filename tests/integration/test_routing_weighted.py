"""End-to-end behavior for the weighted routing policy: load balancing traffic
across providers.

What these defend, each being a claim the feature makes on the wire rather than in
a helper:

* Traffic really lands on more than one provider, and the caller cannot tell: the
  response says the policy name whichever provider served.
* Every attempt is attributable after the fact. The usage row names the model that
  served and carries ``selection_reason: router:weighted``, which is the only
  record of the split, because a weighted decision is deliberately not logged per
  request.
* A failure stays inside the balanced pool before reaching ``on_failure``, so a
  provider having a bad minute sheds its share.
* A drained (zero-weight) candidate takes no traffic and still backs a failure.
* The split is bounded by the caller's allow-list, like every other policy.
* No pricing is required, unlike the learned router: weight is operator-declared
  capacity, not a number derived from cost.

The RNG is seeded through the backend the registry hands out, so these assert an
exact split rather than a distribution: a statistical assertion here would be a
flaky test that also tells you less.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
)
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, GatewayConfig
from gateway.models.routing import RoutingConfig
from gateway.services.routing.weighted import WeightedRouterBackend

from .conftest import build_test_client

HEADERS = {API_KEY_HEADER: "Bearer test-master-key"}
USER = "test-user"
HEAVY = "openai:gpt-5"
LIGHT = "anthropic:claude-sonnet-4-5"
BACKUP = "openai:gpt-5-mini"


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
        usage=CompletionUsage(completion_tokens=2, prompt_tokens=3, total_tokens=5),
    )


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    return httpx.HTTPStatusError(str(status), request=request, response=httpx.Response(status, request=request))


@pytest.fixture
def weighted_config(postgres_url: str) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        # No `pricing` block at all: a weighted policy must work without one.
        providers={"openai": {"api_key": "sk-openai"}, "anthropic": {"api_key": "sk-ant"}},
        routing=RoutingConfig.model_validate(
            {
                "policies": {
                    # The shape the docs recommend: two providers sharing the load,
                    # with a third naming the failure chain behind them.
                    "balanced": {
                        "select": [
                            {"router": "weighted", "candidates": [HEAVY, LIGHT], "weights": {HEAVY: 80, LIGHT: 20}},
                            {"default": HEAVY},
                        ],
                        "on_failure": [BACKUP],
                    },
                    # A provider drained to zero: still in the plan, taking no traffic.
                    "drained": {
                        "select": [
                            {"router": "weighted", "candidates": [HEAVY, LIGHT], "weights": {HEAVY: 1, LIGHT: 0}},
                            {"default": HEAVY},
                        ]
                    },
                }
            }
        ),
    )


@pytest.fixture
def client(weighted_config: GatewayConfig) -> Generator[TestClient]:
    # The registry builds a fresh backend per request (it is stateless), so the
    # seeded stream has to be shared across those instances: handing each one its
    # own `Random(11)` would replay the same first draw on every request and every
    # request would land on the same provider.
    stream = random.Random(11)
    with patch(
        "gateway.services.routing.weighted.WeightedRouterBackend",
        new=lambda: WeightedRouterBackend(stream),
    ):
        client_gen = build_test_client(weighted_config)
        test_client = next(client_gen)
        try:
            resp = test_client.post("/v1/users", json={"user_id": USER}, headers=HEADERS)
            assert resp.status_code == 200, resp.text
            yield test_client
        finally:
            client_gen.close()


def _chat(
    client: TestClient,
    model: str,
    *,
    headers: dict[str, str] | None = None,
    fail: set[str] | None = None,
    **extra: Any,
) -> tuple[Any, list[str]]:
    """POST a chat request, returning the response and the models the provider saw.

    ``fail`` names models that raise an upstream error, which is how a failover
    inside the balanced pool is exercised.
    """
    calls: list[str] = []

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        model_name = kwargs["model"]
        calls.append(model_name)
        if fail and model_name in fail:
            raise _http_error(503)
        return _completion(model_name)

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}], "user": USER, **extra},
            headers={**HEADERS, **(headers or {})},
        )
    return resp, calls


def _usage_rows(client: TestClient) -> list[dict[str, Any]]:
    resp = client.get("/v1/usage", params={"limit": 200}, headers=HEADERS)
    assert resp.status_code == 200, resp.text
    payload: Any = resp.json()
    rows: list[dict[str, Any]] = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    return rows


# -- the split itself -------------------------------------------------------


def test_traffic_is_split_across_providers(client: TestClient) -> None:
    served: Counter[str] = Counter()
    for _ in range(40):
        resp, calls = _chat(client, "balanced")
        assert resp.status_code == 200, resp.text
        # The caller never learns which provider served: that is the point of a
        # policy, and it is what makes rebalancing a config change rather than a
        # client change.
        assert resp.json()["model"] == "balanced"
        served[calls[0]] += 1

    # Both providers carried traffic, and the heavier weight carried more of it.
    assert set(served) == {HEAVY, LIGHT}
    assert served[HEAVY] > served[LIGHT]


def test_every_served_request_is_attributable_to_the_router(client: TestClient) -> None:
    # A weighted decision is deliberately not logged per request, so the usage row
    # is the record of the split. If this stops saying `router:weighted`, an
    # operator has no way to audit where the money went.
    for _ in range(6):
        assert _chat(client, "balanced")[0].status_code == 200
    rows = [row for row in _usage_rows(client) if row["model"] in {"gpt-5", "claude-sonnet-4-5"}]
    assert len(rows) == 6
    assert {row["selection_reason"] for row in rows} == {"router:weighted"}


def test_a_provider_failure_stays_inside_the_balanced_pool(client: TestClient) -> None:
    # The draw continues without replacement, so the second attempt is the next
    # weighted candidate rather than the operator's on_failure entry. A provider
    # having a bad minute therefore sheds its share to the others for free.
    resp, calls = _chat(client, "balanced", fail={HEAVY})
    assert resp.status_code == 200, resp.text
    assert calls[0] == HEAVY
    assert calls[1] == LIGHT
    assert BACKUP not in calls

    absorbed = [row for row in _usage_rows(client) if row["status"] == "absorbed"]
    assert [row["model"] for row in absorbed] == ["gpt-5"]


def test_on_failure_still_backs_the_whole_pool(client: TestClient) -> None:
    resp, calls = _chat(client, "balanced", fail={HEAVY, LIGHT})
    assert resp.status_code == 200, resp.text
    assert calls[-1] == BACKUP


def test_a_drained_candidate_takes_no_traffic_but_still_backs_a_failure(client: TestClient) -> None:
    # `{heavy: 1, light: 0}` is how a provider is drained without being deleted.
    for _ in range(10):
        resp, calls = _chat(client, "drained")
        assert resp.status_code == 200, resp.text
        assert calls[0] == HEAVY

    resp, calls = _chat(client, "drained", fail={HEAVY})
    assert resp.status_code == 200, resp.text
    assert calls[1] == LIGHT


def test_otari_router_off_serves_the_default_target(client: TestClient) -> None:
    # One contract for every backend: the caller opts out of the per-request choice
    # and gets the policy's default, which on a weighted policy pins them to one
    # provider. Useful during an incident, and it must not silently do nothing.
    for _ in range(8):
        resp, calls = _chat(client, "balanced", headers={"Otari-Router": "off"})
        assert resp.status_code == 200, resp.text
        assert calls[0] == HEAVY
    rows = [row for row in _usage_rows(client) if row["status"] != "absorbed"]
    assert {row["selection_reason"] for row in rows} == {"default"}


def test_the_split_never_dispatches_a_candidate_the_key_forbids(client: TestClient) -> None:
    resp = client.post(
        "/v1/keys",
        json={"user_id": USER, "name": "anthropic-only", "allowed_models": ["anthropic:*"]},
        headers=HEADERS,
    )
    assert resp.status_code == 200, resp.text
    key = resp.json()["key"]

    for _ in range(8):
        served, calls = _chat(
            client, "balanced", headers={API_KEY_HEADER: f"Bearer {key}"}
        )
        assert served.status_code == 200, served.text
        assert calls[0] == LIGHT


# -- the management surfaces ------------------------------------------------


def test_explain_reports_the_split_rather_than_the_decline_path(client: TestClient) -> None:
    resp = client.post("/v1/routing/policies/explain", json={"name": "balanced"}, headers=HEADERS)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["router_backend"] == "weighted"
    assert body["router_weights"] == {HEAVY: 80.0, LIGHT: 20.0}
    assert body["selection_reason"] == "router:weighted"
    # Heaviest share first, then the failure chain.
    assert [candidate["model"] for candidate in body["candidates"]] == [
        "gpt-5",
        "claude-sonnet-4-5",
        "gpt-5-mini",
    ]


def test_explain_renormalizes_the_split_over_an_allow_list(client: TestClient) -> None:
    # A "balanced" policy that compiles to one provider for a given key is the
    # failure this surface exists to catch, so the share has to reflect filtering.
    resp = client.post(
        "/v1/routing/policies/explain",
        json={"name": "balanced", "allowed_models": ["anthropic:*"]},
        headers=HEADERS,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["router_weights"] == {LIGHT: 100.0}
    # The on_failure entry is filtered out for this caller too, and saying so is
    # the point: this policy is one provider deep for them, with no chain behind it.
    assert [item["selector"] for item in body["dropped"]] == [HEAVY, BACKUP]


def test_a_stored_weighted_policy_needs_no_pricing(client: TestClient) -> None:
    # The learned router is refused here without pricing for every candidate,
    # because it scores by cost. The weighted router reads no prices, so the same
    # gate must not apply: nothing in this gateway has a pricing row.
    resp = client.post(
        "/v1/routing/policies",
        json={
            "name": "stored-balanced",
            "spec": {
                "select": [
                    {"router": "weighted", "candidates": [HEAVY, LIGHT], "weights": {HEAVY: 1, LIGHT: 1}},
                    {"default": HEAVY},
                ]
            },
        },
        headers=HEADERS,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["is_dynamic"] is True

    served, calls = _chat(client, "stored-balanced")
    assert served.status_code == 200, served.text
    assert calls[0] in {HEAVY, LIGHT}


def test_a_malformed_split_is_refused_with_field_level_errors(client: TestClient) -> None:
    resp = client.post(
        "/v1/routing/policies",
        json={
            "name": "bad-split",
            "spec": {
                "select": [
                    {"router": "weighted", "candidates": [HEAVY, LIGHT], "weights": {"openai:nope": 1}},
                    {"default": HEAVY},
                ]
            },
        },
        headers=HEADERS,
    )
    assert resp.status_code == 400, resp.text
    assert "do not name a candidate" in json.dumps(resp.json())


def test_a_weighted_policy_has_no_single_price_in_the_catalog(client: TestClient) -> None:
    resp = client.get("/v1/models", headers=HEADERS)
    assert resp.status_code == 200, resp.text
    entry = next(model for model in resp.json()["data"] if model["id"] == "balanced")
    # Quoting one candidate's rate would be wrong whenever the policy does its job.
    assert entry["pricing_source"] == "dynamic"
