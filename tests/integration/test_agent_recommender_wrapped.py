"""A bootstrap that wraps the core recommender changes the label and keeps the price.

The hosted edition records every recommendation under a label of its own in
place of the core label ``otari:subagent-router``, and does so by wrapping the
core adapter rather than replacing it: it takes the core factory the
container bound, builds the core adapter per request, and rewrites the quote
it answers with. That is the seam this test pins: the usage row carries the
wrapper's label and the stored router's price, and the router the wrapper
sees through ``quoted`` is the one that was asked.
"""

import itertools
import sys
from collections.abc import Generator, Iterator
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.services.routing.recommend import RECOMMENDATION_QUESTION
from gateway.services.secret_box import generate_secret_key

from .conftest import build_test_client

MODULE = "probe_wrapped_recommender"
WRAPPER_BOOTSTRAP = """
from dataclasses import replace

from gateway.container import Container
from gateway.ports.agent_model_recommender_port import AgentModelRecommenderPort, Quote

QUOTED = []


class Relabel:
    def __init__(self, inner):
        self.inner = inner

    def quote(self):
        quote = self.inner.quote()
        QUOTED.append((self.inner.quoted.provider, self.inner.quoted.model, quote.charge))
        return Quote(provider="hosted", model="subagent-router", charge=quote.charge)

    async def recommend(self, spawn, *, candidates):
        answer = await self.inner.recommend(spawn, candidates=candidates)
        return replace(answer, reason=f"Otari chose {answer.model}")


def register(container: Container) -> None:
    core = dict(container.bindings())[AgentModelRecommenderPort]
    container.bind(AgentModelRecommenderPort, lambda session: Relabel(core(session)))
"""
MASTER = {API_KEY_HEADER: "Bearer test-master-key"}
RECOMMEND = f"{API_ROOT}/routing/recommend"
ROUTERS = f"{API_ROOT}/routing/recommenders"
STORED_KEY = "sk-or-stored-router-1234"
ROUTER = {"provider": "openrouter", "model": "stored-decider", "api_key": STORED_KEY, "input_price_per_million": 20}
SPAWN: dict[str, Any] = {
    "harness": "claude-code",
    "session_id": "bf05abe2-5ff2-4eb0-8459-ab578d5c9468",
    "tool_use_id": "toolu_01",
    "agent_type": "Explore",
    "description": "List files",
    "prompt": "List the files under src/ and report what each module does.",
    "parent_model": "claude-opus-5",
    "requested_model": None,
}
ANSWER: dict[str, Any] = {
    "model": "decider-1.0",
    "answers": {RECOMMENDATION_QUESTION: {"type": "choice", "choice": "haiku"}},
    "usage": {"input_tokens": 800, "output_tokens": 0},
}

_NAMES = (f"{MODULE}_{n}" for n in itertools.count())


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


@pytest.fixture
def wrapped(
    postgres_url: str, clean_database: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[tuple[TestClient, ModuleType]]:
    name = next(_NAMES)
    (tmp_path / f"{name}.py").write_text(WRAPPER_BOOTSTRAP)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(name, None)
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        # The stored router's quote satisfies this; the relabelled model has no row of its own.
        require_pricing=True,
        bootstrap=f"{name}:register",
    )
    try:
        for client in build_test_client(config):
            yield client, sys.modules[name]
    finally:
        sys.modules.pop(name, None)


def test_the_wrapper_relabels_the_stored_routers_quote_and_keeps_its_price(
    wrapped: tuple[TestClient, ModuleType],
) -> None:
    client, probe = wrapped
    stored = client.post(
        ROUTERS,
        json={
            "provider": "openrouter",
            "model": "stored-decider",
            "api_key": STORED_KEY,
            "input_price_per_million": 20,
        },
        headers=MASTER,
    )
    assert stored.status_code == 201, stored.text
    key = client.post(f"{API_ROOT}/keys", json={"key_name": "probe"}, headers=MASTER).json()

    with patch("gateway.adapters.agent_model_recommender_adapter.request_decision", AsyncMock(return_value=ANSWER)):
        response = client.post(RECOMMEND, json=SPAWN, headers={API_KEY_HEADER: f"Bearer {key['key']}"})

    assert response.status_code == 200, response.text
    assert response.json() == {"model": "haiku", "reason": "Otari chose haiku", "probabilities": None}
    # The stored router's hold: 65,536 input tokens at $20 per million.
    assert probe.QUOTED == [("openrouter", "stored-decider", Decimal("1.310720"))]

    rows = client.get(
        f"{API_ROOT}/usage", params={"user_id": key["user_id"], "endpoint": "/v1/decisions"}, headers=MASTER
    )
    (row,) = rows.json()
    assert (row["status"], row["provider"], row["model"]) == ("success", "hosted", "subagent-router")
    assert float(row["cost"]) == pytest.approx(0.016)


def test_the_wrapper_sees_the_core_refusal_with_every_router_off(wrapped: tuple[TestClient, ModuleType]) -> None:
    client, probe = wrapped
    stored = client.post(
        ROUTERS,
        json={
            "provider": "openrouter",
            "model": "stored-decider",
            "api_key": STORED_KEY,
            "input_price_per_million": 20,
        },
        headers=MASTER,
    ).json()
    assert client.patch(f"{ROUTERS}/{stored['id']}", json={"enabled": False}, headers=MASTER).status_code == 200
    key = client.post(f"{API_ROOT}/keys", json={"key_name": "probe"}, headers=MASTER).json()

    response = client.post(RECOMMEND, json=SPAWN, headers={API_KEY_HEADER: f"Bearer {key['key']}"})

    assert response.status_code == 400, response.text
    assert probe.QUOTED == []
