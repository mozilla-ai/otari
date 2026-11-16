"""A gateway that hands recall's ranking to a memory service: ``memory_service_url`` and ``otari memory-service``."""

from collections.abc import Callable, Generator
from typing import cast

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.adapters.memory_ranking_adapter import RANK_PATH, HttpMemoryRanking
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.ports.memory_ranking_port import MemoryRankingPort
from gateway.services.memory import MemoryModels
from gateway.services.secret_box import generate_secret_key

from .conftest import app_for, build_test_client
from .test_memory_mcp import DEPLOY, _embed, _key
from .test_memory_recall import ASK, _held, _last_user_text, _remembered, _workspace_id
from .test_memory_tool_loop import _chat, _completion, _provider, _switches

TOKEN = "memory-service-token-for-the-tests-0123456789"


def _engine_config(config: GatewayConfig) -> GatewayConfig:
    return config.model_copy(update={"memory_engine": True, "memory_service_url": None, "memory_service_token": TOKEN})


@pytest.fixture
def engine_app(test_config: GatewayConfig, clean_database: None, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """The memory service's app, with no lifespan of its own: the gateway's set up the database it shares."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    monkeypatch.setattr(MemoryModels, "embed", _embed)
    app = app_for(
        _engine_config(
            test_config.model_copy(update={"memory_enabled": True, "memory_embedding_model": "openai:embed"})
        )
    )
    app.state.memory_recall_vectors = None
    app.state.memory_keys = None
    return app


@pytest.fixture
def split_client(test_config: GatewayConfig, engine_app: FastAPI) -> Generator[TestClient]:
    """A gateway that ranks on ``engine_app``, reached in process."""
    config = test_config.model_copy(
        update={
            "memory_enabled": True,
            "memory_embedding_model": "openai:embed",
            "memory_service_url": "http://memory-service",
            "memory_service_token": TOKEN,
        }
    )
    for client in build_test_client(config):
        _rank_on(client, HttpMemoryRanking("http://memory-service", TOKEN, transport=httpx.ASGITransport(engine_app)))
        yield client


def _rank_on(client: TestClient, ranking: MemoryRankingPort) -> None:
    cast(FastAPI, client.app).state.container.bind(MemoryRankingPort, lambda session: ranking)


def _engine_held(app: FastAPI) -> int:
    held = getattr(app.state, "memory_recall_vectors", None)
    return len(held) if held is not None else 0


def test_the_memory_service_answers_only_its_gateways(
    split_client: TestClient, engine_app: FastAPI, master_key_header: dict[str, str]
) -> None:
    del split_client  # Booted for the database the two share.
    body = {"workspace_id": "6f1c1c1e-0000-4000-8000-000000000000", "candidates": [], "queries": [[1.0]]}
    body |= {"order": "similarity", "keep_top": 3}
    engine = TestClient(engine_app)

    assert engine.post(RANK_PATH, json=body).status_code == 401
    assert engine.post(RANK_PATH, json=body, headers=master_key_header).status_code == 401, "not the master key"
    assert engine.post(RANK_PATH, json=body, headers={"Authorization": "Bearer nope"}).status_code == 401
    answer = engine.post(RANK_PATH, json=body, headers={"Authorization": f"Bearer {TOKEN}"})
    assert answer.status_code == 200, answer.text
    assert answer.json() == {"ranked": []}


def test_a_gateway_mounts_no_memory_service_routes(split_client: TestClient, master_key_header: dict[str, str]) -> None:
    response = split_client.post(RANK_PATH, json={}, headers={"Authorization": f"Bearer {TOKEN}"})

    assert response.status_code == 404


def test_recall_through_the_memory_service_hands_over_what_it_would_in_process(
    split_client: TestClient,
    engine_app: FastAPI,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    headers = _key(split_client, master_key_header, db_session_factory)
    _remembered(split_client, headers)
    _switches(split_client, db_session_factory, master_key_header, auto_recall=True)

    with _provider(_completion(None)) as sent:
        response = _chat(split_client, headers, messages=[{"role": "user", "content": ASK}])

    assert response.status_code == 200, response.text
    assert response.headers["Otari-Memory-Recalled"] == "1"
    assert DEPLOY in _last_user_text(sent[0])
    assert _held(split_client) == 0, "the gateway holds no opened embedding"
    assert _engine_held(engine_app) == 1, "the memory service does"


def test_search_ranks_on_the_memory_service(
    split_client: TestClient,
    engine_app: FastAPI,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    from .test_memory_mcp import _call

    headers = _key(split_client, master_key_header, db_session_factory)
    _remembered(split_client, headers)

    found = _call(split_client, headers, "memory_search", query=ASK)

    assert found["isError"] is False
    assert DEPLOY in str(found["structuredContent"])
    assert _engine_held(engine_app) == 1


def test_recall_goes_on_without_memory_when_the_service_is_down(
    split_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    headers = _key(split_client, master_key_header, db_session_factory)
    _remembered(split_client, headers)
    _switches(split_client, db_session_factory, master_key_header, auto_recall=True)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    _rank_on(split_client, HttpMemoryRanking("http://memory-service", TOKEN, transport=httpx.MockTransport(refuse)))
    with _provider(_completion(None)) as sent:
        response = _chat(split_client, headers, messages=[{"role": "user", "content": ASK}])

    assert response.status_code == 200, response.text
    assert response.headers["Otari-Memory-Recalled"] == "0"
    assert DEPLOY not in _last_user_text(sent[0])


def test_erasing_a_workspace_drops_what_the_memory_service_holds_of_it(
    split_client: TestClient,
    engine_app: FastAPI,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    headers = _key(split_client, master_key_header, db_session_factory)
    _remembered(split_client, headers)
    _switches(split_client, db_session_factory, master_key_header, auto_recall=True)
    with _provider(_completion(None)):
        _chat(split_client, headers, messages=[{"role": "user", "content": ASK}])
    assert _engine_held(engine_app) == 1

    erased = split_client.delete(
        f"{API_ROOT}/workspaces/{_workspace_id(db_session_factory)}/memory", headers=master_key_header
    )

    assert erased.status_code == 200, erased.text
    assert _engine_held(engine_app) == 0
