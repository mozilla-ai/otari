"""End to end through a ``router: smart_router`` policy, against a stub smart-router service.

The stub is a real HTTP server on a loopback port, so the backend's own client,
timeouts and JSON are exercised. It answers ``/v1/route`` with whatever the test
scripts and records every request, which is how these tests see what the
gateway told the service: the messages and candidates it routed on, the
completion it reported once the usage row was written, and the rating a caller
sent through ``POST /api/v1/routing/feedback``.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT, REQUEST_ID_HEADER, GatewayConfig
from gateway.core.settings.pricing import PricingConfig
from gateway.models.routing import RoutingConfig
from gateway.models.usage import UsageLog

from .conftest import build_test_client

MASTER = {API_KEY_HEADER: "Bearer test-master-key"}
CHEAP = "openai:gpt-5-mini"
STRONG = "openai:gpt-5"


class StubSmartRouter:
    """A smart-router service that answers ``/v1/route`` as scripted and records everything."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        # One sample per route call, as the real service mints them.
        self.samples: list[str] = []
        self.route_model: str | None = CHEAP
        self.route_status = 200
        self.feedback_status = 204
        self._lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"null")
                with stub._lock:
                    stub.requests.append((self.path, body))
                if self.path == "/v1/route":
                    sample_id = str(uuid.uuid4())
                    stub.samples.append(sample_id)
                    status = stub.route_status
                    payload: Any = {
                        "decision": {
                            "cluster_id": 1,
                            "model_id": stub.route_model,
                            "utility_score": 0.5,
                            "explored": False,
                        },
                        "sample_id": sample_id,
                    }
                elif self.path == "/v1/feedback":
                    status, payload = stub.feedback_status, None
                else:
                    status, payload = 204, None
                data = json.dumps(payload).encode() if payload is not None and status != 204 else b""
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def calls(self, path: str) -> list[dict[str, Any]]:
        with self._lock:
            return [body for seen, body in self.requests if seen == path]

    def wait_for(self, path: str, count: int = 1, timeout: float = 5.0) -> list[dict[str, Any]]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            found = self.calls(path)
            if len(found) >= count:
                return found
            time.sleep(0.02)
        return self.calls(path)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub() -> Iterator[StubSmartRouter]:
    service = StubSmartRouter()
    try:
        yield service
    finally:
        service.close()


def _config(postgres_url: str, smart_router_url: str | None) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        providers={"openai": {"api_key": "sk-openai"}},
        smart_router_url=smart_router_url,
        smart_router_timeout_seconds=2.0,
        pricing={
            CHEAP: PricingConfig(input_price_per_million=1.0, output_price_per_million=2.0),
            STRONG: PricingConfig(input_price_per_million=10.0, output_price_per_million=20.0),
        },
        routing=RoutingConfig.model_validate(
            {
                "policies": {
                    "smart": {
                        "select": [
                            {"router": "smart_router", "candidates": [CHEAP, STRONG], "cost_weight": 0.5},
                            {"default": STRONG},
                        ]
                    },
                    "plain": {"select": [{"default": STRONG}]},
                }
            }
        ),
    )


@contextmanager
def _client(config: GatewayConfig) -> Generator[TestClient]:
    client_gen = build_test_client(config)
    test_client = next(client_gen)
    try:
        yield test_client
    finally:
        client_gen.close()


@pytest.fixture
def client(postgres_url: str, stub: StubSmartRouter) -> Iterator[TestClient]:
    with _client(_config(postgres_url, stub.url)) as test_client:
        yield test_client


def _completion(model: str) -> ChatCompletion:
    return ChatCompletion(
        id="cmpl-1",
        choices=[Choice(finish_reason="stop", index=0, message=ChatCompletionMessage(role="assistant", content="ok"))],
        created=0,
        model=model,
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
    )


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://upstream")
    return httpx.HTTPStatusError(str(status), request=request, response=httpx.Response(status, request=request))


def _key(client: TestClient) -> str:
    response = client.post(f"{API_ROOT}/keys", json={"key_name": "caller"}, headers=MASTER)
    assert response.status_code == 200, response.text
    key: str = response.json()["key"]
    return key


def _chat(
    client: TestClient, key: str, model: str = "smart", *, failing: frozenset[str] = frozenset()
) -> tuple[Any, list[str]]:
    calls: list[str] = []

    async def acompletion(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs["model"])
        if kwargs["model"] in failing:
            raise _http_error(503)
        return _completion(kwargs["model"])

    with patch("gateway.api.routes.chat.acompletion", new=acompletion):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Answer briefly."},
                    {"role": "user", "content": "Reverse a string in Python."},
                ],
            },
            headers={API_KEY_HEADER: f"Bearer {key}"},
        )
    return response, calls


def _rows(test_db: Session, request_id: str) -> list[UsageLog]:
    test_db.expire_all()
    return list(test_db.scalars(select(UsageLog).where(UsageLog.request_id == request_id)))


# -- routing and the outcome -----------------------------------------------


def test_the_smart_router_picks_and_hears_what_it_cost(
    client: TestClient, stub: StubSmartRouter, test_db: Session
) -> None:
    response, calls = _chat(client, _key(client))

    assert response.status_code == 200, response.text
    # Routed on the real conversation, over the candidates as the policy wrote them.
    (route,) = stub.calls("/v1/route")
    assert route == {
        "application_id": "smart",
        "inputs": [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "Reverse a string in Python."},
        ],
        "lambda": 0.5,
        "allowed_model_ids": [CHEAP, STRONG],
    }
    # The pick served, and the caller still sees the policy name.
    assert calls == [CHEAP]
    assert response.json()["model"] == "smart"

    request_id = response.headers[REQUEST_ID_HEADER]
    (row,) = _rows(test_db, request_id)
    assert (row.status, row.model, row.selection_reason) == ("success", "gpt-5-mini", "router:smart_router")
    assert (row.routing_backend, row.routing_decision_id) == ("smart_router", stub.samples[0])

    (completion,) = stub.wait_for("/v1/completion")
    assert completion["sample_id"] == stub.samples[0]
    assert completion["model_id"] == CHEAP
    assert completion["success"] is True
    assert completion["error"] is None
    assert (completion["prompt_tokens"], completion["completion_tokens"], completion["total_tokens"]) == (
        1000,
        500,
        1500,
    )
    # 1000 prompt tokens at $1/M and 500 completion tokens at $2/M.
    assert completion["prompt_cost_usd"] == pytest.approx(0.001)
    assert completion["completion_cost_usd"] == pytest.approx(0.001)
    assert completion["total_cost_usd"] == pytest.approx(0.002)
    assert completion["request_started_at"] <= completion["request_completed_at"]


def test_a_failed_pick_reports_the_candidate_that_served(
    client: TestClient, stub: StubSmartRouter, test_db: Session
) -> None:
    response, calls = _chat(client, _key(client), failing=frozenset({CHEAP}))

    assert response.status_code == 200, response.text
    assert calls == [CHEAP, STRONG]
    statuses = sorted(row.status for row in _rows(test_db, response.headers[REQUEST_ID_HEADER]))
    assert statuses == ["absorbed", "success"]
    # One outcome for the request, naming what actually served; the absorbed attempt is not one.
    completions = stub.wait_for("/v1/completion")
    time.sleep(0.2)
    assert [body["model_id"] for body in stub.calls("/v1/completion")] == [STRONG]
    assert completions[0]["success"] is True


def test_an_exhausted_plan_reports_one_failure(client: TestClient, stub: StubSmartRouter) -> None:
    response, _ = _chat(client, _key(client), failing=frozenset({CHEAP, STRONG}))

    assert response.status_code >= 500
    (completion,) = stub.wait_for("/v1/completion")
    assert completion["success"] is False
    # The status the caller got for the exhausted plan, as the row classifies it.
    assert completion["error"]["status_code"] == response.status_code
    assert completion["total_tokens"] is None


@pytest.mark.parametrize("failure", ["no_opinion", "unknown_model", "error"])
def test_a_router_that_cannot_decide_serves_the_default(
    client: TestClient, stub: StubSmartRouter, test_db: Session, failure: str
) -> None:
    if failure == "no_opinion":
        stub.route_model = None
    elif failure == "unknown_model":
        stub.route_model = "openai:gpt-4o"
    else:
        stub.route_status = 500

    response, calls = _chat(client, _key(client))

    assert response.status_code == 200, response.text
    assert calls == [STRONG]
    (row,) = _rows(test_db, response.headers[REQUEST_ID_HEADER])
    assert row.selection_reason == "default"
    assert row.routing_decision_id is None
    time.sleep(0.2)
    assert stub.calls("/v1/completion") == []


def test_an_unset_smart_router_url_serves_the_default(postgres_url: str, stub: StubSmartRouter) -> None:
    with _client(_config(postgres_url, None)) as client:
        response, calls = _chat(client, _key(client))

    assert response.status_code == 200, response.text
    assert calls == [STRONG]
    assert stub.requests == []


def test_every_request_records_its_request_id(client: TestClient, test_db: Session) -> None:
    """Not only routed ones: the id a caller got back names the row whatever the model was."""
    response, _ = _chat(client, _key(client), model="plain")

    request_id = response.headers[REQUEST_ID_HEADER]
    uuid.UUID(request_id)
    (row,) = _rows(test_db, request_id)
    assert row.policy_name == "plain"
    assert row.routing_decision_id is None
