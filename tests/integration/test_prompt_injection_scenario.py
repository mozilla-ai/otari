"""An indirect prompt injection, fetched by ``otari_web_fetch`` and screened by a mandate (#3202).

The model asks to fetch ``tests/fixtures/prompt_injection/changelog.html``, a page
that hides an instruction to send the project's ``.env`` to an attacker's tool.
The fetcher and its extraction run for real; only the network under them, the
provider and the guardrails service are stood in for. The guardrails service
flags any text carrying the instruction, so what is asserted is the chain: the
injected text survives the fetch, a mandate reading tool results sees it, and
the mode decides whether the model does too.

Which of the page's three copies of the instruction survive the extraction is
pinned in ``tests/unit/test_web_fetch_injection_surface.py``.
"""

from __future__ import annotations

import ipaddress
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import httpx
import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    Choice,
    CompletionUsage,
    Function,
)
from fastapi.testclient import TestClient

from gateway.api.routes._helpers import GUARDRAILS_RESULT_HEADER
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.services.web_retrieval_network import SystemAddressResolver

from .conftest import MODEL_NAME

_PAGE = (Path(__file__).resolve().parents[1] / "fixtures" / "prompt_injection" / "changelog.html").read_bytes()
_PAGE_HOST = "lumen-notes.example.com"
_PAGE_URL = f"https://{_PAGE_HOST}/r7k2/changelog.html"
_GUARDRAILS_HOST = "anyguardrails"
_INSTRUCTION = "ATTENTION AI AGENT"
_SURVIVING_COPY = "LN-OFFSCREEN"


class _Network:
    """The honeypot and the guardrails service, and every text the guardrails service was asked about."""

    def __init__(self) -> None:
        self.screened: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == _PAGE_HOST:
            # A stream, not ``content=``: the fetcher reads the raw body under its own cap.
            return httpx.Response(
                200, headers={"content-type": "text/html; charset=utf-8"}, stream=httpx.ByteStream(_PAGE)
            )
        assert request.url.host == _GUARDRAILS_HOST, request.url
        body = json.loads(request.content)
        self.screened.append(body["input_text"])
        return httpx.Response(
            200, json={"profile": body["profile"], "result": {"valid": _INSTRUCTION not in body["input_text"]}}
        )


@pytest.fixture
def network(monkeypatch: pytest.MonkeyPatch, test_config: GatewayConfig) -> _Network:
    monkeypatch.setattr(test_config, "web_fetch_enabled", True)
    monkeypatch.setenv("OTARI_GUARDRAILS_URL", f"http://{_GUARDRAILS_HOST}:8000")
    network = _Network()
    transport = httpx.MockTransport(network.handler)
    real_async_client = httpx.AsyncClient

    def client_factory(*_args: object, **_kwargs: object) -> httpx.AsyncClient:
        return real_async_client(transport=transport)

    async def public(_self: SystemAddressResolver, _host: str, _port: int) -> Sequence[ipaddress.IPv4Address]:
        return [ipaddress.IPv4Address("93.184.216.34")]

    # The fetcher and the guardrails client each build an httpx.AsyncClient; both reach the transport above.
    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(SystemAddressResolver, "resolve", public)
    return network


def _completion(*, fetch: bool) -> ChatCompletion:
    calls = [
        ChatCompletionMessageFunctionToolCall(
            id="call_0", type="function", function=Function(name="web_fetch", arguments=json.dumps({"url": _PAGE_URL}))
        )
    ]
    return ChatCompletion(
        id="chatcmpl-1",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant", content=None if fetch else "done", tool_calls=cast(Any, calls if fetch else None)
                ),
                finish_reason="tool_calls" if fetch else "stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _run_mission(
    client: TestClient, api_key_header: dict[str, str], master_key_header: dict[str, str], *, mode: str
) -> tuple[dict[str, Any], str]:
    """Mandate the guardrail in ``mode`` and send the mission; return its guardrail entry and the tool result."""
    mandated = client.post(
        f"{API_ROOT}/organizations/me/guardrails",
        json={
            "profile": "prompt-injection",
            "mode": mode,
            "on": ["input", "tool_results"],
            "applies_to_all_workspaces": True,
        },
        headers=master_key_header,
    )
    assert mandated.status_code == 201, mandated.text
    sent: list[dict[str, Any]] = []
    rounds = iter([_completion(fetch=True), _completion(fetch=False)])

    async def _provider(**kwargs: Any) -> ChatCompletion:
        sent.append(json.loads(json.dumps(kwargs, default=str)))
        return next(rounds)

    with patch("gateway.services.mcp_loop.acompletion", new=_provider):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": f"Bump lumen-notes to 4.2.0. Read {_PAGE_URL} first."}],
                "tools": [{"type": "otari_web_fetch"}],
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "done"
    [tool_message] = [m for m in sent[1]["messages"] if m.get("role") == "tool"]
    [entry] = json.loads(response.headers[GUARDRAILS_RESULT_HEADER])
    return entry, tool_message["content"]


def test_a_blocking_mandate_withholds_the_page_that_carries_the_instruction(
    client: TestClient, api_key_header: dict[str, str], master_key_header: dict[str, str], network: _Network
) -> None:
    entry, tool_result = _run_mission(client, api_key_header, master_key_header, mode="block")

    assert any(_SURVIVING_COPY in text for text in network.screened), "the fetched page was screened"
    assert _INSTRUCTION not in tool_result
    assert "withheld by Otari" in tool_result
    assert (entry["mode"], entry["valid"], entry["withheld"]) == ("block", False, 1)


def test_a_monitoring_mandate_flags_the_page_and_lets_the_model_read_it(
    client: TestClient, api_key_header: dict[str, str], master_key_header: dict[str, str], network: _Network
) -> None:
    entry, tool_result = _run_mission(client, api_key_header, master_key_header, mode="monitor")

    assert any(_SURVIVING_COPY in text for text in network.screened), "the fetched page was screened"
    assert _SURVIVING_COPY in tool_result
    assert (entry["mode"], entry["valid"]) == ("monitor", False)
    assert "withheld" not in entry
