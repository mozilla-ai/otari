"""A stream the provider cuts after output reached the caller is billed on an estimate and stays an error."""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from any_llm.types.completion import ChatCompletionChunk
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT

from .conftest import MODEL_NAME


def _text_chunk(text: str) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-cut",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
    )


async def _provider_drops_the_connection() -> AsyncIterator[ChatCompletionChunk]:
    yield _text_chunk("abcd" * 25)
    raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")


def _usage_rows(client: TestClient, master_key_header: dict[str, str], **params: Any) -> list[dict[str, Any]]:
    response = client.get(f"{API_ROOT}/usage", params=params, headers=master_key_header)
    assert response.status_code == 200, response.text
    rows: list[dict[str, Any]] = response.json()
    return rows


def test_a_provider_cut_stream_is_an_error_row_that_names_the_provider_and_bills_its_estimate(
    client: TestClient, api_key_header: dict[str, str], master_key_header: dict[str, str]
) -> None:
    priced = client.post(
        f"{API_ROOT}/organizations/me/pricing",
        json={"model_key": MODEL_NAME, "input_price_per_million": 1000.0, "output_price_per_million": 1000.0},
        headers=master_key_header,
    )
    assert priced.status_code == 201, priced.text
    body = {"model": MODEL_NAME, "messages": [{"role": "user", "content": "hello there"}], "stream": True}
    with patch(
        "gateway.api.routes.chat.acompletion",
        new=AsyncMock(return_value=_provider_drops_the_connection()),
    ):
        response = client.post(f"{API_ROOT}/chat/completions", json=body, headers=api_key_header)
    assert response.status_code == 200, response.text

    (row,) = _usage_rows(client, master_key_header, status="error")
    assert row["stream_cut_by"] == "provider"
    assert row["completion_tokens"] == 25
    assert row["prompt_tokens"] and row["prompt_tokens"] > 0
    assert row["error_message"]
    assert row["cost"] and row["cost"] > 0
    assert _usage_rows(client, master_key_header, status="incomplete") == []

    totals = client.get(f"{API_ROOT}/usage/summary", headers=master_key_header).json()["totals"]
    assert totals["error_count"] == 1
    assert totals["cost"] == pytest.approx(row["cost"])
