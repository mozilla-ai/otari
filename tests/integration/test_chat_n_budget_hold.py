"""A chat request for ``n`` choices holds the budget for ``n`` outputs (mozilla-ai/otari#3239).

The provider bills every choice, so the hold the request is admitted on must
cover them all; settlement already does. What the route hands the admission
step as its output estimate is the figure the dollar and token holds are priced
from.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT

from .conftest import MODEL_NAME

_DEFAULT_OUTPUT_TOKENS = 1024


def _estimated_output_tokens(client: TestClient, api_key_header: dict[str, str], body: dict[str, Any]) -> int | None:
    captured: dict[str, Any] = {}

    async def fake_resolve_request_context(**kwargs: Any) -> Any:
        captured.update(kwargs)
        raise HTTPException(status_code=418, detail="stop after admission inputs")

    with patch("gateway.api.routes.chat.resolve_request_context", new=fake_resolve_request_context):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}], **body},
            headers=api_key_header,
        )

    assert response.status_code == 418
    return captured["estimate_max_output_tokens"]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"max_tokens": 300}, 300),
        ({"max_tokens": 300, "n": 1}, 300),
        ({"max_tokens": 300, "n": 8}, 2400),
        ({"max_completion_tokens": 300, "n": 8}, 2400),
        ({"max_tokens": 300, "n": 0}, 300),
        ({"max_tokens": 0, "n": 8}, 0),
        ({"n": 8}, 8 * _DEFAULT_OUTPUT_TOKENS),
    ],
)
def test_output_estimate_covers_every_choice(
    client: TestClient, api_key_header: dict[str, str], body: dict[str, Any], expected: int
) -> None:
    assert _estimated_output_tokens(client, api_key_header, body) == expected


def test_unbounded_single_choice_keeps_the_deployment_default(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    """No cap and no ``n`` stays unset, so the estimate falls back to the deployment default as before."""
    assert _estimated_output_tokens(client, api_key_header, {}) is None
