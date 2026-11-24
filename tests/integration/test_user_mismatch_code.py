"""A key naming another user than its own is refused with ``user_mismatch``.

File uploads and direct searches are covered beside their other tests.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT
from gateway.core.error_codes import ERROR_CODE_HEADER, USER_MISMATCH

_MESSAGES = [{"role": "user", "content": "hi"}]

_JSON_ROUTES: list[tuple[str, dict[str, Any]]] = [
    ("chat/completions", {"model": "openai:gpt-4o", "messages": _MESSAGES, "user": "someone-else"}),
    ("responses", {"model": "openai:gpt-4o", "input": "hi", "user": "someone-else"}),
    (
        "messages",
        {"model": "openai:gpt-4o", "messages": _MESSAGES, "max_tokens": 16, "metadata": {"user_id": "someone-else"}},
    ),
    ("embeddings", {"model": "openai:text-embedding-3-small", "input": "hi", "user": "someone-else"}),
    (
        "batches",
        {
            "model": "openai:gpt-4o-mini",
            "requests": [{"custom_id": "req-1", "body": {"messages": _MESSAGES, "max_tokens": 16}}],
            "user": "someone-else",
        },
    ),
]


# The routes where a service key bills the end user it names, and so the detail suggests one.
_END_USER_ROUTES = {"chat/completions", "responses", "messages"}
# The routes whose body takes ``metadata``, and so the detail suggests it for a free-form ID.
_METADATA_ROUTES = {"chat/completions", "responses"}


@pytest.mark.parametrize(("route", "body"), _JSON_ROUTES, ids=[route for route, _ in _JSON_ROUTES])
def test_a_mismatched_user_is_refused_with_its_code(
    client: TestClient, api_key_header: dict[str, str], route: str, body: dict[str, Any]
) -> None:
    response = client.post(f"{API_ROOT}/{route}", json=body, headers=api_key_header)

    assert response.status_code == 403, response.text
    assert response.headers[ERROR_CODE_HEADER] == USER_MISMATCH
    assert response.json()["code"] == USER_MISMATCH
    # The detail says how to get past the refusal, and only the ways this route offers.
    assert "Always accept" in response.text
    assert ("service key" in response.text) is (route in _END_USER_ROUTES)
    assert ("metadata instead" in response.text) is (route in _METADATA_ROUTES)


def test_messages_names_the_field_it_read(client: TestClient, api_key_header: dict[str, str]) -> None:
    """Messages reads the user from ``metadata.user_id``, so its detail names that field."""
    route, body = _JSON_ROUTES[2]
    response = client.post(f"{API_ROOT}/{route}", json=body, headers=api_key_header)

    assert response.json()["error"]["message"].startswith("'metadata.user_id' does not match")
