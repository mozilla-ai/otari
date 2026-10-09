"""A request that fails schema validation answers with the stable ``invalid_request`` code.

The 422 keeps FastAPI's ``detail`` list, so a client reading it is unaffected,
and gains the ``code`` field and ``Otari-Error-Code`` header every other coded
refusal carries.
"""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.error_codes import ERROR_CODE_HEADER, INVALID_REQUEST
from gateway.main import create_app

MASTER_KEY = "sk-test-master"


def _client(tmp_path: Path) -> TestClient:
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'validation-test.db'}",
        master_key=MASTER_KEY,
        require_pricing=False,
    )
    return TestClient(create_app(config))


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/chat/completions", {"model": "openai:gpt-4o-mini"}),
        ("/messages", {"model": "anthropic:claude-sonnet-4-5", "max_tokens": "many", "messages": []}),
        ("/responses", {"input": "hi"}),
        ("/auth/session", {"email": "operator@example.com"}),
    ],
    ids=["chat-completions", "messages", "responses", "management"],
)
def test_a_body_that_fails_validation_carries_the_invalid_request_code(
    tmp_path: Path, path: str, body: dict[str, Any]
) -> None:
    with _client(tmp_path) as client:
        response = client.post(f"{API_ROOT}{path}", json=body, headers={"Otari-Key": MASTER_KEY})

    assert response.status_code == 422, response.text
    assert response.headers[ERROR_CODE_HEADER] == INVALID_REQUEST
    payload = response.json()
    assert payload["code"] == INVALID_REQUEST
    # The existing contract is untouched: a list of field errors, values not echoed.
    assert payload["detail"]
    assert all(set(error) == {"type", "loc", "msg"} for error in payload["detail"])


def test_a_malformed_json_body_is_also_an_invalid_request(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post(
            f"{API_ROOT}/chat/completions",
            content=b"{not json",
            headers={"Otari-Key": MASTER_KEY, "Content-Type": "application/json"},
        )

    assert response.status_code == 422, response.text
    assert response.json()["code"] == INVALID_REQUEST
    assert response.headers[ERROR_CODE_HEADER] == INVALID_REQUEST


def test_a_valid_request_carries_no_error_code(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.get(f"{API_ROOT}/health")

    assert response.status_code == 200, response.text
    assert ERROR_CODE_HEADER not in response.headers


def test_the_published_validation_error_schema_names_the_code(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        schema = client.app.openapi()  # type: ignore[attr-defined]

    validation_error = schema["components"]["schemas"]["HTTPValidationError"]
    assert validation_error["properties"]["code"]["const"] == INVALID_REQUEST
    # Optional in the schema, so a generated client stays lenient with an older gateway.
    assert "code" not in validation_error.get("required", [])
