"""Tests for the POST /api/v1/embeddings endpoint."""

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest
from any_llm.types.completion import CreateEmbeddingResponse, Embedding, Usage
from any_llm.utils.exception_handler import convert_exception
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.api.routes._pipeline import PROVIDER_TIMEOUT_DETAIL
from gateway.core.config import API_ROOT
from gateway.core.error_codes import ERROR_CODE_HEADER, UPSTREAM_RATE_LIMITED
from gateway.exceptions import TenancyValidationError
from gateway.metrics import REGISTRY
from gateway.models.usage import UsageLog


def _mock_embedding_response(prompt_tokens: int = 10) -> CreateEmbeddingResponse:
    """Build a real CreateEmbeddingResponse for testing."""
    return CreateEmbeddingResponse(
        data=[Embedding(embedding=[0.1, 0.2, 0.3], index=0, object="embedding")],
        model="text-embedding-3-small",
        object="list",
        usage=Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
    )


def _attached_settlements() -> float:
    """How many inline costs this process has attached to a response."""
    value = REGISTRY.get_sample_value("gateway_inline_cost_settlements_total", {"outcome": "attached"})
    return value or 0.0


def test_embeddings_requires_auth(client: TestClient) -> None:
    """POST /api/v1/embeddings requires authentication."""
    resp = client.post(f"{API_ROOT}/embeddings", json={"model": "openai:text-embedding-3-small", "input": "hello"})
    assert resp.status_code == 401


def test_embeddings_with_api_key(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /api/v1/embeddings works with API key authentication."""
    mock_resp = _mock_embedding_response()
    with patch("gateway.api.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1


def test_embeddings_master_key_requires_user(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """POST /api/v1/embeddings with master key requires 'user' field."""
    mock_resp = _mock_embedding_response()
    with patch("gateway.api.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hello"},
            headers=master_key_header,
        )
    assert resp.status_code == 400
    assert "user" in resp.json()["detail"].lower()


def test_embeddings_master_key_with_user(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """POST /api/v1/embeddings with master key + user field succeeds."""
    mock_resp = _mock_embedding_response()
    with patch("gateway.api.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={
                "model": "openai:text-embedding-3-small",
                "input": "hello",
                "user": test_user["user_id"],
            },
            headers=master_key_header,
        )
    assert resp.status_code == 200


def test_embeddings_list_input(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /api/v1/embeddings accepts a list of strings as input."""
    mock_resp = _mock_embedding_response()
    with patch("gateway.api.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": ["hello", "world"]},
            headers=api_key_header,
        )
    assert resp.status_code == 200


def test_embeddings_provider_error(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /api/v1/embeddings returns 502 when the provider fails."""
    with patch(
        "gateway.api.routes.embeddings.aembedding",
        new_callable=AsyncMock,
        side_effect=RuntimeError("provider down"),
    ):
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 502
    assert "provider" in resp.json()["detail"].lower()
    assert "x-should-retry" not in resp.headers


def test_embeddings_tells_a_client_not_to_retry_a_local_adapter_fault(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """A provider adapter that fails before sending anything fails the same way on every resend."""
    with patch("gateway.api.routes.embeddings.aembedding", new_callable=AsyncMock, side_effect=KeyError("data")):
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 502
    assert resp.headers["x-should-retry"] == "false"


def test_embeddings_logs_usage(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /api/v1/embeddings creates a usage log entry."""
    mock_resp = _mock_embedding_response()
    user_id = api_key_obj["user_id"]

    with patch("gateway.api.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 200

    usage_resp = client.get(f"{API_ROOT}/users/{user_id}/usage", headers=master_key_header)
    assert usage_resp.status_code == 200
    logs = usage_resp.json()
    embedding_logs = [log for log in logs if log["endpoint"] == "/v1/embeddings"]
    assert len(embedding_logs) >= 1
    assert embedding_logs[0]["status"] == "success"
    assert embedding_logs[0]["prompt_tokens"] == 10
    assert embedding_logs[0]["completion_tokens"] == 0


def test_embeddings_logs_error_on_failure(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /api/v1/embeddings logs an error entry when the provider fails."""
    user_id = api_key_obj["user_id"]

    with patch(
        "gateway.api.routes.embeddings.aembedding",
        new_callable=AsyncMock,
        side_effect=RuntimeError("provider down"),
    ):
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 502

    usage_resp = client.get(f"{API_ROOT}/users/{user_id}/usage", headers=master_key_header)
    assert usage_resp.status_code == 200
    logs = usage_resp.json()
    error_logs = [log for log in logs if log["endpoint"] == "/v1/embeddings" and log["status"] == "error"]
    assert len(error_logs) >= 1
    assert "provider down" in error_logs[0]["error_message"]


@pytest.mark.parametrize("extra_field", ["encoding_format", "dimensions"])
def test_embeddings_optional_fields(
    client: TestClient,
    api_key_header: dict[str, str],
    extra_field: str,
) -> None:
    """POST /api/v1/embeddings forwards optional OpenAI fields."""
    mock_resp = _mock_embedding_response()
    values = {"encoding_format": "float", "dimensions": 256}
    with patch("gateway.api.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=mock_resp) as mock:
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={
                "model": "openai:text-embedding-3-small",
                "input": "hello",
                extra_field: values[extra_field],
            },
            headers=api_key_header,
        )
    assert resp.status_code == 200
    call_kwargs = mock.call_args.kwargs
    assert extra_field in call_kwargs
    assert call_kwargs[extra_field] == values[extra_field]


@pytest.mark.parametrize(
    ("model", "input_type"),
    [("cohere:embed-english-v3.0", "search_query"), ("voyage:voyage-3", "query")],
)
def test_embeddings_forwards_input_type_to_providers_that_take_it(
    client: TestClient,
    api_key_header: dict[str, str],
    model: str,
    input_type: str,
) -> None:
    """``input_type`` reaches any-llm for a provider whose embedding depends on it."""
    mock_resp = _mock_embedding_response()
    with patch("gateway.api.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=mock_resp) as mock:
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": model, "input": ["hi"], "input_type": input_type},
            headers=api_key_header,
        )
    assert resp.status_code == 200
    assert mock.call_args.kwargs["input_type"] == input_type


def test_embeddings_omits_input_type_when_not_sent(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """Without the field, any-llm's own provider default applies."""
    mock_resp = _mock_embedding_response()
    with patch("gateway.api.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=mock_resp) as mock:
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "cohere:embed-english-v3.0", "input": ["hi"]},
            headers=api_key_header,
        )
    assert resp.status_code == 200
    assert "input_type" not in mock.call_args.kwargs


def test_embeddings_rejects_input_type_for_a_provider_without_it(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """A provider that does not take ``input_type`` is a 400 before any call, not a 502 or a silent drop."""
    with patch("gateway.api.routes.embeddings.aembedding", new_callable=AsyncMock) as mock:
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hi", "input_type": "search_query"},
            headers=api_key_header,
        )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "'input_type' is not supported for model 'openai:text-embedding-3-small'"
    mock.assert_not_called()

    # Nothing was dispatched, so the reservation was refunded rather than left held.
    user = client.get(f"{API_ROOT}/users/{api_key_obj['user_id']}", headers=master_key_header).json()
    assert user["reserved"] == pytest.approx(0.0)
    assert user["spend"] == pytest.approx(0.0)


def test_a_domain_error_from_the_provider_call_is_refunded_and_rendered_by_its_status(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """A domain error raised in the call keeps its own status and detail, and the hold is refunded."""
    with patch(
        "gateway.api.routes.embeddings.aembedding",
        new_callable=AsyncMock,
        side_effect=TenancyValidationError("the domain refused this"),
    ):
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hi"},
            headers=api_key_header,
        )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "the domain refused this"

    user = client.get(f"{API_ROOT}/users/{api_key_obj['user_id']}", headers=master_key_header).json()
    assert user["reserved"] == pytest.approx(0.0)
    assert user["spend"] == pytest.approx(0.0)


def test_embeddings_rejects_an_oversized_input_type(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """The field is a short enum-like value, so a long string is refused at validation."""
    with patch("gateway.api.routes.embeddings.aembedding", new_callable=AsyncMock) as mock:
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "cohere:embed-english-v3.0", "input": "hi", "input_type": "x" * 65},
            headers=api_key_header,
        )
    assert resp.status_code == 422
    mock.assert_not_called()


def test_embeddings_cost_tracked_with_pricing(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /api/v1/embeddings calculates cost when model pricing exists.

    A realistic prompt, because settlement is to the micro-dollar: ten tokens at
    $0.02 per million is two ten-millionths of a dollar, which the cost column
    cannot hold and rounds to zero.
    """
    client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": "openai:text-embedding-3-small",
            "input_price_per_million": 0.02,
            "output_price_per_million": 0.0,
        },
        headers=master_key_header,
    )

    mock_resp = _mock_embedding_response(prompt_tokens=1_000)
    user_id = api_key_obj["user_id"]

    with patch("gateway.api.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 200

    usage_resp = client.get(f"{API_ROOT}/users/{user_id}/usage", headers=master_key_header)
    logs = usage_resp.json()
    embedding_logs = [log for log in logs if log["endpoint"] == "/v1/embeddings"]
    assert len(embedding_logs) >= 1
    assert embedding_logs[0]["cost"] is not None
    assert embedding_logs[0]["cost"] > 0


def test_embeddings_charge_below_half_a_micro_dollar_settles_at_zero(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
) -> None:
    """The floor of the settlement scale, pinned rather than avoided.

    Ten tokens at $0.02 per million is two ten-millionths of a dollar. The cost
    column holds six decimals, so the row settles at zero; the charge line keeps
    the unrounded amount, so the work is still visible in the audit trail even
    though it was billed at nothing.
    """
    client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": "openai:text-embedding-3-small",
            "input_price_per_million": 0.02,
            "output_price_per_million": 0.0,
        },
        headers=master_key_header,
    )

    mock_resp = _mock_embedding_response(prompt_tokens=10)
    with patch("gateway.api.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 200

    logs = client.get(f"{API_ROOT}/usage", params={"endpoint": "/v1/embeddings"}, headers=master_key_header).json()
    assert logs[0]["cost"] == 0.0
    assert logs[0]["pricing_breakdown"] == [
        {"meter": "input", "units": 10, "rate_per_million": 0.02, "cost": pytest.approx(2e-7)}
    ]


def test_embeddings_billing_meters_tracked_with_pricing(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
) -> None:
    """POST /api/v1/embeddings records auditable charge lines alongside cost."""
    client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": "openai:text-embedding-3-small",
            "input_price_per_million": 0.02,
            "output_price_per_million": 0.0,
        },
        headers=master_key_header,
    )

    mock_resp = _mock_embedding_response()

    with patch("gateway.api.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 200

    usage_resp = client.get(f"{API_ROOT}/usage", params={"endpoint": "/v1/embeddings"}, headers=master_key_header)
    logs = usage_resp.json()
    assert len(logs) >= 1
    assert logs[0]["billing_meters"] == {
        "total_input_tokens": 10,
        "fresh_input_tokens": 10,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cache_write_1h_tokens": 0,
        "completion_tokens": 0,
    }
    breakdown = logs[0]["pricing_breakdown"]
    assert breakdown == [{"meter": "input", "units": 10, "rate_per_million": 0.02, "cost": pytest.approx(2e-7)}]


def test_embeddings_response_carries_inline_cost_when_priced(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """A priced embedding reports its cost and rate source on ``usage``, as chat does."""
    client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": "openai:text-embedding-3-small",
            "input_price_per_million": 0.02,
            "output_price_per_million": 0.0,
        },
        headers=master_key_header,
    )

    mock_resp = _mock_embedding_response(prompt_tokens=1_000)
    attached_before = _attached_settlements()
    with patch("gateway.api.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 200
    assert _attached_settlements() == attached_before + 1
    usage = resp.json()["usage"]
    assert usage == {
        "prompt_tokens": 1_000,
        "total_tokens": 1_000,
        "cost_usd": "0.000020",
        "pricing_source": "deployment",
    }

    logs = client.get(f"{API_ROOT}/usage", params={"endpoint": "/v1/embeddings"}, headers=master_key_header).json()
    assert logs[0]["cost"] == pytest.approx(float(usage["cost_usd"]))

    # The row keeps the same provenance the response reports. The usage API does
    # not expose the column, so it is read directly.
    with db_session_factory() as db:
        row = db.execute(select(UsageLog).where(UsageLog.id == logs[0]["id"])).scalar_one()
        assert row.pricing_source == "deployment"


def test_embeddings_response_reports_a_settled_zero_with_its_source(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
) -> None:
    """A charge below half a micro-dollar settles at zero, and the response says so rather than omitting it."""
    client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": "openai:text-embedding-3-small",
            "input_price_per_million": 0.02,
            "output_price_per_million": 0.0,
        },
        headers=master_key_header,
    )

    mock_resp = _mock_embedding_response(prompt_tokens=2)
    with patch("gateway.api.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hi"},
            headers=api_key_header,
        )
    assert resp.status_code == 200
    usage = resp.json()["usage"]
    assert usage["cost_usd"] == "0.000000"
    assert usage["pricing_source"] == "deployment"


def test_embeddings_response_carries_no_inline_cost_when_unpriced(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """An unpriced model's response carries neither field, matching chat's null semantics."""
    mock_resp = _mock_embedding_response()
    with patch("gateway.api.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:otari-unlisted-embedder", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 200
    usage = resp.json()["usage"]
    assert "cost_usd" not in usage
    assert "pricing_source" not in usage


def _openai_bad_request(message: str) -> Exception:
    """An OpenAI 400 as any-llm hands it back."""
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    upstream = openai.BadRequestError(
        f"Error code: 400 - {{'error': {{'message': '{message}'}}}}",
        response=httpx.Response(400, request=request),
        body={"message": message, "type": "invalid_request_error", "param": None, "code": None},
    )
    return convert_exception(upstream, "openai")


def _error_rows(client: TestClient, master_key_header: dict[str, str], user_id: str) -> list[dict[str, Any]]:
    usage_resp = client.get(
        f"{API_ROOT}/usage", params={"user_id": user_id, "endpoint": "/v1/embeddings"}, headers=master_key_header
    )
    assert usage_resp.status_code == 200, usage_resp.text
    return [log for log in usage_resp.json() if log["status"] == "error"]


def test_embeddings_upstream_bad_request_is_a_400(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """A provider's 400 is the caller's to fix, so it is not answered as a retryable 502."""
    with patch(
        "gateway.api.routes.embeddings.aembedding",
        new_callable=AsyncMock,
        side_effect=_openai_bad_request("input cannot be an empty string"),
    ):
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": [""]},
            headers=api_key_header,
        )
    assert resp.status_code == 400
    assert resp.json() == {"detail": "input cannot be an empty string"}

    rows = _error_rows(client, master_key_header, api_key_obj["user_id"])
    assert [row["status_code"] for row in rows] == [400]


def test_embeddings_param_the_provider_sdk_refuses_is_a_400(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """A param the caller sent that the provider's SDK has no keyword for is named in a 400."""
    with patch(
        "gateway.api.routes.embeddings.aembedding",
        new_callable=AsyncMock,
        side_effect=TypeError("AsyncV2Client.embed() got an unexpected keyword argument 'encoding_format'"),
    ):
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hello", "encoding_format": "float"},
            headers=api_key_header,
        )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "The provider serving this model does not accept the 'encoding_format' parameter"

    rows = _error_rows(client, master_key_header, api_key_obj["user_id"])
    assert [row["status_code"] for row in rows] == [400]


def test_embeddings_keyword_the_caller_did_not_send_stays_a_502(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """An unexpected keyword the caller never sent is the gateway's mismatch, not the caller's."""
    with patch(
        "gateway.api.routes.embeddings.aembedding",
        new_callable=AsyncMock,
        side_effect=TypeError("embed() got an unexpected keyword argument 'encoding_format'"),
    ):
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 502
    assert "encoding_format" not in resp.json()["detail"]


def test_embeddings_upstream_rate_limit_is_a_429_with_its_retry_after(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """The provider's rate limit keeps its status and wait, and names itself as the provider's."""
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    upstream = openai.RateLimitError(
        "Error code: 429 - Rate limit reached for requests",
        response=httpx.Response(429, request=request, headers={"Retry-After": "7"}),
        body={"message": "Rate limit reached for requests"},
    )
    with patch(
        "gateway.api.routes.embeddings.aembedding",
        new_callable=AsyncMock,
        side_effect=convert_exception(upstream, "openai"),
    ):
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 429
    assert resp.headers[ERROR_CODE_HEADER] == UPSTREAM_RATE_LIMITED
    assert resp.headers["Retry-After"] == "7"

    rows = _error_rows(client, master_key_header, api_key_obj["user_id"])
    assert [row["status_code"] for row in rows] == [429]


def test_embeddings_provider_timeout_is_a_504(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    timeout = openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"))
    with patch(
        "gateway.api.routes.embeddings.aembedding",
        new_callable=AsyncMock,
        side_effect=convert_exception(timeout, "openai"),
    ):
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 504
    assert resp.json()["detail"] == PROVIDER_TIMEOUT_DETAIL
