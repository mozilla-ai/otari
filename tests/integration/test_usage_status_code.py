"""End-to-end tests for ``usage_logs.status_code`` (the error taxonomy).

Regression for #433: a usage log recorded ``status`` (success/error) and free-text
``error_message``, so failures could be counted but not classified. Breaking them
down meant substring-matching provider-specific error prose, which differs per
provider and changes without notice. These tests pin the column that makes the
breakdown a GROUP BY: what each failure path records, that it survives the
gateway's deliberate coarsening of the client-facing status, and that it is
filterable and groupable over the API.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from fastapi.testclient import TestClient

_MESSAGES = [{"role": "user", "content": "Hi"}]
_MODEL = "openai:gpt-4o"


class _StatusError(Exception):
    """Upstream failure carrying an HTTP status, as the provider SDKs raise."""

    def __init__(self, status_code: int) -> None:
        super().__init__("raw upstream message SECRET-9f3a")
        self.status_code = status_code


@contextmanager
def _upstream_fails(exc: BaseException) -> Iterator[None]:
    """Fail the provider call, for both streaming and non-streaming requests
    (the chat adapter opens streams through the same ``acompletion``)."""
    with patch("gateway.api.routes.chat.acompletion", new_callable=AsyncMock, side_effect=exc):
        yield


@contextmanager
def _upstream_succeeds() -> Iterator[None]:
    response = ChatCompletion(
        id="chatcmpl-status-code",
        object="chat.completion",
        created=0,
        model=_MODEL,
        choices=[
            Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    with patch("gateway.api.routes.chat.acompletion", new_callable=AsyncMock, return_value=response):
        yield


def _chat(client: TestClient, headers: dict[str, str], *, stream: bool = False) -> int:
    body: dict[str, Any] = {"model": _MODEL, "messages": _MESSAGES}
    if stream:
        body["stream"] = True
    return int(client.post("/v1/chat/completions", json=body, headers=headers).status_code)


def _error_rows(client: TestClient, master_key_header: dict[str, str], **params: Any) -> list[dict[str, Any]]:
    query: dict[str, Any] = {"status": "error", **params}
    response = client.get("/v1/usage", params=query, headers=master_key_header)
    assert response.status_code == 200
    rows: list[dict[str, Any]] = response.json()
    return rows


@pytest.mark.parametrize("upstream", [400, 404, 429, 500])
def test_upstream_status_is_recorded_on_the_error_row(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
    upstream: int,
) -> None:
    """The provider's own status lands on the row, per code, not a boolean."""
    with _upstream_fails(_StatusError(upstream)):
        _chat(client, api_key_header)

    rows = _error_rows(client, master_key_header)
    assert len(rows) == 1
    assert rows[0]["status_code"] == upstream


def test_upstream_credential_fault_is_classifiable_despite_the_generic_502(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """An upstream 401 is recorded as 401 even though the caller sees 502.

    This is the case that makes the column load-bearing rather than cosmetic. A
    provider rejecting the gateway's credentials is a gateway-config fault, so
    the response deliberately says only "502 / provider error" and the raw
    provider message never reaches the caller. That leaves the operator with no
    way to tell "my key is wrong" from "the provider is down": the client-facing
    status is 502 for both, ``status`` is "error" for both, and matching the
    error prose is the brittle heuristic this issue exists to remove.
    """
    with _upstream_fails(_StatusError(401)):
        response = client.post(
            "/v1/chat/completions", json={"model": _MODEL, "messages": _MESSAGES}, headers=api_key_header
        )
    # What the caller is told: a generic 502, with no trace of the upstream 401.
    assert response.status_code == 502
    assert "401" not in response.text
    assert "SECRET" not in response.text

    # What the operator can now see, on the master-key-only usage surface.
    rows = _error_rows(client, master_key_header)
    assert len(rows) == 1
    assert rows[0]["status_code"] == 401


def test_timeout_records_the_gateways_own_classification(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """A provider that never answered carries no status, so the row records the
    gateway's classification (504) rather than staying unclassifiable."""
    with _upstream_fails(TimeoutError("no response")):
        assert _chat(client, api_key_header) == 504

    rows = _error_rows(client, master_key_header)
    assert len(rows) == 1
    assert rows[0]["status_code"] == 504


def test_streaming_failure_records_the_upstream_status(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """A stream that fails before its first chunk is classified too.

    Streaming settles through its own callbacks, so without this the whole
    streaming half of the traffic would log NULL and the taxonomy would silently
    under-report whichever failures happen to arrive on streaming requests.
    """
    with _upstream_fails(_StatusError(429)):
        _chat(client, api_key_header, stream=True)

    rows = _error_rows(client, master_key_header)
    assert len(rows) == 1
    assert rows[0]["status_code"] == 429


def test_successful_request_records_no_status_code(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """A success has no failure to classify, so the column stays NULL rather than
    a constant 200 that would dilute every GROUP BY over it."""
    with _upstream_succeeds():
        assert _chat(client, api_key_header) == 200

    rows = client.get("/v1/usage", params={"status": "success"}, headers=master_key_header).json()
    assert len(rows) == 1
    assert rows[0]["status_code"] is None


def test_status_code_filters_the_list_and_the_count(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """``status_code`` narrows both /v1/usage and /v1/usage/count, so a paginated
    "show me the 429s" view agrees with its own total."""
    for upstream in (429, 429, 500):
        with _upstream_fails(_StatusError(upstream)):
            _chat(client, api_key_header)

    rows = _error_rows(client, master_key_header, status_code=429)
    assert len(rows) == 2
    assert {row["status_code"] for row in rows} == {429}

    count = client.get(
        "/v1/usage/count", params={"status_code": 429}, headers=master_key_header
    ).json()
    assert count["total"] == 2


def test_summary_groups_failures_by_status_code(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """The summary answers "which failures, and how many of each" as an aggregate.

    Two 429s and one 401 must come back as two rows carrying their coarse display
    class, and the counts must reconcile with ``totals.error_count`` so the
    taxonomy cannot disagree with the tile above it.
    """
    for upstream in (429, 429, 401):
        with _upstream_fails(_StatusError(upstream)):
            _chat(client, api_key_header)

    summary = client.get("/v1/usage/summary", headers=master_key_header).json()
    taxonomy = summary["errors_by_status_code"]
    assert taxonomy == [
        {"status_code": 429, "error_class": "rate_limit", "requests": 2},
        {"status_code": 401, "error_class": "auth", "requests": 1},
    ]
    assert sum(row["requests"] for row in taxonomy) == summary["totals"]["error_count"]


def test_summary_taxonomy_excludes_successful_requests(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """Only failures are grouped, so a mostly-healthy window does not bury the
    error taxonomy under one giant NULL bucket of successes."""
    with _upstream_succeeds():
        assert _chat(client, api_key_header) == 200
    with _upstream_fails(_StatusError(429)):
        _chat(client, api_key_header)

    summary = client.get("/v1/usage/summary", headers=master_key_header).json()
    assert summary["totals"]["request_count"] == 2
    assert summary["errors_by_status_code"] == [
        {"status_code": 429, "error_class": "rate_limit", "requests": 1}
    ]
