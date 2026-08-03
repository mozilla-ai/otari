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

# The gateway-rejection gates are set up exactly as the tests that own them do,
# so a gate whose fixture shape changes cannot drift between the two files.
from .test_gateway_rejection_logging import _make_key, _make_user, _zero_budget

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


def _chat(client: TestClient, headers: dict[str, str], *, stream: bool = False, **overrides: Any) -> int:
    body: dict[str, Any] = {"model": _MODEL, "messages": _MESSAGES, **overrides}
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


@pytest.mark.parametrize(
    ("params", "expected_rows"),
    [
        ({}, 1),
        ({"dimensions": "status_code"}, 1),
        ({"dimensions": "none"}, 0),
        ({"dimensions": "model"}, 0),
    ],
)
def test_summary_taxonomy_answers_to_the_dimension_selector(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
    params: dict[str, str],
    expected_rows: int,
) -> None:
    """The taxonomy is one more GROUP BY pass, so it obeys ``dimensions`` like the
    spend breakdowns do (#469): present by default and when asked for by name,
    skipped for a caller that only wants totals and the series, which is what the
    dashboard's tiles and timelines request.
    """
    with _upstream_fails(_StatusError(429)):
        _chat(client, api_key_header)

    summary = client.get("/v1/usage/summary", params=params, headers=master_key_header).json()
    assert len(summary["errors_by_status_code"]) == expected_rows
    # Either way the totals still count the failure: only the extra pass is skipped.
    assert summary["totals"]["error_count"] == 1


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


def test_bare_status_code_filter_returns_only_failures(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """``status_code`` alone is an error-scoped filter, so a code can never pull a
    success row in: the column belongs to failures, and the query says so instead
    of trusting every caller to add ``status=error`` (and every future write path
    to leave the column NULL on success)."""
    with _upstream_succeeds():
        assert _chat(client, api_key_header) == 200
    with _upstream_fails(_StatusError(429)):
        _chat(client, api_key_header)

    rows = client.get("/v1/usage", params={"status_code": 429}, headers=master_key_header).json()
    assert [(row["status"], row["status_code"]) for row in rows] == [("error", 429)]

    count = client.get("/v1/usage/count", params={"status_code": 429}, headers=master_key_header).json()
    assert count["total"] == 1

    # An explicit status still wins, so the filter stays literal rather than
    # quietly overriding what the caller asked for.
    contradictory = client.get(
        "/v1/usage", params={"status": "success", "status_code": 429}, headers=master_key_header
    ).json()
    assert contradictory == []


def test_passthrough_provider_failure_records_the_upstream_status(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """The pass-through scaffold (embeddings, images, rerank, audio) classifies its
    provider failures too.

    These routes settle through their own error row in ``_passthrough.py`` rather
    than through the chat pipeline, so without this the whole non-chat half of
    billable traffic could stop recording a code and every other test would still
    pass.
    """
    with patch(
        "gateway.api.routes.embeddings.aembedding",
        new_callable=AsyncMock,
        side_effect=_StatusError(429),
    ):
        response = client.post(
            "/v1/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hi"},
            headers=api_key_header,
        )
    # The caller sees the deliberately generic provider error, not the upstream 429.
    assert response.status_code == 502
    assert "SECRET" not in response.text

    rows = _error_rows(client, master_key_header, endpoint="/v1/embeddings")
    assert len(rows) == 1
    assert rows[0]["status_code"] == 429


def test_batch_create_failure_records_the_upstream_status(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """``POST /v1/batches`` classifies its provider failure on its own error row.

    Batches write usage through ``log_batch_usage`` instead of the chat pipeline's
    writer, so this is the second independent stamp that a refactor could drop
    silently.
    """

    class _SupportsBatch:
        SUPPORTS_BATCH = True

    body = {
        "model": "openai:gpt-4o-mini",
        "requests": [{"custom_id": "req-1", "body": {"messages": _MESSAGES, "max_tokens": 16}}],
    }
    with (
        patch(
            "gateway.api.routes.batches.acreate_batch",
            new_callable=AsyncMock,
            side_effect=_StatusError(503),
        ),
        patch("gateway.api.routes.batches.AnyLLM.get_provider_class", return_value=_SupportsBatch),
    ):
        response = client.post("/v1/batches", json=body, headers=api_key_header)
    assert response.status_code == 502
    assert "SECRET" not in response.text

    rows = _error_rows(client, master_key_header, endpoint="/v1/batches")
    assert len(rows) == 1
    assert rows[0]["status_code"] == 503
    # And it reaches the taxonomy as a provider fault rather than as "unknown".
    summary = client.get("/v1/usage/summary", headers=master_key_header).json()
    assert summary["errors_by_status_code"] == [
        {"status_code": 503, "error_class": "provider_error", "requests": 1}
    ]


def _one_error_row(client: TestClient, master_key_header: dict[str, str]) -> dict[str, Any]:
    rows = _error_rows(client, master_key_header)
    assert len(rows) == 1, rows
    return rows[0]


def _embeddings(client: TestClient, headers: dict[str, str], model: str, **body: Any) -> int:
    response = client.post("/v1/embeddings", json={"model": model, "input": "hi", **body}, headers=headers)
    return int(response.status_code)


# ---------------------------------------------------------------------------
# Gateway-side rejections (#465): the gateway refused the request itself, so the
# row carries the status it returned rather than an upstream one. Without a code
# these rows classify as "unknown", which is indistinguishable from a row written
# before the column existed, so every gate that writes one is pinned here.
# ---------------------------------------------------------------------------


def test_over_budget_rejection_records_its_403(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The refusal raised inside ``reserve_budget`` records the 403 it returns.

    A budget denial reads as ``auth`` in the taxonomy, which is deliberate for now
    and documented on ``error_class_for``: the code alone cannot separate the
    gateway refusing the caller from a provider refusing the gateway.
    """
    _make_user(client, master_key_header, "broke-user", budget_id=_zero_budget(client, master_key_header))

    assert _chat(client, master_key_header, user="broke-user") == 403

    assert _one_error_row(client, master_key_header)["status_code"] == 403
    summary = client.get("/v1/usage/summary", headers=master_key_header).json()
    assert summary["errors_by_status_code"] == [{"status_code": 403, "error_class": "auth", "requests": 1}]


def test_user_key_mismatch_rejection_records_its_403(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The mismatch gate fires before anything is resolved and still records 403."""
    _make_user(client, master_key_header, "owner")
    _make_user(client, master_key_header, "someone-else")
    key = _make_key(client, master_key_header, "owned", user_id="owner")

    assert _chat(client, key, user="someone-else") == 403

    row = _one_error_row(client, master_key_header)
    assert row["user_id"] == "owner"
    assert row["status_code"] == 403


def test_allow_list_rejection_records_its_403(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A key's allow-list refusal records 403, so a mis-scoped key is not filed as
    a provider outage."""
    _make_user(client, master_key_header, "scoped-user")
    key = _make_key(client, master_key_header, "scoped", user_id="scoped-user", allowed_models=["anthropic:*"])

    assert _chat(client, key) == 403

    assert _one_error_row(client, master_key_header)["status_code"] == 403


def test_unresolvable_selector_rejection_records_its_400(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A selector that no longer resolves records the 400 the caller sees, which
    reads as ``client_error`` rather than as a provider fault."""
    _make_user(client, master_key_header, "curious-user")

    assert _chat(client, master_key_header, model="nosuchprovider:some-model", user="curious-user") == 400

    assert _one_error_row(client, master_key_header)["status_code"] == 400
    summary = client.get("/v1/usage/summary", headers=master_key_header).json()
    assert summary["errors_by_status_code"] == [{"status_code": 400, "error_class": "client_error", "requests": 1}]


@pytest.mark.parametrize(
    ("model", "expected"),
    [("openai:text-embedding-3-small", 403), ("nosuchprovider:embed", 400)],
)
def test_passthrough_gateway_rejections_record_the_status_they_return(
    client: TestClient,
    master_key_header: dict[str, str],
    model: str,
    expected: int,
) -> None:
    """The pass-through scaffold stamps its own rejections too: a blocked user
    refused after the selector resolved (403) and a selector that never resolved
    (400). These settle through the scaffold's own helper, so the chat coverage
    above says nothing about them."""
    _make_user(client, master_key_header, "blocked-embedder", blocked=True)

    assert _embeddings(client, master_key_header, model, user="blocked-embedder") == expected

    row = _one_error_row(client, master_key_header)
    assert row["endpoint"] == "/v1/embeddings"
    assert row["status_code"] == expected


def test_passthrough_allow_list_rejection_records_its_403(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The pass-through allow-list gate refunds and then records its 403."""
    _make_user(client, master_key_header, "scoped-embedder")
    key = _make_key(
        client, master_key_header, "scoped-embed", user_id="scoped-embedder", allowed_models=["cohere:*"]
    )

    assert _embeddings(client, key, "openai:text-embedding-3-small") == 403

    assert _one_error_row(client, master_key_header)["status_code"] == 403


def test_passthrough_user_key_mismatch_records_its_403(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The pass-through mismatch gate records 403, like its pipeline counterpart."""
    _make_user(client, master_key_header, "embed-owner")
    _make_user(client, master_key_header, "embed-stranger")
    key = _make_key(client, master_key_header, "embed-owned", user_id="embed-owner")

    assert _embeddings(client, key, "openai:text-embedding-3-small", user="embed-stranger") == 403

    row = _one_error_row(client, master_key_header)
    assert row["user_id"] == "embed-owner"
    assert row["status_code"] == 403
