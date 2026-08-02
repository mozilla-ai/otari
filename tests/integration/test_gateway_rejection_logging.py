"""End-to-end coverage for gateway-side rejections landing in the usage log.

#449 made the missing-pricing rejection visible (an error row the dashboard's
failure count and activity log read); every other gate still raised without
writing anything, so the count read 0 and the activity log showed nothing while
an operator's traffic was being dropped for a key allow-list, a blocked or
over-budget user, a user/key mismatch, or a selector that stopped resolving.

Each test here pins one gate: the request is refused, a single error row appears
with no cost, and the row is an enforced gateway row
(``counts_toward_budget=True``) rather than an imported-looking one. The
dashboard offers ``counts_toward_budget=False`` rows for bulk delete and
set-price, which must never reach a row the gateway wrote itself, so that
assertion is repeated per gate rather than factored out.

The two deliberate omissions (an unauthenticated 401, and an unknown user whose
row could not satisfy the ``usage_logs.user_id`` foreign key) are pinned too, so
they stay decisions rather than regressions.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from gateway.core.config import API_KEY_HEADER

_MESSAGES = [{"role": "user", "content": "hi"}]


def _errors(client: TestClient, headers: dict[str, str]) -> list[dict[str, Any]]:
    rows = client.get("/v1/usage", params={"status": "error"}, headers=headers).json()
    return [dict(r) for r in rows]


def _one_error(client: TestClient, headers: dict[str, str]) -> dict[str, Any]:
    """The single error row the request under test should have written."""
    rows = _errors(client, headers)
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["status"] == "error"
    # cost=null is what keeps these rows out of spend entirely.
    assert row["cost"] is None
    # Never an imported-looking row: see the module docstring.
    assert row["counts_toward_budget"] is True
    return row


def _make_user(client: TestClient, headers: dict[str, str], user_id: str, **fields: Any) -> None:
    resp = client.post("/v1/users", json={"user_id": user_id, **fields}, headers=headers)
    assert resp.status_code == 200, resp.text


def _make_key(client: TestClient, headers: dict[str, str], name: str, **fields: Any) -> dict[str, str]:
    resp = client.post("/v1/keys", json={"key_name": name, **fields}, headers=headers)
    assert resp.status_code == 200, resp.text
    return {API_KEY_HEADER: f"Bearer {resp.json()['key']}"}


def _zero_budget(client: TestClient, headers: dict[str, str]) -> str:
    resp = client.post("/v1/budgets", json={"max_budget": 0.0}, headers=headers)
    assert resp.status_code == 200, resp.text
    return str(resp.json()["budget_id"])


def _chat(client: TestClient, headers: dict[str, str], **body: Any) -> int:
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": _MESSAGES, **body},
        headers=headers,
    )
    return int(resp.status_code)


def test_model_not_allowed_for_key_is_recorded(client: TestClient, master_key_header: dict[str, str]) -> None:
    """A key's allow-list rejection (403) leaves a row, so a mis-scoped key is visible."""
    _make_user(client, master_key_header, "scoped-user")
    key = _make_key(client, master_key_header, "scoped", user_id="scoped-user", allowed_models=["anthropic:*"])

    assert _chat(client, key, model="openai:gpt-4o") == 403

    row = _one_error(client, master_key_header)
    assert row["user_id"] == "scoped-user"
    assert row["endpoint"] == "/v1/chat/completions"
    assert "not permitted" in row["error_message"]
    # The resolved target, matching the form every success row uses, so a model's
    # failures and successes group together in the activity log and its filter.
    assert row["model"] == "gpt-4o"
    assert row["provider"] == "openai"

    # The dashboard's "N failed in the last hour" signal reads the count scoped to
    # gateway traffic; a gate that logs must show up there or the alarm undercounts.
    scoped = client.get(
        "/v1/usage/count", params={"status": "error", "source": "gateway"}, headers=master_key_header
    ).json()
    assert scoped["total"] == 1


def test_blocked_user_rejection_is_recorded(client: TestClient, master_key_header: dict[str, str]) -> None:
    """A blocked user's 403 (raised inside reserve_budget) leaves a row."""
    _make_user(client, master_key_header, "blocked-user", blocked=True)

    assert _chat(client, master_key_header, model="openai:gpt-4o", user="blocked-user") == 403

    row = _one_error(client, master_key_header)
    assert row["user_id"] == "blocked-user"
    assert "blocked" in row["error_message"]
    assert row["model"] == "gpt-4o"
    assert row["provider"] == "openai"


def test_over_budget_rejection_is_recorded(client: TestClient, master_key_header: dict[str, str]) -> None:
    """An over-budget 403 leaves a row: the incident #317 was filed about."""
    budget_id = _zero_budget(client, master_key_header)
    _make_user(client, master_key_header, "broke-user", budget_id=budget_id)

    assert _chat(client, master_key_header, model="openai:gpt-4o", user="broke-user") == 403

    row = _one_error(client, master_key_header)
    assert row["user_id"] == "broke-user"
    assert "budget" in row["error_message"]


def test_user_key_mismatch_rejection_is_recorded_against_the_keys_user(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """A user/key mismatch (403) is attributed to the key's own user.

    Spend always binds to the key's user, so that is the only user the row can
    name. The selector is resolved later (aliases are user-scoped), so the row
    carries the requested selector with no provider.
    """
    _make_user(client, master_key_header, "owner")
    _make_user(client, master_key_header, "someone-else")
    key = _make_key(client, master_key_header, "owned", user_id="owner")

    assert _chat(client, key, model="openai:gpt-4o", user="someone-else") == 403

    row = _one_error(client, master_key_header)
    assert row["user_id"] == "owner"
    assert row["model"] == "openai:gpt-4o"
    assert row["provider"] is None


def test_unresolvable_selector_rejection_is_recorded(client: TestClient, master_key_header: dict[str, str]) -> None:
    """A selector that no longer resolves (400) leaves a row.

    This is what a provider removed from config looks like to its callers, so it
    should read as an incident in the activity log, not only in client errors.
    """
    _make_user(client, master_key_header, "curious-user")

    assert _chat(client, master_key_header, model="nosuchprovider:some-model", user="curious-user") == 400

    row = _one_error(client, master_key_header)
    assert row["user_id"] == "curious-user"
    assert row["model"] == "nosuchprovider:some-model"
    assert row["provider"] is None
    assert "Unknown or unsupported model" in row["error_message"]


def test_unresolvable_selector_releases_the_reservation(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Any
) -> None:
    """The 400 refunds its reservation, which it did not before #465.

    The preamble tolerates a selector it cannot resolve and carries it into the
    pricing lookup as the bare model with no provider. ``find_model_pricing``
    then builds its key as the model alone, and that raw selector is exactly the
    ``provider:model`` form stored pricing rows use, so an instance removed from
    config while its pricing row survives still prices, still reserves a nonzero
    estimate, and only fails later at dispatch. Before the refund, that hold
    stayed on ``users.reserved`` until the next budget reset (forever for a
    budget with no reset period).

    Structured as an A/B, because ``reserved == 0.0`` alone would also hold if
    the estimate were 0, which would make this test pass while guarding nothing.
    The control puts the same request under a budget smaller than the estimate:
    it is refused at the budget gate rather than reaching dispatch, which is only
    possible if the raw selector really did match pricing and really did produce
    a nonzero estimate. That is the precondition that makes the refund matter.
    """
    priced = client.post(
        "/v1/pricing",
        json={
            "model_key": "ghostprovider:some-model",
            "input_price_per_million": 2.5,
            "output_price_per_million": 10.0,
        },
        headers=master_key_header,
    )
    assert priced.status_code == 200, priced.text

    # Control: a budget far below the estimate turns the same request into a
    # budget refusal (403), proving the estimate is nonzero and pricing matched.
    tiny_budget = client.post("/v1/budgets", json={"max_budget": 0.001}, headers=master_key_header).json()[
        "budget_id"
    ]
    _make_user(client, master_key_header, "tiny-budget-user", budget_id=tiny_budget)
    assert _chat(client, master_key_header, model="ghostprovider:some-model", user="tiny-budget-user") == 403

    budget_id = client.post("/v1/budgets", json={"max_budget": 100.0}, headers=master_key_header).json()[
        "budget_id"
    ]
    _make_user(client, master_key_header, "stranded-user", budget_id=budget_id)

    assert _chat(client, master_key_header, model="ghostprovider:some-model", user="stranded-user") == 400

    session = db_session_factory()
    try:
        reserved = session.execute(
            text("SELECT reserved FROM users WHERE user_id = 'stranded-user'")
        ).scalar_one()
    finally:
        session.close()
    assert float(reserved) == 0.0

    # And the drop is still recorded, as for every other gate. Scoped by user
    # because the control above left a budget-refusal row of its own.
    rows = [r for r in _errors(client, master_key_header) if r["user_id"] == "stranded-user"]
    assert len(rows) == 1, rows
    assert rows[0]["model"] == "ghostprovider:some-model"
    assert rows[0]["cost"] is None
    assert rows[0]["counts_toward_budget"] is True


def test_budget_exempt_key_rejection_row_still_counts_toward_budget(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """An exempt key's rejection row must not look like imported usage.

    A key flagged ``exclude_from_budget`` writes ``counts_toward_budget=False``
    on its *served* rows, and it can still be refused by these gates (a block
    applies whether or not spend is enforced). If a rejection row inherited that
    flag, the dashboard would classify it as imported usage and offer it for bulk
    delete and set-price. The flag is pinned True for every gateway-written
    rejection instead; this covers the blocked-user gate, which an exempt key
    reaches (unlike the pricing gate, which is guarded by ``not budget_exempt``
    and is pinned separately in test_require_pricing.py).
    """
    _make_user(client, master_key_header, "exempt-blocked", blocked=True)
    key = _make_key(client, master_key_header, "exempt", user_id="exempt-blocked", exclude_from_budget=True)

    assert _chat(client, key, model="openai:gpt-4o") == 403

    row = _one_error(client, master_key_header)
    assert row["user_id"] == "exempt-blocked"
    assert row["counts_toward_budget"] is True


@pytest.mark.parametrize(
    ("model", "expected_status", "expected_model", "expected_provider"),
    [
        ("openai:text-embedding-3-small", 403, "text-embedding-3-small", "openai"),
        ("nosuchprovider:embed", 400, "nosuchprovider:embed", None),
    ],
)
def test_passthrough_rejections_are_recorded(
    client: TestClient,
    master_key_header: dict[str, str],
    model: str,
    expected_status: int,
    expected_model: str,
    expected_provider: str | None,
) -> None:
    """The pass-through scaffold records its gates too, not only chat.

    Both a rejection raised after the selector resolved (blocked user) and one
    raised because it did not resolve (400) have to land, or the failure count
    covers only part of the traffic the gateway is dropping.
    """
    _make_user(client, master_key_header, "blocked-embedder", blocked=True)

    resp = client.post(
        "/v1/embeddings",
        json={"model": model, "input": "hi", "user": "blocked-embedder"},
        headers=master_key_header,
    )
    assert resp.status_code == expected_status

    row = _one_error(client, master_key_header)
    assert row["endpoint"] == "/v1/embeddings"
    assert row["user_id"] == "blocked-embedder"
    assert row["model"] == expected_model
    assert row["provider"] == expected_provider


def test_passthrough_model_not_allowed_is_recorded(client: TestClient, master_key_header: dict[str, str]) -> None:
    """The pass-through allow-list gate records its 403, after refunding.

    This gate sits after the reservation on both scaffolds, so unlike the chat
    one it exercises the refund-then-log ordering.
    """
    _make_user(client, master_key_header, "scoped-embedder")
    key = _make_key(
        client, master_key_header, "scoped-embed", user_id="scoped-embedder", allowed_models=["cohere:*"]
    )

    resp = client.post(
        "/v1/embeddings",
        json={"model": "openai:text-embedding-3-small", "input": "hi"},
        headers=key,
    )
    assert resp.status_code == 403

    row = _one_error(client, master_key_header)
    assert row["endpoint"] == "/v1/embeddings"
    assert row["user_id"] == "scoped-embedder"
    assert row["model"] == "text-embedding-3-small"
    assert row["provider"] == "openai"
    assert "not permitted" in row["error_message"]


def test_passthrough_user_key_mismatch_is_recorded(client: TestClient, master_key_header: dict[str, str]) -> None:
    """The pass-through scaffold records a user/key mismatch like the pipeline does."""
    _make_user(client, master_key_header, "embed-owner")
    _make_user(client, master_key_header, "embed-stranger")
    key = _make_key(client, master_key_header, "embed-owned", user_id="embed-owner")

    resp = client.post(
        "/v1/embeddings",
        json={"model": "openai:text-embedding-3-small", "input": "hi", "user": "embed-stranger"},
        headers=key,
    )
    assert resp.status_code == 403

    row = _one_error(client, master_key_header)
    assert row["user_id"] == "embed-owner"
    # Nothing is resolved this early, so the row carries the raw selector.
    assert row["model"] == "openai:text-embedding-3-small"
    assert row["provider"] is None


def test_unknown_user_rejection_writes_no_row(client: TestClient, master_key_header: dict[str, str]) -> None:
    """The 404 for a user that does not exist stays unlogged, deliberately.

    ``usage_logs.user_id`` is a foreign key to ``users``, so a row naming a
    nonexistent user could not be inserted; the writer would drop it.
    """
    assert _chat(client, master_key_header, model="openai:gpt-4o", user="ghost-user") == 404
    assert _errors(client, master_key_header) == []


def test_auth_failure_writes_no_row(client: TestClient, master_key_header: dict[str, str]) -> None:
    """A 401 stays unlogged: it is refused before any user is known.

    Decided explicitly rather than by omission (#465): a row with no user_id
    could not be attributed, and logging unauthenticated requests would let an
    anonymous caller write into the usage table.
    """
    assert _chat(client, {API_KEY_HEADER: "Bearer not-a-real-key"}, model="openai:gpt-4o") == 401
    assert _errors(client, master_key_header) == []
