"""Integration tests for POST /v1/usage/external-events.

Covers auth, content-free validation, idempotency, historical + cache pricing,
budget isolation, and the read-surface (source filter, by_source, CSV).
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.models.entities import UsageLog, User

_SRC = "claude_code"
_MODEL_KEY = "anthropic:claude-sonnet-4-6"


def _seed_user(client: TestClient, master_key_header: dict[str, str], user_id: str = "cc-user") -> str:
    resp = client.post("/v1/users", json={"user_id": user_id}, headers=master_key_header)
    assert resp.status_code == 200
    return user_id


def _seed_pricing(
    client: TestClient,
    master_key_header: dict[str, str],
    *,
    input_price: float = 3.0,
    output_price: float = 15.0,
    cache_read_price: float | None = 0.3,
    cache_write_price: float | None = 3.75,
    # Default far in the past so any event timestamp resolves this rate. Historical
    # pricing prices an event at the rate effective AT its timestamp, so a price
    # effective "now" would not apply to a backdated event.
    effective_at: str | None = "2020-01-01T00:00:00Z",
) -> None:
    body: dict[str, Any] = {
        "model_key": _MODEL_KEY,
        "input_price_per_million": input_price,
        "output_price_per_million": output_price,
        "cache_read_price_per_million": cache_read_price,
        "cache_write_price_per_million": cache_write_price,
    }
    if effective_at is not None:
        body["effective_at"] = effective_at
    resp = client.post("/v1/pricing", json=body, headers=master_key_header)
    assert resp.status_code == 200, resp.text


# An hour ago, not a fixed date. `GET /v1/usage/summary` bounds itself to the
# last 30 days when the caller names no window (`_DEFAULT_SUMMARY_LOOKBACK`), so
# a literal timestamp puts every summary assertion in this file on a fuse: it
# passes until the day the clock is 30 days past it, then fails everywhere at
# once with nothing in the diff to explain it. That day was 2026-08-21, between
# one green run on `main` and the next.
#
# The per-run pricing revision this would churn on a *shared* database (see
# `web/e2e/parity-data.ts`, which backdates for exactly that reason) is not a
# concern here: the integration fixtures own their database and drop it in
# teardown.
def _event_timestamp() -> str:
    return (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")


def _event(source_event_id: str = "req_1", **overrides: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "source_event_id": source_event_id,
        "timestamp": _event_timestamp(),
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "status": "success",
        "input_tokens": 1200,
        "output_tokens": 450,
        "cache_read_tokens": 8000,
        "cache_write_tokens": 1024,
        "duration_ms": 4821,
        "session_label": "project:otari",
    }
    event.update(overrides)
    return event


def _post(
    client: TestClient,
    headers: dict[str, str],
    events: list[dict[str, Any]],
    *,
    user_id: str | None = "cc-user",
    source: str = _SRC,
) -> Any:
    body: dict[str, Any] = {"source": source, "events": events}
    if user_id is not None:
        body["user_id"] = user_id
    return client.post("/v1/usage/external-events", json=body, headers=headers)


def _make_key(
    client: TestClient,
    master_key_header: dict[str, str],
    user_id: str,
    *,
    exclude_from_budget: bool = True,
    reject_user_mismatch: bool | None = None,
) -> dict[str, str]:
    """Create an API key bound to `user_id` and return its auth header.

    Import keys must be budget-exempt, so that is the default here.
    """
    resp = client.post(
        "/v1/keys",
        json={
            "key_name": f"importer-{user_id}",
            "user_id": user_id,
            "exclude_from_budget": exclude_from_budget,
            "reject_user_mismatch": reject_user_mismatch,
        },
        headers=master_key_header,
    )
    assert resp.status_code == 200, resp.text
    return {"Otari-Key": f"Bearer {resp.json()['key']}"}


def test_requires_auth(client: TestClient) -> None:
    """No credential -> rejected, nothing ingested."""
    resp = client.post(
        "/v1/usage/external-events",
        json={"source": _SRC, "user_id": "cc-user", "events": [_event()]},
    )
    assert resp.status_code in (401, 403)


def test_non_exempt_key_cannot_import(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """Importing is refused for a key that is not budget-exempt: retrospective usage
    cannot be budget-enforced, so it must not run through a budgeted key."""
    _seed_user(client, master_key_header, "dev-a")
    headers = _make_key(client, master_key_header, "dev-a", exclude_from_budget=False)
    resp = _post(client, headers, [_event("nope")], user_id=None)
    assert resp.status_code == 403
    assert "budget-exempt" in resp.json()["detail"]
    assert db_session.query(UsageLog).filter(UsageLog.source_event_id == "nope").count() == 0


def test_api_key_ingests_for_own_user(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """A budget-exempt API key imports for its own user (no user_id needed), stamps
    its id, and the row never counts toward budget."""
    _seed_user(client, master_key_header, "dev-a")
    _seed_pricing(client, master_key_header)
    headers = _make_key(client, master_key_header, "dev-a")

    resp = _post(client, headers, [_event("via_key")], user_id=None)
    assert resp.status_code == 200, resp.text
    assert resp.json()["accepted"] == 1

    row = db_session.query(UsageLog).filter(UsageLog.source_event_id == "via_key").one()
    assert row.user_id == "dev-a"
    assert row.api_key_id is not None  # attributed to the importing key
    assert row.counts_toward_budget is False


def test_api_key_cannot_attribute_to_another_user(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """A key naming a different user is rejected (default strict binding)."""
    _seed_user(client, master_key_header, "dev-a")
    _seed_user(client, master_key_header, "dev-b")
    _seed_pricing(client, master_key_header)
    headers = _make_key(client, master_key_header, "dev-a")

    resp = _post(client, headers, [_event("cross")], user_id="dev-b")
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] == 0
    assert body["rejected"] == 1
    assert "does not match" in body["errors"][0]["detail"]
    assert db_session.query(UsageLog).filter(UsageLog.source_event_id == "cross").count() == 0


def test_lenient_key_binds_foreign_user_to_its_own(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """A key overriding reject_user_mismatch to false accepts a foreign user_id and
    still binds the usage to its own user, matching the request path (issue #493)."""
    _seed_user(client, master_key_header, "dev-a")
    _seed_user(client, master_key_header, "dev-b")
    _seed_pricing(client, master_key_header)
    headers = _make_key(client, master_key_header, "dev-a", reject_user_mismatch=False)

    resp = _post(client, headers, [_event("lenient")], user_id="dev-b")
    assert resp.status_code == 200, resp.text
    assert resp.json()["accepted"] == 1

    row = db_session.query(UsageLog).filter(UsageLog.source_event_id == "lenient").one()
    assert row.user_id == "dev-a"


def test_accepts_and_prices_with_cache(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """A valid event is accepted, attributed, and priced including cache tokens."""
    _seed_user(client, master_key_header)
    _seed_pricing(client, master_key_header)

    resp = _post(client, master_key_header, [_event()])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {"accepted": 1, "duplicate": 0, "rejected": 0, "errors": []}

    row = db_session.query(UsageLog).filter(UsageLog.source == _SRC).one()
    assert row.source_event_id == "req_1"
    assert row.user_id == "cc-user"
    assert row.endpoint == "external"
    assert row.counts_toward_budget is False
    assert row.source_label == "project:otari"
    # 1200 input + 450 output + 8000 cache-read + 1024 cache-write, all additive.
    # Cost must be > the input/output-only cost, proving cache tokens were priced.
    io_only = (1200 / 1e6) * 3.0 + (450 / 1e6) * 15.0
    assert row.cost is not None and row.cost > io_only
    assert row.billing_meters and row.billing_meters.get("cache_read_tokens") == 8000


def test_no_pricing_lands_with_null_cost(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """Imported usage is budget-exempt, so missing pricing is not an error: the row
    lands with cost=null rather than being rejected."""
    _seed_user(client, master_key_header)
    # Deliberately DO NOT seed pricing for this model.
    resp = _post(client, master_key_header, [_event("unpriced", model="brand-new-model")])
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"accepted": 1, "duplicate": 0, "rejected": 0, "errors": []}

    row = db_session.query(UsageLog).filter(UsageLog.source_event_id == "unpriced").one()
    assert row.cost is None
    assert row.counts_toward_budget is False

    # The summary reports the unpriced row so a $0 cost is not read as free.
    summary = client.get("/v1/usage/summary", headers=master_key_header).json()
    assert summary["totals"]["unpriced_requests"] == 1


def test_unpriced_inclusive_import_reprices_under_the_convention_it_arrived_with(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """An OpenAI-shaped import with no rate at ingest must still reprice inclusively.

    The row lands unpriced (no rate row for its model), so it has no billing
    meters, and the meter-based recovery in ``usage_admin_service`` cannot say
    which convention it arrived under. The stored ``cache_tokens_in_prompt`` is
    what answers: without it the repricing would add the cached tokens to a
    prompt that already contained them and settle 1500 input tokens for a row
    that reported 1000. mozilla-ai/otari#690.
    """
    _seed_user(client, master_key_header)
    # Deliberately no pricing for this model, so the row is ingested unpriced.
    resp = _post(
        client,
        master_key_header,
        [
            _event(
                "inclusive-unpriced",
                provider="openai",
                model="brand-new-model",
                input_tokens=1000,
                output_tokens=200,
                cache_read_tokens=500,
                cache_write_tokens=0,
                cache_tokens_in_prompt=True,
            )
        ],
    )
    assert resp.status_code == 200, resp.text

    row = db_session.query(UsageLog).filter(UsageLog.source_event_id == "inclusive-unpriced").one()
    assert row.cost is None
    assert row.billing_meters is None
    assert row.cache_tokens_in_prompt is True

    priced = client.post(
        "/v1/usage/set-price",
        json={
            "ids": [row.id],
            "input_price_per_million": 3.0,
            "output_price_per_million": 15.0,
            "cache_read_price_per_million": 0.3,
        },
        headers=master_key_header,
    )
    assert priced.status_code == 200, priced.text
    assert priced.json()["updated"] == 1

    db_session.expire_all()
    row = db_session.query(UsageLog).filter(UsageLog.source_event_id == "inclusive-unpriced").one()
    # Inclusive: 500 fresh input * 3/1M + 200 output * 15/1M + 500 cached * 0.3/1M.
    # The additive reading of the same numbers would be 0.00615, a 32% overcharge.
    assert row.cost == Decimal("0.00465")
    assert row.billing_meters is not None
    assert row.billing_meters["total_input_tokens"] == 1000


def test_ingest_records_the_additive_default(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """An event that states no convention is recorded as additive, not as unrecorded."""
    _seed_user(client, master_key_header)
    _seed_pricing(client, master_key_header)
    resp = _post(client, master_key_header, [_event("default-shape")])
    assert resp.status_code == 200, resp.text

    row = db_session.query(UsageLog).filter(UsageLog.source_event_id == "default-shape").one()
    assert row.cache_tokens_in_prompt is False


def test_idempotent_resubmit(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """Re-posting the same (source, source_event_id) is a duplicate, not a new row."""
    _seed_user(client, master_key_header)
    _seed_pricing(client, master_key_header)

    first = _post(client, master_key_header, [_event("req_dup")])
    assert first.json()["accepted"] == 1
    second = _post(client, master_key_header, [_event("req_dup")])
    assert second.json() == {"accepted": 0, "duplicate": 1, "rejected": 0, "errors": []}

    assert db_session.query(UsageLog).filter(UsageLog.source_event_id == "req_dup").count() == 1


def test_dedupes_within_batch(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """Two events with the same id in one batch collapse to a single row."""
    _seed_user(client, master_key_header)
    _seed_pricing(client, master_key_header)

    resp = _post(client, master_key_header, [_event("same"), _event("same")])
    body = resp.json()
    assert body["accepted"] == 1
    assert body["duplicate"] == 1
    assert db_session.query(UsageLog).filter(UsageLog.source_event_id == "same").count() == 1


def test_unknown_user_rejected_per_event(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """An event attributed to a missing user is rejected with an actionable error."""
    _seed_user(client, master_key_header)
    _seed_pricing(client, master_key_header)

    resp = _post(
        client,
        master_key_header,
        [_event("ok"), _event("bad", user_id="ghost")],
    )
    body = resp.json()
    assert body["accepted"] == 1
    assert body["rejected"] == 1
    assert body["errors"][0]["source_event_id"] == "bad"
    assert "not found" in body["errors"][0]["detail"]
    assert db_session.query(UsageLog).filter(UsageLog.source_event_id == "bad").count() == 0


def test_rejects_content_fields(client: TestClient, master_key_header: dict[str, str]) -> None:
    """A stray prompt/content field is rejected (extra=forbid), never stored."""
    _seed_user(client, master_key_header)
    event = _event()
    event["prompt"] = "secret user text"
    resp = _post(client, master_key_header, [event])
    assert resp.status_code == 422


def test_rejects_content_fields_at_batch_level(client: TestClient, master_key_header: dict[str, str]) -> None:
    """The batch envelope forbids extra fields too, not just the per-event schema."""
    _seed_user(client, master_key_header)
    resp = client.post(
        "/v1/usage/external-events",
        json={"source": _SRC, "user_id": "cc-user", "events": [_event()], "prompt": "secret user text"},
        headers=master_key_header,
    )
    assert resp.status_code == 422


def test_oversized_batch_rejected(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Batches over the cap are rejected before any work."""
    _seed_user(client, master_key_header)
    events = [_event(f"req_{i}") for i in range(1001)]
    resp = _post(client, master_key_header, events)
    assert resp.status_code == 422


def test_budget_isolation(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """Imported cost never touches users.spend or users.reserved."""
    _seed_user(client, master_key_header)
    _seed_pricing(client, master_key_header)

    resp = _post(client, master_key_header, [_event("iso")])
    assert resp.json()["accepted"] == 1

    user = db_session.query(User).filter(User.user_id == "cc-user").one()
    assert float(user.spend) == pytest.approx(0.0)
    assert float(user.reserved) == pytest.approx(0.0)


def test_historical_pricing(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """An event is priced at the rate effective at its own timestamp."""
    _seed_user(client, master_key_header)
    # Old cheap rate effective a year before the event; new expensive rate after.
    # Both relative to the event, for the reason `_event_timestamp` gives: the
    # boundary this test is about is "before and after the event", never a date.
    now = datetime.now(UTC)
    _seed_pricing(
        client,
        master_key_header,
        input_price=1.0,
        output_price=1.0,
        cache_read_price=0.0,
        cache_write_price=0.0,
        effective_at=(now - timedelta(days=365)).isoformat().replace("+00:00", "Z"),
    )
    _seed_pricing(
        client,
        master_key_header,
        input_price=1000.0,
        output_price=1000.0,
        cache_read_price=0.0,
        cache_write_price=0.0,
        effective_at=(now + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
    )

    # The event is an hour old, so it precedes the new rate and is priced cheap.
    resp = _post(
        client,
        master_key_header,
        [_event("hist", input_tokens=1_000_000, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0)],
    )
    assert resp.json()["accepted"] == 1
    row = db_session.query(UsageLog).filter(UsageLog.source_event_id == "hist").one()
    assert row.cost == pytest.approx(1.0)  # 1M input * $1/M, cheap rate


def test_read_surface_source_filter_and_summary(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Imported rows are exposed + labeled via list, summary, and CSV."""
    _seed_user(client, master_key_header)
    _seed_pricing(client, master_key_header)
    assert _post(client, master_key_header, [_event("read_1")]).json()["accepted"] == 1

    listed = client.get("/v1/usage", params={"source": _SRC}, headers=master_key_header)
    assert listed.status_code == 200
    rows = listed.json()
    assert len(rows) == 1
    assert rows[0]["source"] == _SRC
    assert rows[0]["source_label"] == "project:otari"
    assert rows[0]["counts_toward_budget"] is False

    summary = client.get("/v1/usage/summary", headers=master_key_header).json()
    sources = {r["key"]: r for r in summary["by_source"]}
    assert _SRC in sources and sources[_SRC]["requests"] == 1

    csv_resp = client.get("/v1/usage/summary.csv", headers=master_key_header)
    assert csv_resp.status_code == 200
    assert "source" in csv_resp.text and _SRC in csv_resp.text


def test_per_event_user_override(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """A per-event user_id overrides the batch default so one feed serves a team."""
    _seed_user(client, master_key_header, "dev-a")
    _seed_user(client, master_key_header, "dev-b")
    _seed_pricing(client, master_key_header)

    resp = _post(
        client,
        master_key_header,
        [_event("e_a"), _event("e_b", user_id="dev-b")],
        user_id="dev-a",
    )
    assert resp.json()["accepted"] == 2
    a = db_session.query(UsageLog).filter(UsageLog.source_event_id == "e_a").one()
    b = db_session.query(UsageLog).filter(UsageLog.source_event_id == "e_b").one()
    assert a.user_id == "dev-a"
    assert b.user_id == "dev-b"


def test_rejects_reserved_gateway_source(client: TestClient, master_key_header: dict[str, str]) -> None:
    """An import claiming source=gateway would masquerade as native traffic."""
    _seed_user(client, master_key_header)
    resp = _post(client, master_key_header, [_event()], source="gateway")
    assert resp.status_code == 422
    assert "reserved" in resp.text


def test_rejects_reserved_otari_ai_source_prefix(client: TestClient, master_key_header: dict[str, str]) -> None:
    """otari.ai stamps the rows its backfill writes `otari-ai:<slug>`; an import under
    that prefix is a lookalike in every reconciliation total."""
    _seed_user(client, master_key_header)
    for source in ("otari-ai:gateway", "otari-ai:claude_code", "OTARI-AI:gateway"):
        resp = _post(client, master_key_header, [_event()], source=source)
        assert resp.status_code == 422, f"{source}: {resp.text}"
        assert "reserved" in resp.text


def test_accepts_source_with_colon_outside_the_reserved_prefix(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """The guard is a prefix check, not a ban on colons in a slug."""
    _seed_user(client, master_key_header)
    _seed_pricing(client, master_key_header)

    resp = _post(client, master_key_header, [_event("colon_ok")], source="foo:bar")
    assert resp.status_code == 200, resp.text
    assert resp.json()["accepted"] == 1

    row = db_session.query(UsageLog).filter(UsageLog.source_event_id == "colon_ok").one()
    assert row.source == "foo:bar"


def test_rejects_token_counts_above_column_width(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Counts past the 32-bit column cap are a 422, not a DB error that 500s the batch."""
    _seed_user(client, master_key_header)
    resp = _post(client, master_key_header, [_event(input_tokens=2_147_483_648)])
    assert resp.status_code == 422
