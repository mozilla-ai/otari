"""Integration tests for ``GET /v1/pricing/current``.

``GET /v1/pricing`` answers the stored history, one row per ``effective_at``, so
a page of it is a page of revisions: pricing three models twice gives six rows
and no way to ask for the first two models. This route answers one row per key
with a total, which is what lets the dashboard's price table page instead of
reading the collection (otari#1376).
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT
from gateway.models.pricing import ModelPricing


def _price(
    client: TestClient,
    headers: dict[str, str],
    model_key: str,
    *,
    rate: float,
    effective_at: datetime | None = None,
) -> None:
    body: dict[str, object] = {
        "model_key": model_key,
        "input_price_per_million": rate,
        "output_price_per_million": rate * 2,
    }
    if effective_at is not None:
        body["effective_at"] = effective_at.isoformat()
    response = client.post(f"{API_ROOT}/pricing", json=body, headers=headers)
    assert response.status_code == 200, response.text


def test_current_answers_one_row_per_model_not_one_per_revision(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A repriced model is one row, at the rate in force."""

    now = datetime.now(UTC)
    _price(client, master_key_header, "openai:gpt-4o", rate=1, effective_at=now - timedelta(days=2))
    _price(client, master_key_header, "openai:gpt-4o", rate=3, effective_at=now - timedelta(days=1))

    response = client.get(f"{API_ROOT}/pricing/current", headers=master_key_header)

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert [row["model_key"] for row in body["data"]] == ["openai:gpt-4o"]
    assert body["data"][0]["input_price_per_million"] == 3
    # The history is still the history: both revisions are there to be read.
    history = client.get(f"{API_ROOT}/pricing", headers=master_key_header).json()
    assert len([row for row in history if row["model_key"] == "openai:gpt-4o"]) == 2


def test_a_future_rate_does_not_displace_the_one_in_force(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A price scheduled for later is not what a request is metered at today."""

    now = datetime.now(UTC)
    _price(client, master_key_header, "openai:gpt-4o", rate=1, effective_at=now - timedelta(days=1))
    _price(client, master_key_header, "openai:gpt-4o", rate=99, effective_at=now + timedelta(days=30))

    body = client.get(f"{API_ROOT}/pricing/current", headers=master_key_header).json()

    assert body["count"] == 1
    assert body["data"][0]["input_price_per_million"] == 1


def test_a_key_priced_only_for_later_still_appears(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A queued rate is shown rather than dropped.

    The catalog is where an operator sees what they have set, so a key whose only
    row is scheduled falls back to the earliest of them. Settlement disagrees on
    purpose: ``rates_in_force`` has no rate for such a key, because none applies.
    """

    now = datetime.now(UTC)
    _price(client, master_key_header, "openai:gpt-5", rate=7, effective_at=now + timedelta(days=10))
    _price(client, master_key_header, "openai:gpt-5", rate=9, effective_at=now + timedelta(days=20))

    body = client.get(f"{API_ROOT}/pricing/current", headers=master_key_header).json()

    assert body["count"] == 1
    assert body["data"][0]["input_price_per_million"] == 7


def test_the_page_is_a_window_on_models_and_the_count_is_all_of_them(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """``skip`` and ``limit`` walk models, and ``count`` is the total."""

    now = datetime.now(UTC)
    for index in range(5):
        # Two revisions each, so a route paging revisions rather than models
        # would answer this differently.
        _price(client, master_key_header, f"openai:m{index}", rate=1, effective_at=now - timedelta(days=2))
        _price(client, master_key_header, f"openai:m{index}", rate=2, effective_at=now - timedelta(days=1))

    first = client.get(f"{API_ROOT}/pricing/current?skip=0&limit=2", headers=master_key_header).json()
    second = client.get(f"{API_ROOT}/pricing/current?skip=2&limit=2", headers=master_key_header).json()
    last = client.get(f"{API_ROOT}/pricing/current?skip=4&limit=2", headers=master_key_header).json()

    assert first["count"] == second["count"] == last["count"] == 5
    assert [row["model_key"] for row in first["data"]] == ["openai:m0", "openai:m1"]
    assert [row["model_key"] for row in second["data"]] == ["openai:m2", "openai:m3"]
    assert [row["model_key"] for row in last["data"]] == ["openai:m4"]


def test_a_model_stored_under_both_spellings_is_reported_once(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """The legacy ``provider/model`` key is dropped where the canonical one exists.

    A lookup resolves such a model to the ``provider:model`` row, so listing the
    legacy one would name a rate nothing is metered at and count one model twice.
    ``rates_in_force`` drops it too; the two have to agree. Written straight to
    the table because ``POST /v1/pricing`` normalizes the key, which is why only
    older rows carry the legacy form.
    """

    now = datetime.now(UTC)
    _price(client, master_key_header, "openai:gpt-4o", rate=3, effective_at=now - timedelta(days=1))
    db_session.add(
        ModelPricing(
            model_key="openai/gpt-4o",
            effective_at=now - timedelta(days=1),
            input_price_per_million=Decimal(99),
            output_price_per_million=Decimal(99),
        )
    )
    db_session.commit()

    body = client.get(f"{API_ROOT}/pricing/current", headers=master_key_header).json()

    assert body["count"] == 1
    assert [row["model_key"] for row in body["data"]] == ["openai:gpt-4o"]


def test_a_legacy_key_with_no_canonical_twin_is_still_listed(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """Dropping it would hide the only rate that model has."""

    now = datetime.now(UTC)
    db_session.add(
        ModelPricing(
            model_key="openai/gpt-4o",
            effective_at=now - timedelta(days=1),
            input_price_per_million=Decimal(5),
            output_price_per_million=Decimal(5),
        )
    )
    db_session.commit()

    body = client.get(f"{API_ROOT}/pricing/current", headers=master_key_header).json()

    assert body["count"] == 1
    assert [row["model_key"] for row in body["data"]] == ["openai/gpt-4o"]


def test_current_is_a_route_rather_than_a_model_key(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """``/pricing/current`` is not swallowed by the ``{model_key}`` catch-all."""

    response = client.get(f"{API_ROOT}/pricing/current", headers=master_key_header)

    assert response.status_code == 200
    assert response.json() == {"data": [], "count": 0}


def test_the_window_is_bounded_like_every_other_list_route(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A limit past the ceiling is refused rather than clamped."""

    assert client.get(f"{API_ROOT}/pricing/current?limit=1001", headers=master_key_header).status_code == 422
    assert client.get(f"{API_ROOT}/pricing/current?skip=-1", headers=master_key_header).status_code == 422
