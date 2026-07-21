"""Dashboard-facing budget endpoints: per-budget usage rollup and reset history."""

from datetime import UTC, datetime

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.models.entities import BudgetResetLog, User


def _make_budget(client: TestClient, headers: dict[str, str], max_budget: float | None = 100.0) -> str:
    response = client.post("/v1/budgets", json={"max_budget": max_budget}, headers=headers)
    assert response.status_code == 200, response.json()
    budget_id: str = response.json()["budget_id"]
    return budget_id


def test_budget_name_roundtrips_and_clears(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Name is stored on create, renamed on patch, and cleared by an explicit null."""
    created = client.post(
        "/v1/budgets", json={"name": "team-free-tier", "max_budget": 25.0}, headers=master_key_header
    ).json()
    assert created["name"] == "team-free-tier"
    budget_id = created["budget_id"]

    renamed = client.patch(f"/v1/budgets/{budget_id}", json={"name": "team-pro"}, headers=master_key_header).json()
    assert renamed["name"] == "team-pro"

    # Explicit null clears back to unnamed; the limit is untouched.
    cleared = client.patch(f"/v1/budgets/{budget_id}", json={"name": None}, headers=master_key_header).json()
    assert cleared["name"] is None
    assert cleared["max_budget"] == 25.0


def test_new_budget_reports_zero_rollup(client: TestClient, master_key_header: dict[str, str]) -> None:
    """A budget with no assigned users reports zeros, not nulls or an error."""
    budget_id = _make_budget(client, master_key_header)

    data = client.get(f"/v1/budgets/{budget_id}", headers=master_key_header).json()
    assert data["user_count"] == 0
    assert data["total_spend"] == 0.0
    assert data["total_reserved"] == 0.0


def test_budget_rollup_aggregates_assigned_users(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """The rollup sums spend/reserved and counts the users assigned to a budget."""
    budget_id = _make_budget(client, master_key_header)

    for user_id in ("roll-a", "roll-b"):
        assert (
            client.post(
                "/v1/users",
                json={"user_id": user_id, "budget_id": budget_id},
                headers=master_key_header,
            ).status_code
            == 200
        )

    # Seed spend/reserved directly; there is no API to set them without a live call.
    users = db_session.execute(select(User).where(User.budget_id == budget_id)).scalars().all()
    users[0].spend = 10.0
    users[0].reserved = 1.5
    users[1].spend = 4.0
    users[1].reserved = 0.5
    db_session.commit()

    # Single-budget aggregate.
    data = client.get(f"/v1/budgets/{budget_id}", headers=master_key_header).json()
    assert data["user_count"] == 2
    assert data["total_spend"] == 14.0
    assert data["total_reserved"] == 2.0

    # Same numbers from the grouped list query.
    listed = client.get("/v1/budgets", headers=master_key_header).json()
    row = next(b for b in listed if b["budget_id"] == budget_id)
    assert row["user_count"] == 2
    assert row["total_spend"] == 14.0
    assert row["total_reserved"] == 2.0


def test_budget_rollup_excludes_deleted_users(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """A soft-deleted user drops out of the budget's rollup."""
    budget_id = _make_budget(client, master_key_header)
    client.post("/v1/users", json={"user_id": "gone", "budget_id": budget_id}, headers=master_key_header)

    assert client.get(f"/v1/budgets/{budget_id}", headers=master_key_header).json()["user_count"] == 1

    assert client.delete("/v1/users/gone", headers=master_key_header).status_code == 204
    assert client.get(f"/v1/budgets/{budget_id}", headers=master_key_header).json()["user_count"] == 0


def test_reset_logs_returned_newest_first(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """The reset-logs endpoint surfaces BudgetResetLog rows, most recent first."""
    budget_id = _make_budget(client, master_key_header)
    client.post("/v1/users", json={"user_id": "resetter", "budget_id": budget_id}, headers=master_key_header)

    db_session.add_all(
        [
            BudgetResetLog(
                user_id="resetter",
                budget_id=budget_id,
                previous_spend=5.0,
                reset_at=datetime(2026, 1, 1, tzinfo=UTC),
                next_reset_at=datetime(2026, 1, 8, tzinfo=UTC),
            ),
            BudgetResetLog(
                user_id="resetter",
                budget_id=budget_id,
                previous_spend=7.0,
                reset_at=datetime(2026, 1, 8, tzinfo=UTC),
                next_reset_at=datetime(2026, 1, 15, tzinfo=UTC),
            ),
        ]
    )
    db_session.commit()

    logs = client.get(f"/v1/budgets/{budget_id}/reset-logs", headers=master_key_header).json()
    assert [log["previous_spend"] for log in logs] == [7.0, 5.0]
    assert logs[0]["user_id"] == "resetter"
    assert logs[0]["budget_id"] == budget_id
    assert logs[0]["next_reset_at"] is not None


def test_reset_logs_empty_for_fresh_budget(client: TestClient, master_key_header: dict[str, str]) -> None:
    budget_id = _make_budget(client, master_key_header)
    assert client.get(f"/v1/budgets/{budget_id}/reset-logs", headers=master_key_header).json() == []


def test_reset_logs_unknown_budget_404(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.get("/v1/budgets/does-not-exist/reset-logs", headers=master_key_header)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
