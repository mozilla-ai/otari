"""Dashboard-facing budget endpoints: per-budget usage rollup and reset history."""

from datetime import UTC, datetime
from decimal import Decimal

from fastapi.testclient import TestClient
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from gateway.models.entities import BudgetResetLog, ScopedBudget, User, WorkspaceBudgetDefault
from gateway.models.tenancy import Organization, Workspace


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
    users[0].spend = Decimal("10.0")
    users[0].reserved = Decimal("1.5")
    users[1].spend = Decimal("4.0")
    users[1].reserved = Decimal("0.5")
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


def test_deleting_a_budget_a_workspace_hands_out_is_refused_by_name(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """A budget that is a workspace's member default cannot be deleted out from under it.

    The foreign key is RESTRICT, so the database would refuse this anyway, but as
    an opaque integrity error. The route checks first so the refusal names the
    workspace an operator has to go and change, and so the answer does not depend
    on whether the engine is enforcing foreign keys (SQLite only does with
    ``PRAGMA foreign_keys`` on).
    """
    budget_id = _make_budget(client, master_key_header)
    organization = Organization(name="Acme", slug="acme-delete-guard")
    db_session.add(organization)
    db_session.flush()
    workspace = Workspace(organization_id=organization.id, name="Research")
    db_session.add(workspace)
    db_session.flush()
    db_session.add(WorkspaceBudgetDefault(workspace_id=workspace.id, budget_id=budget_id))
    db_session.commit()

    refused = client.delete(f"/v1/budgets/{budget_id}", headers=master_key_header)
    assert refused.status_code == 409, refused.text
    assert "Research" in refused.json()["detail"]

    # Still there, and deletable once nothing hands it out.
    assert client.get(f"/v1/budgets/{budget_id}", headers=master_key_header).status_code == 200
    db_session.execute(delete(WorkspaceBudgetDefault).where(WorkspaceBudgetDefault.budget_id == budget_id))
    db_session.commit()
    assert client.delete(f"/v1/budgets/{budget_id}", headers=master_key_header).status_code == 204


def test_an_explicit_null_budget_detaches_and_clears_the_reset_clock(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Null is a state the assignment has to be able to reach.

    ``budget_id`` was gated on ``is not None``, so a budget was assignable and
    never removable. The dashboard's deselect writes exactly this null, got a 200
    back, and reported success while the person stayed on the budget. The reset
    clock goes with the assignment: one pointing at a budget nobody is on would
    fire against nothing.
    """
    # With a cadence, so the reset clock is non-null while attached and the
    # clearing below is visible rather than vacuously true.
    created = client.post(
        "/v1/budgets",
        json={"max_budget": 100.0, "budget_duration_sec": 86400},
        headers=master_key_header,
    )
    assert created.status_code == 200, created.text
    budget_id = created.json()["budget_id"]
    assert client.post("/v1/users", json={"user_id": "alice"}, headers=master_key_header).status_code == 200

    attached = client.patch("/v1/users/alice", json={"budget_id": budget_id}, headers=master_key_header)
    assert attached.status_code == 200, attached.text
    assert attached.json()["budget_id"] == budget_id
    assert attached.json()["next_budget_reset_at"] is not None

    detached = client.patch("/v1/users/alice", json={"budget_id": None}, headers=master_key_header)
    assert detached.status_code == 200, detached.text
    assert detached.json()["budget_id"] is None
    assert detached.json()["next_budget_reset_at"] is None
    assert detached.json()["budget_started_at"] is None

    # Omitting the field is still "leave it alone", which is the half that
    # already worked and must keep working.
    reattached = client.patch("/v1/users/alice", json={"budget_id": budget_id}, headers=master_key_header)
    assert reattached.json()["budget_id"] == budget_id
    renamed = client.patch("/v1/users/alice", json={"alias": "Alice"}, headers=master_key_header)
    assert renamed.json()["budget_id"] == budget_id


def test_a_calendar_aligned_budget_gives_a_user_a_boundary_reset(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The user plane reads both cadences, not just the duration.

    ``/v1/budgets`` accepts ``reset_alignment``, and the assignment path used to
    read only ``budget_duration_sec``, so a user on a calendar-aligned budget got
    a null next reset. A null next reset never fires, so their spend never
    refilled and they were eventually refused permanently.
    """
    monthly = client.post(
        "/v1/budgets",
        json={"max_budget": 100.0, "reset_alignment": "calendar_month"},
        headers=master_key_header,
    )
    assert monthly.status_code == 200, monthly.text
    assert monthly.json()["reset_alignment"] == "calendar_month"

    assert client.post("/v1/users", json={"user_id": "bruno"}, headers=master_key_header).status_code == 200
    assigned = client.patch(
        "/v1/users/bruno",
        json={"budget_id": monthly.json()["budget_id"]},
        headers=master_key_header,
    )

    assert assigned.status_code == 200, assigned.text
    reset_at = assigned.json()["next_budget_reset_at"]
    assert reset_at is not None, "a calendar cadence has to produce a reset the sweep can fire"
    # The boundary, not "a month from now": everyone on this budget rolls together.
    assert datetime.fromisoformat(reset_at).day == 1


def test_deleting_a_budget_a_ceiling_enforces_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """The other RESTRICT reference, and the one with no page to go and clear.

    A scoped ceiling names a budget directly. Deleting it out from under one
    would be refused by the database anyway, as an opaque integrity error; this
    says how many hold it so the refusal is actionable.
    """
    budget_id = _make_budget(client, master_key_header)
    db_session.add(
        ScopedBudget(scope_type="organization", scope_id="org-1", budget_id=budget_id),
    )
    db_session.commit()

    refused = client.delete(f"/v1/budgets/{budget_id}", headers=master_key_header)
    assert refused.status_code == 409, refused.text
    assert "1 spend ceiling" in refused.json()["detail"]

    db_session.execute(delete(ScopedBudget).where(ScopedBudget.budget_id == budget_id))
    db_session.commit()
    assert client.delete(f"/v1/budgets/{budget_id}", headers=master_key_header).status_code == 204
