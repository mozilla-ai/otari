"""Dashboard-facing budget endpoints: per-budget usage rollup and reset history."""

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT
from gateway.models.budgets import Budget, BudgetResetLog, ScopedBudget, WorkspaceBudgetDefault
from gateway.models.tenancy import Organization, Workspace
from gateway.models.users import User


def _make_budget(client: TestClient, headers: dict[str, str], max_budget: float | None = 100.0) -> str:
    response = client.post(f"{API_ROOT}/budgets", json={"max_budget": max_budget}, headers=headers)
    assert response.status_code == 200, response.json()
    budget_id: str = response.json()["budget_id"]
    return budget_id


def test_budget_name_roundtrips_and_clears(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Name is stored on create, renamed on patch, and cleared by an explicit null."""
    created = client.post(
        f"{API_ROOT}/budgets", json={"name": "team-free-tier", "max_budget": 25.0}, headers=master_key_header
    ).json()
    assert created["name"] == "team-free-tier"
    budget_id = created["budget_id"]

    renamed = client.patch(
        f"{API_ROOT}/budgets/{budget_id}", json={"name": "team-pro"}, headers=master_key_header
    ).json()
    assert renamed["name"] == "team-pro"

    # Explicit null clears back to unnamed; the limit is untouched.
    cleared = client.patch(f"{API_ROOT}/budgets/{budget_id}", json={"name": None}, headers=master_key_header).json()
    assert cleared["name"] is None
    assert cleared["max_budget"] == 25.0


def test_new_budget_reports_zero_rollup(client: TestClient, master_key_header: dict[str, str]) -> None:
    """A budget with no assigned users reports zeros, not nulls or an error."""
    budget_id = _make_budget(client, master_key_header)

    data = client.get(f"{API_ROOT}/budgets/{budget_id}", headers=master_key_header).json()
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
                f"{API_ROOT}/users",
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
    data = client.get(f"{API_ROOT}/budgets/{budget_id}", headers=master_key_header).json()
    assert data["user_count"] == 2
    assert data["total_spend"] == 14.0
    assert data["total_reserved"] == 2.0

    # Same numbers from the grouped list query.
    listed = client.get(f"{API_ROOT}/budgets", headers=master_key_header).json()
    row = next(b for b in listed if b["budget_id"] == budget_id)
    assert row["user_count"] == 2
    assert row["total_spend"] == 14.0
    assert row["total_reserved"] == 2.0


def test_budget_rollup_excludes_deleted_users(client: TestClient, master_key_header: dict[str, str]) -> None:
    """A soft-deleted user drops out of the budget's rollup."""
    budget_id = _make_budget(client, master_key_header)
    client.post(f"{API_ROOT}/users", json={"user_id": "gone", "budget_id": budget_id}, headers=master_key_header)

    assert client.get(f"{API_ROOT}/budgets/{budget_id}", headers=master_key_header).json()["user_count"] == 1

    assert client.delete(f"{API_ROOT}/users/gone", headers=master_key_header).status_code == 204
    assert client.get(f"{API_ROOT}/budgets/{budget_id}", headers=master_key_header).json()["user_count"] == 0


def test_reset_logs_returned_newest_first(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """The reset-logs endpoint surfaces BudgetResetLog rows, most recent first."""
    budget_id = _make_budget(client, master_key_header)
    client.post(f"{API_ROOT}/users", json={"user_id": "resetter", "budget_id": budget_id}, headers=master_key_header)

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

    logs = client.get(f"{API_ROOT}/budgets/{budget_id}/reset-logs", headers=master_key_header).json()
    assert [log["previous_spend"] for log in logs] == [7.0, 5.0]
    assert logs[0]["user_id"] == "resetter"
    assert logs[0]["budget_id"] == budget_id
    assert logs[0]["next_reset_at"] is not None


def test_reset_logs_empty_for_fresh_budget(client: TestClient, master_key_header: dict[str, str]) -> None:
    budget_id = _make_budget(client, master_key_header)
    assert client.get(f"{API_ROOT}/budgets/{budget_id}/reset-logs", headers=master_key_header).json() == []


def test_reset_logs_unknown_budget_404(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.get(f"{API_ROOT}/budgets/does-not-exist/reset-logs", headers=master_key_header)
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

    refused = client.delete(f"{API_ROOT}/budgets/{budget_id}", headers=master_key_header)
    assert refused.status_code == 409, refused.text
    assert "Research" in refused.json()["detail"]

    # Still there, and deletable once nothing hands it out.
    assert client.get(f"{API_ROOT}/budgets/{budget_id}", headers=master_key_header).status_code == 200
    db_session.execute(delete(WorkspaceBudgetDefault).where(WorkspaceBudgetDefault.budget_id == budget_id))
    db_session.commit()
    assert client.delete(f"{API_ROOT}/budgets/{budget_id}", headers=master_key_header).status_code == 204


def test_deleting_an_organization_budget_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """The operator may edit a tenant's budget but not delete it out from under them."""
    organization = Organization(name="Acme", slug="acme-owned-budget")
    db_session.add(organization)
    db_session.flush()
    budget = Budget(organization_id=organization.id, name="Tenant cap", max_budget=Decimal(50))
    db_session.add(budget)
    db_session.commit()
    budget_id = budget.budget_id

    refused = client.delete(f"{API_ROOT}/budgets/{budget_id}", headers=master_key_header)
    assert refused.status_code == 409, refused.text
    assert "organization" in refused.json()["detail"]
    assert client.get(f"{API_ROOT}/budgets/{budget_id}", headers=master_key_header).status_code == 200


def test_deleting_a_budget_that_has_reset_clears_its_history(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """A reset log names its budget NOT NULL, so it goes with the budget rather than failing the delete."""
    budget_id = _make_budget(client, master_key_header)
    client.post(
        f"{API_ROOT}/users", json={"user_id": "reset-then-delete", "budget_id": budget_id}, headers=master_key_header
    )
    db_session.add(BudgetResetLog(user_id="reset-then-delete", budget_id=budget_id, previous_spend=5.0))
    db_session.commit()

    assert client.delete(f"{API_ROOT}/budgets/{budget_id}", headers=master_key_header).status_code == 204
    db_session.expire_all()
    remaining = db_session.execute(select(BudgetResetLog).where(BudgetResetLog.budget_id == budget_id)).all()
    assert remaining == []


def test_deleting_a_budget_leaves_its_users_uncapped(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Assigned gateway users lose the cap, which is what the dashboard's confirmation promises."""
    budget_id = _make_budget(client, master_key_header)
    client.post(f"{API_ROOT}/users", json={"user_id": "capped", "budget_id": budget_id}, headers=master_key_header)

    assert client.delete(f"{API_ROOT}/budgets/{budget_id}", headers=master_key_header).status_code == 204
    user = client.get(f"{API_ROOT}/users/capped", headers=master_key_header).json()
    assert user["budget_id"] is None


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
        f"{API_ROOT}/budgets",
        json={"max_budget": 100.0, "reset_cycle": "daily"},
        headers=master_key_header,
    )
    assert created.status_code == 200, created.text
    budget_id = created.json()["budget_id"]
    assert client.post(f"{API_ROOT}/users", json={"user_id": "alice"}, headers=master_key_header).status_code == 200

    attached = client.patch(f"{API_ROOT}/users/alice", json={"budget_id": budget_id}, headers=master_key_header)
    assert attached.status_code == 200, attached.text
    assert attached.json()["budget_id"] == budget_id
    assert attached.json()["next_budget_reset_at"] is not None

    detached = client.patch(f"{API_ROOT}/users/alice", json={"budget_id": None}, headers=master_key_header)
    assert detached.status_code == 200, detached.text
    assert detached.json()["budget_id"] is None
    assert detached.json()["next_budget_reset_at"] is None
    assert detached.json()["budget_started_at"] is None

    # Omitting the field is still "leave it alone", which is the half that
    # already worked and must keep working.
    reattached = client.patch(f"{API_ROOT}/users/alice", json={"budget_id": budget_id}, headers=master_key_header)
    assert reattached.json()["budget_id"] == budget_id
    renamed = client.patch(f"{API_ROOT}/users/alice", json={"alias": "Alice"}, headers=master_key_header)
    assert renamed.json()["budget_id"] == budget_id


def test_a_calendar_aligned_budget_gives_a_user_a_boundary_reset(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A user on a calendar cycle gets a next reset.

    A null next reset never fires, so the user's spend would never refill and
    they would eventually be refused for good.
    """
    monthly = client.post(
        f"{API_ROOT}/budgets",
        json={"max_budget": 100.0, "reset_cycle": "monthly", "reset_month_day": 1},
        headers=master_key_header,
    )
    assert monthly.status_code == 200, monthly.text
    assert monthly.json()["reset_cycle"] == "monthly"

    assert client.post(f"{API_ROOT}/users", json={"user_id": "bruno"}, headers=master_key_header).status_code == 200
    assigned = client.patch(
        f"{API_ROOT}/users/bruno",
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

    refused = client.delete(f"{API_ROOT}/budgets/{budget_id}", headers=master_key_header)
    assert refused.status_code == 409, refused.text
    assert "1 spend ceiling" in refused.json()["detail"]

    db_session.execute(delete(ScopedBudget).where(ScopedBudget.budget_id == budget_id))
    db_session.commit()
    assert client.delete(f"{API_ROOT}/budgets/{budget_id}", headers=master_key_header).status_code == 204


def test_a_cadence_change_retimes_the_users_holding_the_budget(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A user's reset only fires once their stored date passes, so the date has to follow the budget.

    From "no reset" the user has no date at all and would never reset; between two
    cadences they would stay on the old one until its date came round.
    """
    budget_id = _make_budget(client, master_key_header)
    client.post(f"{API_ROOT}/users", json={"user_id": "held", "budget_id": budget_id}, headers=master_key_header)
    assert client.get(f"{API_ROOT}/users/held", headers=master_key_header).json()["next_budget_reset_at"] is None

    client.patch(
        f"{API_ROOT}/budgets/{budget_id}",
        json={"reset_cycle": "monthly", "reset_month_day": 15},
        headers=master_key_header,
    )
    monthly = client.get(f"{API_ROOT}/users/held", headers=master_key_header).json()
    assert datetime.fromisoformat(monthly["next_budget_reset_at"]).day == 15

    client.patch(f"{API_ROOT}/budgets/{budget_id}", json={"reset_month_day": 3}, headers=master_key_header)
    moved = client.get(f"{API_ROOT}/users/held", headers=master_key_header).json()
    assert datetime.fromisoformat(moved["next_budget_reset_at"]).day == 3

    # A rename is not a cadence change, so it does not restart the period.
    client.patch(f"{API_ROOT}/budgets/{budget_id}", json={"name": "renamed"}, headers=master_key_header)
    renamed = client.get(f"{API_ROOT}/users/held", headers=master_key_header).json()
    assert renamed["next_budget_reset_at"] == moved["next_budget_reset_at"]
    assert renamed["budget_started_at"] == moved["budget_started_at"]


def test_replacing_a_budget_retimes_the_users_holding_it(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A replace changes the cadence as an update does, so the users follow it the same way."""
    assert client.put(f"{API_ROOT}/budgets/held", json={}, headers=master_key_header).status_code == 201
    client.post(f"{API_ROOT}/users", json={"user_id": "held", "budget_id": "held"}, headers=master_key_header)

    replaced = client.put(
        f"{API_ROOT}/budgets/held",
        json={"reset_cycle": "monthly", "reset_month_day": 15},
        headers=master_key_header,
    )
    assert replaced.status_code == 200
    held = client.get(f"{API_ROOT}/users/held", headers=master_key_header).json()
    assert datetime.fromisoformat(held["next_budget_reset_at"]).day == 15


def test_a_cadence_change_retimes_the_ceilings_naming_the_budget(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """The direction that is an enforcement bug, not a cosmetic one.

    `_roll_expired_periods` only updates a row whose `period_end` is not null, so
    a ceiling left with a NULL window under a periodic budget never rolls at all:
    it accumulates spend forever while the API reports the new cadence. Since
    `b7e1c4a9d2f5` a budget can also belong to an organization while this route
    still sees every one, so the ceilings stranded this way may be a tenant's.
    """
    budget_id = _make_budget(client, master_key_header)
    # A budget with no cadence materializes a ceiling with no window.
    ceiling = ScopedBudget(scope_type="workspace", scope_id=str(uuid4()), budget_id=budget_id)
    db_session.add(ceiling)
    db_session.commit()
    assert ceiling.period_end is None

    patched = client.patch(
        f"{API_ROOT}/budgets/{budget_id}",
        json={"reset_cycle": "monthly", "reset_month_day": 1},
        headers=master_key_header,
    )
    assert patched.status_code == 200, patched.json()

    db_session.expire_all()
    retimed = db_session.get(ScopedBudget, ceiling.id)
    assert retimed is not None
    assert retimed.period_start is not None
    assert retimed.period_end is not None


def test_dropping_a_cadence_clears_the_ceiling_window(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """The reverse, which would otherwise roll once at a boundary that no longer means anything."""
    created = client.post(
        f"{API_ROOT}/budgets", json={"max_budget": 100.0, "reset_cycle": "daily"}, headers=master_key_header
    ).json()
    budget_id = created["budget_id"]
    ceiling = ScopedBudget(
        scope_type="workspace",
        scope_id=str(uuid4()),
        budget_id=budget_id,
        period_start=datetime(2026, 8, 1, tzinfo=UTC),
        period_end=datetime(2026, 8, 2, tzinfo=UTC),
    )
    db_session.add(ceiling)
    db_session.commit()

    patched = client.patch(
        f"{API_ROOT}/budgets/{budget_id}",
        json={"reset_cycle": None},
        headers=master_key_header,
    )
    assert patched.status_code == 200, patched.json()

    db_session.expire_all()
    cleared = db_session.get(ScopedBudget, ceiling.id)
    assert cleared is not None
    assert cleared.period_start is None
    assert cleared.period_end is None


def test_a_rename_does_not_restart_a_ceiling_period(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """Retiming is keyed on the cadence, so unrelated edits leave the window alone.

    Otherwise every typo fix in a budget's name would throw away the part of the
    period its ceilings had already spent.
    """
    created = client.post(
        f"{API_ROOT}/budgets", json={"max_budget": 100.0, "reset_cycle": "daily"}, headers=master_key_header
    ).json()
    budget_id = created["budget_id"]
    start, end = datetime(2026, 8, 1, tzinfo=UTC), datetime(2026, 8, 2, tzinfo=UTC)
    ceiling = ScopedBudget(
        scope_type="workspace",
        scope_id=str(uuid4()),
        budget_id=budget_id,
        period_start=start,
        period_end=end,
    )
    db_session.add(ceiling)
    db_session.commit()

    patched = client.patch(
        f"{API_ROOT}/budgets/{budget_id}",
        json={"name": "renamed", "max_budget": 500.0},
        headers=master_key_header,
    )
    assert patched.status_code == 200, patched.json()

    db_session.expire_all()
    untouched = db_session.get(ScopedBudget, ceiling.id)
    assert untouched is not None
    assert untouched.period_start == start
    assert untouched.period_end == end


# The counters roll when the next request arrives, not when the period ends, so
# an idle row still holds the period that is over. Every read reports it as zero.


def test_a_ceiling_whose_period_ended_reads_as_nothing_spent_yet(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    budget_id = _make_budget(client, master_key_header)
    ceiling = ScopedBudget(
        scope_type="workspace",
        scope_id=str(uuid4()),
        budget_id=budget_id,
        current_spend=Decimal("9.5"),
        reserved_spend=Decimal("1.25"),
        current_requests=4,
        period_end=datetime(2026, 1, 1, tzinfo=UTC),
    )
    db_session.add(ceiling)
    db_session.commit()

    read = client.get(f"{API_ROOT}/scoped-budgets/{ceiling.id}", headers=master_key_header).json()
    assert read["current_spend"] == 0.0
    assert read["current_requests"] == 0
    # A hold is still held: the roll that will come leaves it too.
    assert read["reserved_spend"] == 1.25


def test_a_user_whose_period_ended_reads_as_nothing_spent_yet(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    budget_id = _make_budget(client, master_key_header)
    client.post(f"{API_ROOT}/users", json={"user_id": "idle", "budget_id": budget_id}, headers=master_key_header)
    user = db_session.get(User, "idle")
    assert user is not None
    user.spend = Decimal("7")
    user.next_budget_reset_at = datetime(2026, 1, 1, tzinfo=UTC)
    db_session.commit()

    assert client.get(f"{API_ROOT}/users/idle", headers=master_key_header).json()["spend"] == 0.0
    assert client.get(f"{API_ROOT}/budgets/{budget_id}", headers=master_key_header).json()["total_spend"] == 0.0
    listed = client.get(f"{API_ROOT}/budgets", headers=master_key_header).json()
    assert next(row for row in listed if row["budget_id"] == budget_id)["total_spend"] == 0.0
    # Same cadence, so no retime rolls the counter first: the replace reads it as stored.
    replaced = client.put(f"{API_ROOT}/budgets/{budget_id}", json={"max_budget": 100.0}, headers=master_key_header)
    assert replaced.json()["total_spend"] == 0.0


def test_a_cadence_change_rolls_a_period_that_had_ended_rather_than_carrying_it(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """Moving the window forward without rolling would put last period's spend in the new one for good."""
    budget_id = _make_budget(client, master_key_header)
    ended = ScopedBudget(
        scope_type="workspace",
        scope_id=str(uuid4()),
        budget_id=budget_id,
        current_spend=Decimal("9.5"),
        period_end=datetime(2026, 1, 1, tzinfo=UTC),
    )
    db_session.add(ended)
    db_session.commit()

    client.patch(
        f"{API_ROOT}/budgets/{budget_id}",
        json={"reset_cycle": "monthly", "reset_month_day": 1},
        headers=master_key_header,
    )

    db_session.expire_all()
    rolled = db_session.get(ScopedBudget, ended.id)
    assert rolled is not None
    assert rolled.current_spend == 0


def test_deleting_a_key_deletes_the_budgets_applied_to_it(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """Nothing cascades to a ceiling, so a deleted key's budget would otherwise linger as "An API key"."""
    created = client.post(f"{API_ROOT}/keys", json={"key_name": "doomed"}, headers=master_key_header)
    assert created.status_code == 200, created.text
    key_id = created.json()["id"]
    ceiling = ScopedBudget(scope_type="api_token", scope_id=key_id, budget_id=_make_budget(client, master_key_header))
    db_session.add(ceiling)
    db_session.commit()
    ceiling_id = ceiling.id

    assert client.delete(f"{API_ROOT}/keys/{key_id}", headers=master_key_header).status_code == 204

    db_session.expire_all()
    assert db_session.get(ScopedBudget, ceiling_id) is None
