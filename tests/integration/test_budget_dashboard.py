"""Dashboard-facing budget endpoints: per-budget usage rollup and reset history."""

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from gateway.models.entities import Budget, BudgetResetLog, ScopedBudget, User, WorkspaceBudgetDefault
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


def test_budget_rollup_excludes_deleted_users(client: TestClient, master_key_header: dict[str, str]) -> None:
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


def test_deleting_an_organization_owned_budget_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """The deployment surface may not delete a tenant's organization-owned budget (otari#902, #898).

    ``Budget.organization_id`` is the discriminator: ``None`` on the deployment's
    own budgets, a uuid on a tenant's. The operator's Budgets page labels such a
    row "Owned by an organization" and still offers Delete; the route refuses it
    with a 409 so it is managed through the organization instead. ``PATCH`` still
    reaches the same budget to retime its ceilings, so the refusal is delete-only.
    """
    budget_id = _make_budget(client, master_key_header)
    organization = Organization(name="Acme", slug="acme-owned-budget")
    db_session.add(organization)
    db_session.flush()
    budget = db_session.execute(select(Budget).where(Budget.budget_id == budget_id)).scalar_one()
    budget.organization_id = organization.id
    db_session.commit()

    refused = client.delete(f"/v1/budgets/{budget_id}", headers=master_key_header)
    assert refused.status_code == 409, refused.text
    assert "organization" in refused.json()["detail"].lower()

    # Untouched, and an edit still goes through.
    assert client.get(f"/v1/budgets/{budget_id}", headers=master_key_header).status_code == 200
    edited = client.patch(f"/v1/budgets/{budget_id}", json={"max_budget": 50.0}, headers=master_key_header)
    assert edited.status_code == 200, edited.text


def test_deleting_an_own_budget_with_reset_history_succeeds_without_500(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """A deployment-own budget that has ever reset deletes cleanly (otari#902).

    ``budget_reset_logs.budget_id`` is NOT NULL behind a plain relationship, so
    the ORM's null-out would fail at the commit as an opaque 500. The route clears
    those rows in the same transaction first, so the delete returns 204 and the
    reset logs are gone with the budget.
    """
    budget_id = _make_budget(client, master_key_header)
    db_session.add(
        BudgetResetLog(
            user_id=None,
            budget_id=budget_id,
            previous_spend=Decimal("1.00"),
            reset_at=datetime.now(UTC),
        )
    )
    db_session.commit()

    deleted = client.delete(f"/v1/budgets/{budget_id}", headers=master_key_header)
    assert deleted.status_code == 204, deleted.text

    # Gone, and its reset history went with it rather than 500-ing.
    assert client.get(f"/v1/budgets/{budget_id}", headers=master_key_header).status_code == 404
    remaining = db_session.execute(
        select(BudgetResetLog).where(BudgetResetLog.budget_id == budget_id)
    ).scalars().all()
    assert remaining == []


def test_deleting_an_own_budget_with_an_assigned_user_succeeds(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A deployment-own budget with a user assigned to it deletes with 204 (otari#902).

    Assigning a user at creation is the path that makes a deployment-own budget
    enforceable, and it is the flow the e2e parity test exercises. Deleting the
    budget nulls that cap by design (a deleted budget caps no one); the route does
    not refuse on assigned users the way the organization-scoped delete does.
    """
    budget_id = _make_budget(client, master_key_header)
    assert (
        client.post(
            "/v1/users", json={"user_id": "capped", "budget_id": budget_id}, headers=master_key_header
        ).status_code
        == 200
    )

    deleted = client.delete(f"/v1/budgets/{budget_id}", headers=master_key_header)
    assert deleted.status_code == 204, deleted.text

    # The budget is gone and the user is uncapped, not orphaned on a missing budget.
    assert client.get(f"/v1/budgets/{budget_id}", headers=master_key_header).status_code == 404
    assert client.get("/v1/users/capped", headers=master_key_header).json()["budget_id"] is None


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
        f"/v1/budgets/{budget_id}",
        json={"reset_alignment": "calendar_month"},
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
        "/v1/budgets", json={"max_budget": 100.0, "reset_alignment": "calendar_day"}, headers=master_key_header
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
        f"/v1/budgets/{budget_id}",
        json={"reset_alignment": None},
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
        "/v1/budgets", json={"max_budget": 100.0, "reset_alignment": "calendar_day"}, headers=master_key_header
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
        f"/v1/budgets/{budget_id}",
        json={"name": "renamed", "max_budget": 500.0},
        headers=master_key_header,
    )
    assert patched.status_code == 200, patched.json()

    db_session.expire_all()
    untouched = db_session.get(ScopedBudget, ceiling.id)
    assert untouched is not None
    assert untouched.period_start == start
    assert untouched.period_end == end
