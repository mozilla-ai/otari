"""``GET /api/v1/organizations/me/subagent-savings``: what the subagent router saved, by scope.

Written against the two-organization world of the usage scope suite, with
decisions in every workspace. These pin the arithmetic (the router's cost
comes off for every decision, a loss stays a loss, an unpriced outcome stays
out of the money) and that the caller's membership decides which decisions
are read.
"""

import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT
from gateway.models.routing import SubagentDecision
from gateway.models.usage import UsageLog
from gateway.services.secret_box import generate_secret_key

from .test_organization_usage_scope import _as, _World, world  # noqa: F401

SAVINGS = f"{API_ROOT}/organizations/me/subagent-savings"
OPUS = "anthropic:claude-opus-5"
HAIKU = "claude-haiku-5"


def _decision(workspace_id: uuid.UUID, agent_type: str, router_cost: str, **fields: Any) -> SubagentDecision:
    return SubagentDecision(
        id=uuid.uuid4(),
        workspace_id=workspace_id,
        harness="claude-code",
        agent_type=agent_type,
        parent_model="claude-opus-5",
        baseline_model="claude-opus-5",
        baseline_source="parent",
        recommended_model="haiku",
        router_cost=Decimal(router_cost),
        **fields,
    )


def _reported(kind: str, baseline: str | None, actual: str | None, **fields: Any) -> dict[str, Any]:
    """The outcome columns of a reported subagent, priced on both sides or on neither."""
    return {
        "outcome_kind": kind,
        "outcome_at": datetime.now(UTC),
        "ran_on": HAIKU,
        "baseline_model_key": OPUS,
        "baseline_cost": Decimal(baseline) if baseline is not None else None,
        "actual_cost": Decimal(actual) if actual is not None else None,
        **fields,
    }


@pytest.fixture
def decided(world: _World, db_session_factory: Callable[[], Session]) -> Iterator[_World]:  # noqa: F811
    """Decisions in all three workspaces. Alpha one holds every kind of outcome."""
    one, two, beta = (world.workspaces[name] for name in ("alpha_one", "alpha_two", "beta_one"))
    session = db_session_factory()
    try:
        session.add_all(
            [
                _decision(one, "Explore", "0.01", **_reported("estimated", "1.00", "0.20")),
                # The cheaper model cost more: a loss.
                _decision(one, "Plan", "0.01", **_reported("estimated", "0.10", "0.30")),
                _decision(one, "Explore", "0.02"),
                _decision(
                    one,
                    "Explore",
                    "0.01",
                    **_reported(
                        "estimated", None, None, baseline_model_key=None, unpriced_reason="baseline_unresolved"
                    ),
                ),
                _decision(one, "Explore", "0", **_reported("measured", "2.00", "1.00")),
                # Before the default window.
                _decision(
                    one,
                    "Explore",
                    "9",
                    created_at=datetime.now(UTC) - timedelta(days=60),
                    **_reported("estimated", "90", "1"),
                ),
                _decision(two, "general-purpose", "0.05", **_reported("estimated", "5.00", "1.00")),
                _decision(beta, "Explore", "7", **_reported("estimated", "100", "1")),
            ]
        )
        session.add(
            UsageLog(
                id=str(uuid.uuid4()),
                workspace_id=one,
                model="subagent-router",
                provider="otari",
                endpoint="/v1/decisions",
                source="gateway",
                status="error",
                status_code=502,
                timestamp=datetime.now(UTC),
            )
        )
        session.commit()
        yield world
    finally:
        session.close()


def _savings(client: TestClient, decided: _World, who: str, query: str = "") -> dict[str, Any]:
    code, body = _as(client, decided, who, f"{SAVINGS}{query}")
    assert code == 200, body
    assert isinstance(body, dict)
    return body


def test_an_owner_reads_what_the_whole_organization_saved(client: TestClient, decided: _World) -> None:
    totals = _savings(client, decided, "alpha_owner")["totals"]

    counts = ("decisions", "with_outcome", "measured", "estimated", "no_outcome", "unpriced", "router_unanswered")
    assert [totals[name] for name in counts] == [6, 5, 1, 3, 1, 1, 1]
    # Four priced outcomes: 1.00 + 0.10 + 2.00 + 5.00 against 0.20 + 0.30 + 1.00 + 1.00.
    assert (totals["baseline_cost"], totals["actual_cost"]) == (pytest.approx(8.1), pytest.approx(2.5))
    # All six decisions' router cost, the unreported and the unpriced one too.
    assert totals["router_cost"] == pytest.approx(0.10)
    assert (totals["gross_saved"], totals["net_saved"]) == (pytest.approx(5.6), pytest.approx(5.5))


def test_measured_and_estimated_are_told_apart(client: TestClient, decided: _World) -> None:
    body = _savings(client, decided, "alpha_owner")

    measured, estimated = body["measured"], body["estimated"]
    assert (measured["decisions"], measured["measured"], measured["estimated"]) == (1, 1, 0)
    assert (measured["baseline_cost"], measured["actual_cost"]) == (pytest.approx(2.0), pytest.approx(1.0))
    # The unpriced outcome is the harness's count too, and adds no money.
    assert (estimated["decisions"], estimated["measured"], estimated["estimated"]) == (4, 0, 3)
    assert (estimated["baseline_cost"], estimated["actual_cost"]) == (pytest.approx(6.1), pytest.approx(1.5))


def test_a_member_reads_only_the_workspaces_they_belong_to(client: TestClient, decided: _World) -> None:
    totals = _savings(client, decided, "alpha_member")["totals"]

    assert totals["decisions"] == 5
    assert (totals["baseline_cost"], totals["router_cost"]) == (pytest.approx(3.1), pytest.approx(0.05))
    assert totals["net_saved"] == pytest.approx(3.1 - 1.5 - 0.05)


def test_someone_in_no_workspace_reads_nothing(client: TestClient, decided: _World) -> None:
    body = _savings(client, decided, "alpha_newcomer")

    assert (body["totals"]["decisions"], body["totals"]["net_saved"]) == (0, 0)
    assert (body["series"], body["by_agent_type"], body["by_swap"]) == ([], [], [])


def test_a_workspace_narrows_the_scope_and_never_widens_it(client: TestClient, decided: _World) -> None:
    two = decided.workspaces["alpha_two"]

    assert _savings(client, decided, "alpha_owner", f"?workspace_id={two}")["totals"]["decisions"] == 1
    assert _as(client, decided, "alpha_member", f"{SAVINGS}?workspace_id={two}")[0] == 404
    assert _as(client, decided, "alpha_owner", f"{SAVINGS}?workspace_id={decided.workspaces['beta_one']}")[0] == 404


def test_subagent_types_are_ranked_by_what_they_saved_and_a_loss_is_shown(client: TestClient, decided: _World) -> None:
    rows = _savings(client, decided, "alpha_owner")["by_agent_type"]

    assert [row["agent_type"] for row in rows] == ["general-purpose", "Explore", "Plan"]
    explore, plan = rows[1], rows[2]
    assert (explore["decisions"], explore["with_outcome"], explore["estimated"], explore["measured"]) == (4, 3, 1, 1)
    assert (plan["gross_saved"], plan["net_saved"]) == (pytest.approx(-0.2), pytest.approx(-0.21))


def test_a_type_whose_router_cost_outweighs_its_saving_ranks_below_a_smaller_loss(
    client: TestClient, decided: _World, db_session_factory: Callable[[], Session]
) -> None:
    session = db_session_factory()
    try:
        # Saved 1.00 before the router's 2.00: a net loss of 1.00, worse than Plan's 0.21.
        session.add(
            _decision(decided.workspaces["alpha_one"], "Costly", "2.00", **_reported("estimated", "3.00", "2.00"))
        )
        session.commit()
    finally:
        session.close()

    rows = _savings(client, decided, "alpha_owner")["by_agent_type"]

    assert [row["agent_type"] for row in rows] == ["general-purpose", "Explore", "Plan", "Costly"]


def test_swaps_cover_the_reported_outcomes_by_baseline_and_model_run_on(client: TestClient, decided: _World) -> None:
    rows = _savings(client, decided, "alpha_owner")["by_swap"]

    assert [(row["baseline_model"], row["ran_on"], row["decisions"]) for row in rows] == [
        (OPUS, HAIKU, 4),
        # Read as no priced model, so named as the harness named it, with no money.
        ("claude-opus-5", HAIKU, 1),
    ]
    assert (rows[1]["baseline_cost"], rows[1]["estimated"]) == (0, 0)


def test_the_series_covers_the_window_and_adds_up_to_the_totals(client: TestClient, decided: _World) -> None:
    body = _savings(client, decided, "alpha_owner")

    series = body["series"]
    assert len(series) in (30, 31)
    assert sum(point["decisions"] for point in series) == 6
    assert sum(point["net_saved"] for point in series) == pytest.approx(body["totals"]["net_saved"])
    assert series[-1]["decisions"] == 6


def test_a_wider_window_reaches_older_decisions(client: TestClient, decided: _World) -> None:
    start = (datetime.now(UTC) - timedelta(days=90)).strftime("%Y-%m-%dT%H:%M:%SZ")

    totals = _savings(client, decided, "alpha_owner", f"?start_date={start}")["totals"]

    assert (totals["decisions"], totals["router_cost"]) == (7, pytest.approx(9.10))


def test_a_user_narrows_the_decisions(
    client: TestClient,
    master_key_header: dict[str, str],
    decided: _World,
    db_session_factory: Callable[[], Session],
) -> None:
    assert client.post(f"{API_ROOT}/users", json={"user_id": "alice"}, headers=master_key_header).is_success
    session = db_session_factory()
    try:
        session.add(
            _decision(
                decided.workspaces["alpha_one"], "Explore", "0.5", user_id="alice", **_reported("estimated", "3", "1")
            )
        )
        session.commit()
    finally:
        session.close()

    totals = _savings(client, decided, "alpha_owner", "?user_id=alice")["totals"]

    assert (totals["decisions"], totals["net_saved"]) == (1, pytest.approx(1.5))
    # The failed recommendation was nobody's named here.
    assert totals["router_unanswered"] == 0


def test_whether_anything_answers_a_recommendation_is_reported(
    client: TestClient, master_key_header: dict[str, str], decided: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _savings(client, decided, "alpha_owner")["recommender_available"] is False

    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    stored = client.post(
        f"{API_ROOT}/routing/recommenders",
        json={"provider": "openrouter", "model": "stored-decider", "api_key": "sk-or-1234"},
        headers=master_key_header,
    )
    assert stored.status_code == 201, stored.text

    assert _savings(client, decided, "alpha_owner")["recommender_available"] is True


def test_a_reader_needs_a_session(client: TestClient) -> None:
    assert client.get(SAVINGS).status_code == 401
