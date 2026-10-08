import os
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT
from gateway.services.budgets import cycle_window

from .conftest import MODEL_NAME

_HAS_GEMINI_KEY = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def test_an_interval_period_counts_whole_steps_from_its_anchor() -> None:
    """The interval half of the one derivation both planes share.

    ``calculate_next_reset`` used to own this and read only a duration, which is
    why a calendar-aligned budget never reset: every caller of it was blind to
    the other cadence. The window is now counted from the budget's anchor rather
    than from whenever it was last asked, which is what stops a quiet workspace
    walking its reset forward through the day.
    """
    anchor = datetime(2025, 10, 1, 0, 0, 0, tzinfo=UTC)

    assert cycle_window(
        anchor,
        cycle="every_n_days",
        every_n=1,
        anchor_at=anchor,
        weekdays=None,
        month_day=None,
        month=None,
    ) == (anchor, datetime(2025, 10, 2, 0, 0, 0, tzinfo=UTC))

    # A week later, and asked about a moment mid-period: the window is still the
    # one the anchor puts it in, not one starting at the question.
    assert cycle_window(
        datetime(2025, 10, 9, 13, 0, 0, tzinfo=UTC),
        cycle="every_n_days",
        every_n=7,
        anchor_at=anchor,
        weekdays=None,
        month_day=None,
        month=None,
    ) == (
        datetime(2025, 10, 8, 0, 0, 0, tzinfo=UTC),
        datetime(2025, 10, 15, 0, 0, 0, tzinfo=UTC),
    )


def test_create_budget_with_a_reset_cycle(client: TestClient, master_key_header: dict[str, str]) -> None:
    """A budget is created with the reset cycle it names."""
    response = client.post(
        f"{API_ROOT}/budgets",
        json={"max_budget": 100.0, "reset_cycle": "daily"},
        headers=master_key_header,
    )
    assert response.status_code == 200, f"Response: {response.json()}"
    data = response.json()
    assert data["max_budget"] == 100.0
    assert data["reset_cycle"] == "daily"


def test_user_with_budget_gets_reset_fields_set(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that creating a user with a budget sets budget tracking fields."""
    budget_response = client.post(
        f"{API_ROOT}/budgets",
        json={"max_budget": 50.0, "reset_cycle": "weekly", "reset_weekdays": 1},
        headers=master_key_header,
    )
    budget_id = budget_response.json()["budget_id"]

    response = client.post(
        f"{API_ROOT}/users",
        json={"user_id": "test-user-1", "budget_id": budget_id},
        headers=master_key_header,
    )

    assert response.status_code == 200, f"Response: {response.json()}"
    data = response.json()
    assert data["budget_id"] == budget_id
    assert data["budget_started_at"] is not None
    assert data["next_budget_reset_at"] is not None


def test_updating_user_budget_sets_reset_fields(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that updating a user's budget sets budget tracking fields."""
    budget_response = client.post(
        f"{API_ROOT}/budgets",
        json={"max_budget": 75.0, "reset_cycle": "daily"},
        headers=master_key_header,
    )
    budget_id = budget_response.json()["budget_id"]

    client.post(
        f"{API_ROOT}/users",
        json={"user_id": "test-user-1"},
        headers=master_key_header,
    )

    response = client.patch(
        f"{API_ROOT}/users/test-user-1",
        json={"budget_id": budget_id},
        headers=master_key_header,
    )

    assert response.status_code == 200, f"Response: {response.json()}"
    data = response.json()
    assert data["budget_started_at"] is not None
    assert data["next_budget_reset_at"] is not None


def test_budget_without_duration_no_reset(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that budgets without duration don't set reset schedules."""
    budget_response = client.post(
        f"{API_ROOT}/budgets",
        json={"max_budget": 100.0},
        headers=master_key_header,
    )
    budget_id = budget_response.json()["budget_id"]

    response = client.post(
        f"{API_ROOT}/users",
        json={"user_id": "test-user-1", "budget_id": budget_id},
        headers=master_key_header,
    )

    assert response.status_code == 200, f"Response: {response.json()}"
    data = response.json()
    assert data["budget_started_at"] is not None
    assert data["next_budget_reset_at"] is None


def test_nonexistent_budget_returns_404(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that assigning a nonexistent budget returns 404."""
    response = client.post(
        f"{API_ROOT}/users",
        json={"user_id": "test-user-1", "budget_id": "nonexistent-budget"},
        headers=master_key_header,
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.skipif(not _HAS_GEMINI_KEY, reason="requires GEMINI_API_KEY or GOOGLE_API_KEY")
def test_budget_actually_resets_when_duration_passes(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    test_messages: list[dict[str, str]],
) -> None:
    """Test that budget actually resets when duration passes - THE CRITICAL TEST."""
    budget_response = client.post(
        f"{API_ROOT}/budgets",
        json={
            "max_budget": 100.0,
            "reset_cycle": "every_n_hours",
            "reset_every_n": 1,
            "reset_anchor_at": "2026-01-01T00:00:00Z",
        },
        headers=master_key_header,
    )
    budget_id = budget_response.json()["budget_id"]

    client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": MODEL_NAME,
            "input_price_per_million": 2.5,
            "output_price_per_million": 10.0,
        },
        headers=master_key_header,
    )

    initial_time = datetime(2025, 10, 1, 12, 0, 0, tzinfo=UTC)

    with patch("gateway.api.routes.users.datetime") as mock_datetime:
        mock_datetime.now.return_value = initial_time

        client.post(
            f"{API_ROOT}/users",
            json={"user_id": "test-user-1", "budget_id": budget_id},
            headers=master_key_header,
        )

    user_response = client.get(f"{API_ROOT}/users/test-user-1", headers=master_key_header)
    assert user_response.status_code == 200
    user_data = user_response.json()
    assert user_data["spend"] == 0.0

    with patch("gateway.services.budgets._reservations.datetime") as mock_datetime_budget:
        mock_datetime_budget.now.return_value = initial_time

        with patch("gateway.api.routes._pipeline.datetime") as mock_datetime_chat:
            mock_datetime_chat.now.return_value = initial_time

            response = client.post(
                f"{API_ROOT}/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": test_messages,
                    "user": "test-user-1",
                },
                headers=api_key_header,
            )

    assert response.status_code == 200, f"Response: {response.json()}"

    user_response = client.get(f"{API_ROOT}/users/test-user-1", headers=master_key_header)
    user_data = user_response.json()
    spend_before_reset = user_data["spend"]
    assert spend_before_reset > 0.0

    time_after_reset = initial_time + timedelta(seconds=61)

    with patch("gateway.services.budgets._reservations.datetime") as mock_datetime_budget:
        mock_datetime_budget.now.return_value = time_after_reset

        with patch("gateway.api.routes._pipeline.datetime") as mock_datetime_chat:
            mock_datetime_chat.now.return_value = time_after_reset

            response = client.post(
                f"{API_ROOT}/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": test_messages,
                    "user": "test-user-1",
                },
                headers=api_key_header,
            )

    assert response.status_code == 200, f"Response: {response.json()}"

    user_response = client.get(f"{API_ROOT}/users/test-user-1", headers=master_key_header)
    user_data = user_response.json()
    spend_after_reset = user_data["spend"]

    assert spend_after_reset > 0.0
    assert spend_after_reset < (spend_before_reset * 2)


def test_users_on_one_budget_share_its_cycle(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Users on one budget land on that budget's boundary, whenever they joined.

    This used to assert the opposite, and the opposite was the bug: a duration
    was counted from each user's own start, so two people on one weekly budget
    reset on different days and neither on the day the budget named.
    """
    budget_response = client.post(
        f"{API_ROOT}/budgets",
        json={"max_budget": 100.0, "reset_cycle": "weekly", "reset_weekdays": 1},
        headers=master_key_header,
    )
    budget_id = budget_response.json()["budget_id"]

    user_a_time = datetime(2025, 10, 1, 0, 0, 0, tzinfo=UTC)
    user_b_time = datetime(2025, 10, 2, 0, 0, 0, tzinfo=UTC)

    with patch("gateway.api.routes.users.datetime") as mock_datetime:
        mock_datetime.now.return_value = user_a_time

        response_a = client.post(
            f"{API_ROOT}/users",
            json={"user_id": "user-a", "budget_id": budget_id},
            headers=master_key_header,
        )

    with patch("gateway.api.routes.users.datetime") as mock_datetime:
        mock_datetime.now.return_value = user_b_time

        response_b = client.post(
            f"{API_ROOT}/users",
            json={"user_id": "user-b", "budget_id": budget_id},
            headers=master_key_header,
        )

    assert response_a.status_code == 200
    assert response_b.status_code == 200

    user_a_data = response_a.json()
    user_b_data = response_b.json()

    reset_a = datetime.fromisoformat(user_a_data["next_budget_reset_at"]).replace(tzinfo=UTC)
    reset_b = datetime.fromisoformat(user_b_data["next_budget_reset_at"]).replace(tzinfo=UTC)

    # The Monday after each of them, which for these two is the same Monday.
    assert reset_a == datetime(2025, 10, 6, 0, 0, 0, tzinfo=UTC)
    assert reset_b == reset_a
