from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.models.users import User
from gateway.services.budgets._periods import seconds_until


def test_create_user(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test creating a new user."""
    response = client.post(
        f"{API_ROOT}/users",
        json={"user_id": "test-user-1", "alias": "Test User"},
        headers=master_key_header,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "test-user-1"
    assert data["alias"] == "Test User"
    assert data["spend"] == 0.0
    assert data["blocked"] is False


def test_list_users(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test listing users."""
    client.post(
        f"{API_ROOT}/users",
        json={"user_id": "test-user-1"},
        headers=master_key_header,
    )
    client.post(
        f"{API_ROOT}/users",
        json={"user_id": "test-user-2"},
        headers=master_key_header,
    )

    response = client.get(f"{API_ROOT}/users", headers=master_key_header)
    assert response.status_code == 200
    data = response.json()
    user_ids = {item["user_id"] for item in data}
    assert {"test-user-1", "test-user-2"}.issubset(user_ids)


def test_get_user(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test getting a specific user."""
    client.post(
        f"{API_ROOT}/users",
        json={"user_id": "test-user-1"},
        headers=master_key_header,
    )

    response = client.get(f"{API_ROOT}/users/test-user-1", headers=master_key_header)
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "test-user-1"


def test_update_user(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test updating a user."""
    client.post(
        f"{API_ROOT}/users",
        json={"user_id": "test-user-1"},
        headers=master_key_header,
    )

    response = client.patch(
        f"{API_ROOT}/users/test-user-1",
        json={"blocked": True, "alias": "Updated User"},
        headers=master_key_header,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["blocked"] is True
    assert data["alias"] == "Updated User"


def test_delete_user(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test deleting a user."""
    client.post(
        f"{API_ROOT}/users",
        json={"user_id": "test-user-1"},
        headers=master_key_header,
    )

    response = client.delete(f"{API_ROOT}/users/test-user-1", headers=master_key_header)
    assert response.status_code == 204

    response = client.get(f"{API_ROOT}/users/test-user-1", headers=master_key_header)
    assert response.status_code == 404


def test_create_budget(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test creating a budget."""
    response = client.post(
        f"{API_ROOT}/budgets",
        json={"max_budget": 100.0},
        headers=master_key_header,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["max_budget"] == 100.0


def test_list_budgets(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test listing budgets."""
    client.post(
        f"{API_ROOT}/budgets",
        json={"max_budget": 100.0},
        headers=master_key_header,
    )
    client.post(
        f"{API_ROOT}/budgets",
        json={"max_budget": 200.0},
        headers=master_key_header,
    )

    response = client.get(f"{API_ROOT}/budgets", headers=master_key_header)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


def test_set_model_pricing(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test setting model pricing."""
    response = client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": "openai:gpt-4o",
            "input_price_per_million": 2.5,
            "output_price_per_million": 10.0,
        },
        headers=master_key_header,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_key"] == "openai:gpt-4o"
    assert data["input_price_per_million"] == 2.5
    assert data["output_price_per_million"] == 10.0


def test_get_pricing(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test getting model pricing."""
    client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": "openai:gpt-4o",
            "input_price_per_million": 2.5,
            "output_price_per_million": 10.0,
        },
        headers=master_key_header,
    )

    response = client.get(f"{API_ROOT}/pricing/openai:gpt-4o", headers=master_key_header)
    assert response.status_code == 200
    data = response.json()
    assert data["model_key"] == "openai:gpt-4o"


def test_get_pricing_requires_auth(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Pricing reads require API key or master key authentication."""
    client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": "openai:gpt-4o",
            "input_price_per_million": 2.5,
            "output_price_per_million": 10.0,
        },
        headers=master_key_header,
    )

    response = client.get(f"{API_ROOT}/pricing/openai:gpt-4o")
    assert response.status_code == 401


def test_user_with_budget(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test creating a user with a budget."""
    budget_response = client.post(
        f"{API_ROOT}/budgets",
        json={"max_budget": 50.0},
        headers=master_key_header,
    )
    budget_id = budget_response.json()["budget_id"]

    response = client.post(
        f"{API_ROOT}/users",
        json={"user_id": "test-user-1", "budget_id": budget_id},
        headers=master_key_header,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["budget_id"] == budget_id


def test_blocked_user_cannot_make_requests(
    client: TestClient,
    master_key_header: dict[str, str],
    test_messages: list[dict[str, str]],
) -> None:
    """Test that blocked users cannot make completion requests."""
    client.post(
        f"{API_ROOT}/users",
        json={"user_id": "test-user-1", "blocked": True},
        headers=master_key_header,
    )

    # Bill the named user via the master key — a non-master key may only spend
    # against its own user (see test_api_key_cannot_bill_other_user).
    response = client.post(
        f"{API_ROOT}/chat/completions",
        json={
            "model": "openai:gpt-4o",
            "messages": test_messages,
            "user": "test-user-1",
        },
        headers=master_key_header,
    )
    assert response.status_code == 403
    assert "blocked" in response.json()["detail"].lower()


def test_user_not_found_with_master_key(
    client: TestClient,
    master_key_header: dict[str, str],
    test_messages: list[dict[str, str]],
) -> None:
    """Test that master key requests fail when user doesn't exist."""
    response = client.post(
        f"{API_ROOT}/chat/completions",
        json={
            "model": "openai:gpt-4o",
            "messages": test_messages,
            "user": "nonexistent-user",
        },
        headers=master_key_header,
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_api_key_cannot_bill_other_user(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    test_messages: list[dict[str, str]],
) -> None:
    """A non-master key naming a user other than its own is rejected (IDOR fix).

    Previously a user key could set ``user`` to any value and charge spend to
    that user; the gateway now binds non-master spend to the key's own user.
    """
    # Create the would-be victim so this proves the rejection is an ownership
    # check, not merely a "user not found" path.
    client.post(
        f"{API_ROOT}/users",
        json={"user_id": "victim-user"},
        headers=master_key_header,
    )

    response = client.post(
        f"{API_ROOT}/chat/completions",
        json={
            "model": "openai:gpt-4o",
            "messages": test_messages,
            "user": "victim-user",
        },
        headers=api_key_header,
    )

    assert response.status_code == 403
    assert "does not match" in response.json()["detail"].lower()


def test_key_without_user_attaches_to_default(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    test_config: GatewayConfig,
) -> None:
    """A key created without a user_id attaches to the shared 'default' user."""
    user_response = client.get(f"{API_ROOT}/users/default", headers=master_key_header)
    assert user_response.status_code == 200
    user = user_response.json()
    assert user["user_id"] == "default"
    assert user["alias"] == "Default"

    api_key_header = {API_KEY_HEADER: f"Bearer {api_key_obj['key']}"}
    client.post(
        f"{API_ROOT}/chat/completions",
        json={
            "model": "openai:gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        },
        headers=api_key_header,
    )


def test_master_key_requires_user(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Test that master key requests require user field."""
    response = client.post(
        f"{API_ROOT}/chat/completions",
        json={
            "model": "openai:gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        },
        headers=master_key_header,
    )
    assert response.status_code == 400
    assert "user" in response.json()["detail"].lower()


def test_user_exceeded_budget_blocked_on_paid_model(
    client: TestClient,
    master_key_header: dict[str, str],
    test_messages: list[dict[str, str]],
) -> None:
    """Test that users who exceeded budget cannot use paid models."""
    budget_response = client.post(
        f"{API_ROOT}/budgets",
        json={"max_budget": 0.0},
        headers=master_key_header,
    )
    budget_id = budget_response.json()["budget_id"]

    client.post(
        f"{API_ROOT}/users",
        json={"user_id": "test-user-budget", "budget_id": budget_id},
        headers=master_key_header,
    )

    client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": "openai:gpt-4o",
            "input_price_per_million": 2.5,
            "output_price_per_million": 10.0,
        },
        headers=master_key_header,
    )

    response = client.post(
        f"{API_ROOT}/chat/completions",
        json={
            "model": "openai:gpt-4o",
            "messages": test_messages,
            "user": "test-user-budget",
        },
        headers=master_key_header,
    )
    assert response.status_code == 403
    assert "budget" in response.json()["detail"].lower()


def _too_large_setup(
    client: TestClient, master_key_header: dict[str, str], user_id: str, model_key: str, prices: tuple[float, float]
) -> None:
    """A $0.0015 budget with nothing spent, and a price for ``model_key``."""
    budget_id = client.post(f"{API_ROOT}/budgets", json={"max_budget": 0.0015}, headers=master_key_header).json()[
        "budget_id"
    ]
    client.post(f"{API_ROOT}/users", json={"user_id": user_id, "budget_id": budget_id}, headers=master_key_header)
    client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": model_key, "input_price_per_million": prices[0], "output_price_per_million": prices[1]},
        headers=master_key_header,
    )


def test_a_first_request_too_large_for_the_budget_says_so(
    client: TestClient,
    master_key_header: dict[str, str],
    test_messages: list[dict[str, str]],
) -> None:
    """Nothing is spent, so the refusal names the estimate and what remains, not an exceeded limit.

    The estimate prices the default output allowance when the request sets no
    output limit, which is what the refusal points the caller at.
    """
    _too_large_setup(client, master_key_header, "too-large-user", "openai:gpt-4o", (2.5, 10.0))

    response = client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": "openai:gpt-4o", "messages": test_messages, "user": "too-large-user"},
        headers=master_key_header,
    )

    assert response.status_code == 403
    assert response.json()["code"] == "budget_exceeded"
    assert response.headers["Otari-Budget-Scope"] == "user"
    detail = response.json()["detail"]
    assert detail.startswith("User 'too-large-user' has $0.0015 of budget remaining, less than this request's")
    assert "exceeded" not in detail
    assert detail.endswith(
        "The estimate assumes 1024 output tokens; a lower output token limit on the request may let it fit"
    )


def test_a_responses_refusal_names_its_own_output_limit(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Responses spells the limit max_output_tokens, so the refusal names the figure and not a field."""
    _too_large_setup(client, master_key_header, "too-large-responses-user", "openai:gpt-4o", (2.5, 10.0))

    response = client.post(
        f"{API_ROOT}/responses",
        json={
            "model": "openai:gpt-4o",
            "input": "Hello",
            "max_output_tokens": 500,
            "user": "too-large-responses-user",
        },
        headers=master_key_header,
    )

    assert response.status_code == 403
    detail = response.json()["detail"]
    assert "500 output tokens" in detail
    assert "max_tokens" not in detail


def test_an_input_only_price_gets_no_output_limit_advice(
    client: TestClient,
    master_key_header: dict[str, str],
    test_messages: list[dict[str, str]],
) -> None:
    """Output that costs nothing does not move the estimate, so lowering its limit is not offered."""
    _too_large_setup(client, master_key_header, "input-only-user", "openai:input-only", (1_000_000.0, 0.0))

    response = client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": "openai:input-only", "messages": test_messages, "user": "input-only-user"},
        headers=master_key_header,
    )

    assert response.status_code == 403
    detail = response.json()["detail"]
    assert detail.startswith("User 'input-only-user' has $0.0015 of budget remaining, less than this request's")
    assert "output token" not in detail


def test_user_exceeded_budget_allowed_on_free_model(
    client: TestClient,
    master_key_header: dict[str, str],
    test_messages: list[dict[str, str]],
) -> None:
    """Test that users who exceeded budget can still use free models."""
    budget_response = client.post(
        f"{API_ROOT}/budgets",
        json={"max_budget": 0.0},
        headers=master_key_header,
    )
    budget_id = budget_response.json()["budget_id"]

    client.post(
        f"{API_ROOT}/users",
        json={"user_id": "test-user-budget", "budget_id": budget_id},
        headers=master_key_header,
    )

    client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": "openai:gpt-4o-mini",
            "input_price_per_million": 0.0,
            "output_price_per_million": 0.0,
        },
        headers=master_key_header,
    )

    response = client.post(
        f"{API_ROOT}/chat/completions",
        json={
            "model": "openai:gpt-4o-mini",
            "messages": test_messages,
            "user": "test-user-budget",
        },
        headers=master_key_header,
    )
    assert response.status_code != 403


def test_user_exceeded_budget_blocked_on_unknown_pricing(
    client: TestClient,
    master_key_header: dict[str, str],
    test_messages: list[dict[str, str]],
) -> None:
    """Test that users who exceeded budget are blocked when model pricing is unknown."""
    budget_response = client.post(
        f"{API_ROOT}/budgets",
        json={"max_budget": 0.0},
        headers=master_key_header,
    )
    budget_id = budget_response.json()["budget_id"]

    client.post(
        f"{API_ROOT}/users",
        json={"user_id": "test-user-budget", "budget_id": budget_id},
        headers=master_key_header,
    )

    response = client.post(
        f"{API_ROOT}/chat/completions",
        json={
            "model": "unknown:model",
            "messages": test_messages,
            "user": "test-user-budget",
        },
        headers=master_key_header,
    )
    assert response.status_code == 403
    assert "budget" in response.json()["detail"].lower()


def _spent_user(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
    user_id: str,
    budget: dict[str, Any],
) -> None:
    """A user on ``budget`` (a $1 cap plus its cadence) with the whole dollar allowance already spent."""
    budget_id = client.post(
        f"{API_ROOT}/budgets", json={"max_budget": 1.0, **budget}, headers=master_key_header
    ).json()["budget_id"]
    client.post(f"{API_ROOT}/users", json={"user_id": user_id, "budget_id": budget_id}, headers=master_key_header)
    client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": "openai:gpt-4o", "input_price_per_million": 2.5, "output_price_per_million": 10.0},
        headers=master_key_header,
    )
    db_session.execute(update(User).where(User.user_id == user_id).values(spend=Decimal("1.0")))
    db_session.commit()


def _next_monday(moment: datetime) -> datetime:
    midnight = moment.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight + timedelta(days=7 - midnight.weekday())


def test_a_spent_calendar_budget_says_when_its_week_ends(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
    test_messages: list[dict[str, str]],
) -> None:
    """``Retry-After`` is the seconds until the ISO week ends, the window the refusing budget counts over."""
    _spent_user(client, master_key_header, db_session, "weekly-user", {"reset_alignment": "calendar_week"})

    before = datetime.now(UTC)
    response = client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": "openai:gpt-4o", "messages": test_messages, "user": "weekly-user"},
        headers=master_key_header,
    )
    after = datetime.now(UTC)

    assert response.status_code == 403
    assert response.json()["code"] == "budget_exceeded"
    assert response.headers["Otari-Budget-Scope"] == "user"
    # Bracketed by two clock reads, since the gateway reads its own between them.
    retry_after = int(response.headers["Retry-After"])
    assert seconds_until(_next_monday(after), after) <= retry_after <= seconds_until(_next_monday(before), before)


def test_a_spent_rolling_budget_says_when_its_period_ends(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
    test_messages: list[dict[str, str]],
) -> None:
    """A rolling period ends where the user's own window does, which only the gateway knows."""
    _spent_user(client, master_key_header, db_session, "hourly-user", {"budget_duration_sec": 3600})
    resets_at = db_session.execute(select(User.next_budget_reset_at).where(User.user_id == "hourly-user")).scalar_one()
    assert resets_at is not None

    before = datetime.now(UTC)
    response = client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": "openai:gpt-4o", "messages": test_messages, "user": "hourly-user"},
        headers=master_key_header,
    )
    after = datetime.now(UTC)

    assert response.status_code == 403
    retry_after = int(response.headers["Retry-After"])
    assert seconds_until(resets_at, after) <= retry_after <= seconds_until(resets_at, before)
    assert 0 < retry_after <= 3600


def test_a_messages_refusal_carries_the_same_wait(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """The Messages dialect renders its own error envelope, and keeps the header through it."""
    _spent_user(client, master_key_header, db_session, "daily-user", {"reset_alignment": "calendar_day"})

    response = client.post(
        f"{API_ROOT}/messages",
        json={
            "model": "openai:gpt-4o",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Hello"}],
            "metadata": {"user_id": "daily-user"},
        },
        headers=master_key_header,
    )

    assert response.status_code == 403
    assert response.headers["Otari-Error-Code"] == "budget_exceeded"
    assert 0 < int(response.headers["Retry-After"]) <= 86400


def test_a_budget_that_never_resets_sends_no_wait(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
    test_messages: list[dict[str, str]],
) -> None:
    """A lifetime cap has no window to wait out, so no ``Retry-After`` tells a client to try again."""
    _spent_user(client, master_key_header, db_session, "lifetime-user", {})

    response = client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": "openai:gpt-4o", "messages": test_messages, "user": "lifetime-user"},
        headers=master_key_header,
    )

    assert response.status_code == 403
    assert response.headers["Otari-Budget-Scope"] == "user"
    assert "Retry-After" not in response.headers


def test_a_request_larger_than_the_whole_budget_sends_no_wait(
    client: TestClient,
    master_key_header: dict[str, str],
    test_messages: list[dict[str, str]],
) -> None:
    """A fresh window would refuse the same estimate, so waiting for one would not help."""
    budget_id = client.post(
        f"{API_ROOT}/budgets", json={"max_budget": 0.0015, "budget_duration_sec": 3600}, headers=master_key_header
    ).json()["budget_id"]
    client.post(f"{API_ROOT}/users", json={"user_id": "tiny-user", "budget_id": budget_id}, headers=master_key_header)
    client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": "openai:gpt-4o", "input_price_per_million": 2.5, "output_price_per_million": 10.0},
        headers=master_key_header,
    )

    response = client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": "openai:gpt-4o", "messages": test_messages, "user": "tiny-user"},
        headers=master_key_header,
    )

    assert response.status_code == 403
    assert "remaining" in response.json()["detail"]
    assert "Retry-After" not in response.headers
