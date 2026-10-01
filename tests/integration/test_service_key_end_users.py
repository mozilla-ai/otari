"""A service key bills the end users it names, each under a budget of its own.

The end user's budget and the key's own ceiling (the pool) are both checked on
every request, and a key can only ever reach end users of its own user.
"""

from typing import Any
from unittest.mock import patch

from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT
from gateway.models.budgets import SCOPE_API_TOKEN, ScopedBudget
from gateway.models.usage import UsageLog
from gateway.models.users import User
from gateway.rate_limit import RateLimiter

from .conftest import MODEL_NAME


async def _mock_acompletion(**_kwargs: Any) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-x",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _chat(client: TestClient, headers: dict[str, str], user: str | None) -> Any:
    body: dict[str, Any] = {"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}]}
    if user is not None:
        body["user"] = user
    with patch("gateway.api.routes.chat.acompletion") as mock:
        mock.side_effect = _mock_acompletion
        return client.post(f"{API_ROOT}/chat/completions", json=body, headers=headers)


def _budget(client: TestClient, master_key_header: dict[str, str], *, request_limit: int) -> str:
    response = client.post(
        f"{API_ROOT}/budgets",
        json={"request_limit": request_limit, "budget_duration_sec": 86400},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    return str(response.json()["budget_id"])


def _service_key(
    client: TestClient, master_key_header: dict[str, str], owner: str, **settings: Any
) -> tuple[str, dict[str, str]]:
    client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": MODEL_NAME, "input_price_per_million": 1.0, "output_price_per_million": 1.0},
        headers=master_key_header,
    )
    response = client.post(
        f"{API_ROOT}/keys",
        json={"key_name": owner, "user_id": owner, "is_service_key": True, **settings},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"]), {API_KEY_HEADER: f"Bearer {response.json()['key']}"}


def _end_user(db_session: Session, owner: str, external_id: str) -> User | None:
    db_session.expire_all()
    return (
        db_session.query(User).filter(User.parent_user_id == owner, User.external_id == external_id).one_or_none()
    )


def test_an_end_user_is_created_on_first_use_and_billed(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    budget_id = _budget(client, master_key_header, request_limit=5)
    _, headers = _service_key(client, master_key_header, "svc", end_user_budget_id=budget_id)

    assert _chat(client, headers, "alice").status_code == 200
    assert _chat(client, headers, "alice").status_code == 200

    alice = _end_user(db_session, "svc", "alice")
    assert alice is not None
    assert alice.user_id != "alice"
    assert alice.budget_id == budget_id
    assert alice.next_budget_reset_at is not None
    assert alice.current_requests == 2
    owner = db_session.query(User).filter(User.user_id == "svc").one()
    assert owner.current_requests == 0
    rows = db_session.query(UsageLog).filter(UsageLog.user_id == alice.user_id).all()
    assert len(rows) == 2


def test_each_end_user_is_held_to_a_budget_of_its_own(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    budget_id = _budget(client, master_key_header, request_limit=1)
    _, headers = _service_key(client, master_key_header, "svc", end_user_budget_id=budget_id)

    assert _chat(client, headers, "alice").status_code == 200
    assert _chat(client, headers, "alice").status_code == 403
    assert _chat(client, headers, "bob").status_code == 200


def test_the_keys_own_ceiling_pools_every_end_user(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    pool_budget_id = _budget(client, master_key_header, request_limit=2)
    key_id, headers = _service_key(client, master_key_header, "svc")
    db_session.add(ScopedBudget(scope_type=SCOPE_API_TOKEN, scope_id=key_id, budget_id=pool_budget_id))
    db_session.commit()

    assert _chat(client, headers, "alice").status_code == 200
    assert _chat(client, headers, "bob").status_code == 200
    assert _chat(client, headers, "carol").status_code == 403


def test_end_users_belong_to_the_keys_owner(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """Two services naming the same end user get two rows, and naming another service's user reaches none of theirs."""
    _, svc_a = _service_key(client, master_key_header, "svc-a")
    _, svc_b = _service_key(client, master_key_header, "svc-b")

    assert _chat(client, svc_a, "alice").status_code == 200
    assert _chat(client, svc_b, "alice").status_code == 200
    assert _chat(client, svc_a, "svc-b").status_code == 200

    alice_a = _end_user(db_session, "svc-a", "alice")
    alice_b = _end_user(db_session, "svc-b", "alice")
    assert alice_a is not None and alice_b is not None
    assert alice_a.user_id != alice_b.user_id
    assert _end_user(db_session, "svc-a", "svc-b") is not None
    other_owner = db_session.query(User).filter(User.user_id == "svc-b").one()
    assert other_owner.current_requests == 0


def test_naming_its_own_user_bills_the_owner(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _, headers = _service_key(client, master_key_header, "svc")

    assert _chat(client, headers, "svc").status_code == 200
    assert _chat(client, headers, None).status_code == 200

    db_session.expire_all()
    assert db_session.query(User).filter(User.parent_user_id == "svc").count() == 0
    assert db_session.query(UsageLog).filter(UsageLog.user_id == "svc").count() == 2


def test_end_users_share_their_owners_rate_limit_and_a_limited_one_is_not_created(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """The limit is checked before an end user is created, so it bounds how fast a key adds them."""
    _, headers = _service_key(client, master_key_header, "svc")
    client.app.state.rate_limiter = RateLimiter(2)  # type: ignore[attr-defined]

    assert _chat(client, headers, "alice").status_code == 200
    assert _chat(client, headers, "bob").status_code == 200
    assert _chat(client, headers, "carol").status_code == 429
    assert _end_user(db_session, "svc", "carol") is None


def test_a_blocked_owner_stops_its_end_users(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _, headers = _service_key(client, master_key_header, "svc")
    assert _chat(client, headers, "alice").status_code == 200

    blocked = client.patch(f"{API_ROOT}/users/svc", json={"blocked": True}, headers=master_key_header)
    assert blocked.status_code == 200, blocked.text

    assert _chat(client, headers, "alice").status_code == 403


def test_an_overlong_end_user_id_is_refused(client: TestClient, master_key_header: dict[str, str]) -> None:
    _, headers = _service_key(client, master_key_header, "svc")

    assert _chat(client, headers, "x" * 257).status_code == 400


def test_a_key_that_is_not_a_service_key_still_cannot_name_a_user(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    response = client.post(f"{API_ROOT}/keys", json={"user_id": "plain"}, headers=master_key_header)
    headers = {API_KEY_HEADER: f"Bearer {response.json()['key']}"}

    assert _chat(client, headers, "alice").status_code == 403
    assert _end_user(db_session, "plain", "alice") is None


def test_the_service_key_settings_round_trip(client: TestClient, master_key_header: dict[str, str]) -> None:
    created = client.post(f"{API_ROOT}/keys", json={"key_name": "plain"}, headers=master_key_header)
    assert created.status_code == 200
    assert created.json()["is_service_key"] is False
    assert created.json()["end_user_budget_id"] is None

    budget_id = _budget(client, master_key_header, request_limit=1)
    key_id = created.json()["id"]
    updated = client.patch(
        f"{API_ROOT}/keys/{key_id}",
        json={"is_service_key": True, "end_user_budget_id": budget_id},
        headers=master_key_header,
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["is_service_key"] is True
    assert updated.json()["end_user_budget_id"] == budget_id

    cleared = client.patch(f"{API_ROOT}/keys/{key_id}", json={"end_user_budget_id": None}, headers=master_key_header)
    assert cleared.json()["end_user_budget_id"] is None
    assert cleared.json()["is_service_key"] is True


def test_an_unknown_end_user_budget_is_refused(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.post(
        f"{API_ROOT}/keys",
        json={"is_service_key": True, "end_user_budget_id": "no-such-budget"},
        headers=master_key_header,
    )
    assert response.status_code == 404


def test_a_deleted_end_user_comes_back_with_what_it_spent(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """Deleting an end user cannot be used to clear its budget."""
    budget_id = _budget(client, master_key_header, request_limit=1)
    _, headers = _service_key(client, master_key_header, "svc", end_user_budget_id=budget_id)
    assert _chat(client, headers, "alice").status_code == 200
    alice = _end_user(db_session, "svc", "alice")
    assert alice is not None

    deleted = client.delete(f"{API_ROOT}/users/{alice.user_id}", headers=master_key_header)
    assert deleted.status_code in (200, 204), deleted.text

    assert _chat(client, headers, "alice").status_code == 403
    revived = _end_user(db_session, "svc", "alice")
    assert revived is not None
    assert revived.user_id == alice.user_id
    assert revived.deleted_at is None
