"""A service key manages its own end users, authenticated with that key alone.

The service holds one secret: it reads, lists, blocks and moves the end users
its key bills without the master key and without knowing its key's id, and it
reaches no other user's end users nor a budget outside its key's list.
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.adapters.telemetry_storage_adapter import DatabaseTelemetryStorageAdapter
from gateway.core.config import API_KEY_HEADER, API_ROOT
from gateway.models.tenancy import Workspace
from gateway.models.traces import Trace
from gateway.models.users import User

from .conftest import MODEL_NAME

END_USERS = f"{API_ROOT}/end-users"
END_USER_BUDGETS = f"{API_ROOT}/end-user-budgets"


async def _mock_acompletion(**_kwargs: Any) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-x",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _chat(client: TestClient, headers: dict[str, str], user: str) -> Any:
    body = {"model": MODEL_NAME, "user": user, "messages": [{"role": "user", "content": "hi"}]}
    with patch("gateway.api.routes.chat.acompletion") as mock:
        mock.side_effect = _mock_acompletion
        return client.post(f"{API_ROOT}/chat/completions", json=body, headers=headers)


def _budgets(client: TestClient, master_key_header: dict[str, str]) -> None:
    client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": MODEL_NAME, "input_price_per_million": 1.0, "output_price_per_million": 1.0},
        headers=master_key_header,
    )
    for budget_id in ("eu-ai", "eu-memories", "eu-other"):
        created = client.put(
            f"{API_ROOT}/budgets/{budget_id}",
            json={"budget_duration_sec": 86400, "request_limit": 5},
            headers=master_key_header,
        )
        assert created.status_code in (200, 201), created.text


def _key(client: TestClient, master_key_header: dict[str, str], owner: str, **settings: Any) -> dict[str, str]:
    created = client.post(
        f"{API_ROOT}/keys", json={"key_name": owner, "user_id": owner, **settings}, headers=master_key_header
    )
    assert created.status_code == 200, created.text
    return {API_KEY_HEADER: f"Bearer {created.json()['key']}"}


def _service_key(client: TestClient, master_key_header: dict[str, str], owner: str) -> dict[str, str]:
    """A service key that may assign ``eu-ai`` (its default) and ``eu-memories``, and not ``eu-other``."""
    return _key(
        client,
        master_key_header,
        owner,
        is_service_key=True,
        end_user_budget_ids=["eu-ai", "eu-memories"],
        end_user_budget_id="eu-ai",
    )


def test_a_service_key_puts_reads_blocks_and_moves_its_end_user(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _budgets(client, master_key_header)
    mlpa = _service_key(client, master_key_header, "mlpa")
    url = f"{END_USERS}/fxa1:ai"

    created = client.put(url, json={"budget_id": "eu-memories"}, headers=mlpa)
    repeated = client.put(url, json={"budget_id": "eu-memories"}, headers=mlpa)
    read = client.get(url, headers=mlpa)
    blocked = client.patch(url, json={"blocked": True}, headers=mlpa)
    refused = _chat(client, mlpa, "fxa1:ai")
    moved = client.patch(url, json={"blocked": False, "budget_id": "eu-ai"}, headers=mlpa)

    assert created.status_code == 201, created.text
    assert repeated.status_code == 200
    assert read.status_code == 200
    assert read.json()["external_id"] == "fxa1:ai"
    assert read.json()["owner_user_id"] == "mlpa"
    assert read.json()["budget_id"] == "eu-memories"
    assert blocked.json()["blocked"] is True
    assert refused.status_code == 403
    assert refused.headers["Otari-Error-Code"] == "user_blocked"
    assert moved.status_code == 200
    assert moved.json() | {"blocked": False, "budget_id": "eu-ai"} == moved.json()
    assert _chat(client, mlpa, "fxa1:ai").status_code == 200


def test_a_service_key_reaches_an_end_user_whose_id_contains_a_slash(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """An App Attest key id is standard base64, so about half of them contain '/'."""
    _budgets(client, master_key_header)
    mlpa = _service_key(client, master_key_header, "mlpa")
    device = "/Ab+cd//ef0123456789ABCDEFghijklmnopqrstuvw="
    named = {"external_id": device}

    served = _chat(client, mlpa, device)
    read = client.get(END_USERS, params=named, headers=mlpa)
    blocked = client.patch(END_USERS, params=named, json={"blocked": True}, headers=mlpa)
    refused = _chat(client, mlpa, device)
    moved = client.patch(END_USERS, params=named, json={"blocked": False, "budget_id": "eu-memories"}, headers=mlpa)
    created = client.put(
        END_USERS, params={"external_id": "org/alice"}, json={"budget_id": "eu-memories"}, headers=mlpa
    )
    repeated = client.put(
        END_USERS, params={"external_id": "org/alice"}, json={"budget_id": "eu-memories"}, headers=mlpa
    )
    nobody = client.get(END_USERS, params={"external_id": "org", "include_total": True}, headers=mlpa)

    assert served.status_code == 200, served.text
    assert read.status_code == 200, read.text
    assert [user["external_id"] for user in read.json()] == [device]
    assert read.json()[0]["owner_user_id"] == "mlpa"
    assert blocked.status_code == 200, blocked.text
    assert blocked.json()["blocked"] is True
    assert refused.status_code == 403
    assert refused.headers["Otari-Error-Code"] == "user_blocked"
    assert moved.status_code == 200
    assert moved.json() | {"external_id": device, "blocked": False, "budget_id": "eu-memories"} == moved.json()
    assert created.status_code == 201, created.text
    assert created.json()["external_id"] == "org/alice"
    assert repeated.status_code == 200
    assert nobody.json() == []
    assert nobody.headers["Otari-Total-Count"] == "0"
    assert _chat(client, mlpa, device).status_code == 200


def test_the_query_form_names_only_the_calling_keys_end_users(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _budgets(client, master_key_header)
    mlpa = _service_key(client, master_key_header, "mlpa")
    other = _service_key(client, master_key_header, "other-svc")
    named = {"external_id": "org/alice"}
    assert client.put(END_USERS, params=named, json={"budget_id": "eu-memories"}, headers=mlpa).status_code == 201

    read = client.get(END_USERS, params=named, headers=other)
    patched = client.patch(END_USERS, params=named, json={"blocked": True}, headers=other)
    unnamed = client.patch(END_USERS, json={"blocked": True}, headers=mlpa)
    empty = client.put(END_USERS, params={"external_id": ""}, json={"budget_id": "eu-ai"}, headers=mlpa)
    empty_filter = client.get(END_USERS, params={"external_id": ""}, headers=mlpa)

    assert read.json() == []
    assert patched.status_code == 404
    assert unnamed.status_code == 422
    assert empty.status_code == 422
    assert empty_filter.status_code == 422
    assert client.get(END_USERS, params=named, headers=mlpa).json()[0]["blocked"] is False


def test_a_dot_segment_is_refused_only_in_a_path(client: TestClient, master_key_header: dict[str, str]) -> None:
    _budgets(client, master_key_header)
    mlpa = _service_key(client, master_key_header, "mlpa")

    for dots in (".", "..", "a/./b"):
        created = client.put(END_USERS, params={"external_id": dots}, json={"budget_id": "eu-ai"}, headers=mlpa)
        assert created.status_code == 201, dots
    # Encoded, so the client does not collapse them before Otari sees them.
    for encoded in ("%2E", "%2E%2E", "a/%2E/b"):
        path = f"{END_USERS}/{encoded}"
        refused = client.get(path, headers=mlpa)
        assert refused.status_code == 400, encoded
        assert "external_id" in refused.json()["detail"]
        assert client.put(path, json={"budget_id": "eu-ai"}, headers=mlpa).status_code == 400, encoded
        assert client.patch(path, json={"blocked": True}, headers=mlpa).status_code == 400, encoded


def test_a_service_key_cannot_move_an_end_user_off_its_list(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _budgets(client, master_key_header)
    mlpa = _service_key(client, master_key_header, "mlpa")
    url = f"{END_USERS}/fxa1"
    assert client.put(url, json={"budget_id": "eu-ai"}, headers=mlpa).status_code == 201

    moved = client.patch(url, json={"budget_id": "eu-other"}, headers=mlpa)
    put = client.put(url, json={"budget_id": "eu-other"}, headers=mlpa)
    created = client.put(f"{END_USERS}/fxa2", json={"budget_id": "eu-other"}, headers=mlpa)

    for refused in (moved, put, created):
        assert refused.status_code == 403
        assert refused.json()["code"] == "end_user_budget_not_allowed"
    assert client.get(url, headers=mlpa).json()["budget_id"] == "eu-ai"
    assert client.get(f"{END_USERS}/fxa2", headers=mlpa).status_code == 404


def test_a_service_key_never_reaches_another_users_end_users(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _budgets(client, master_key_header)
    mlpa = _service_key(client, master_key_header, "mlpa")
    other = _service_key(client, master_key_header, "other-svc")
    assert client.put(f"{END_USERS}/alice", json={"budget_id": "eu-memories"}, headers=mlpa).status_code == 201

    read = client.get(f"{END_USERS}/alice", headers=other)
    patched = client.patch(f"{END_USERS}/alice", json={"blocked": True}, headers=other)
    listed = client.get(END_USERS, headers=other)
    counted = client.get(END_USER_BUDGETS, headers=other)
    # The same id names an end user of its own, and leaves mlpa's untouched.
    own = client.put(f"{END_USERS}/alice", json={"budget_id": "eu-ai"}, headers=other)

    assert read.status_code == 404
    assert patched.status_code == 404
    assert listed.json() == []
    assert all(budget["end_user_count"] == 0 for budget in counted.json())
    assert own.status_code == 201
    assert own.json()["owner_user_id"] == "other-svc"
    theirs = client.get(f"{END_USERS}/alice", headers=mlpa).json()
    assert theirs["user_id"] != own.json()["user_id"]
    assert theirs["blocked"] is False
    assert theirs["budget_id"] == "eu-memories"


def test_service_keys_of_one_user_share_its_end_users(client: TestClient, master_key_header: dict[str, str]) -> None:
    _budgets(client, master_key_header)
    first = _service_key(client, master_key_header, "mlpa")
    second = _service_key(client, master_key_header, "mlpa")
    assert client.put(f"{END_USERS}/alice", json={"budget_id": "eu-ai"}, headers=first).status_code == 201

    assert client.get(f"{END_USERS}/alice", headers=second).status_code == 200


def test_only_a_service_key_authenticates(client: TestClient, master_key_header: dict[str, str]) -> None:
    _budgets(client, master_key_header)
    plain = _key(client, master_key_header, "plain")
    requests = [
        ("GET", END_USERS, None),
        ("GET", f"{END_USERS}/alice", None),
        ("PUT", f"{END_USERS}/alice", {"budget_id": "eu-ai"}),
        ("PATCH", f"{END_USERS}/alice", {"blocked": True}),
        ("PUT", f"{END_USERS}?external_id=org%2Falice", {"budget_id": "eu-ai"}),
        ("PATCH", f"{END_USERS}?external_id=org%2Falice", {"blocked": True}),
        ("GET", END_USER_BUDGETS, None),
        ("DELETE", f"{END_USERS}/alice", None),
        ("DELETE", f"{END_USERS}?external_id=alice", None),
        ("DELETE", f"{END_USERS}?created_before=2999-01-01T00:00:00Z", None),
    ]

    for method, url, body in requests:
        as_plain = client.request(method, url, json=body, headers=plain)
        as_master = client.request(method, url, json=body, headers=master_key_header)
        anonymous = client.request(method, url, json=body)
        assert as_plain.status_code == 403, (method, url)
        assert as_plain.json()["detail"] == "This endpoint requires a service key."
        # The master key names no key, so it has no end users to scope a request to.
        assert as_master.status_code == 401, (method, url)
        assert anonymous.status_code == 401, (method, url)


def test_an_inactive_service_key_is_refused(client: TestClient, master_key_header: dict[str, str]) -> None:
    _budgets(client, master_key_header)
    created = client.post(
        f"{API_ROOT}/keys", json={"user_id": "mlpa", "is_service_key": True}, headers=master_key_header
    ).json()
    client.patch(f"{API_ROOT}/keys/{created['id']}", json={"is_active": False}, headers=master_key_header)
    mlpa = {API_KEY_HEADER: f"Bearer {created['key']}"}

    assert client.get(END_USERS, headers=mlpa).status_code == 401


def test_a_service_key_lists_its_end_users(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _budgets(client, master_key_header)
    mlpa = _service_key(client, master_key_header, "mlpa")
    other = _service_key(client, master_key_header, "other-svc")
    for external_id, budget_id in (("a", "eu-ai"), ("b", "eu-memories"), ("c", "eu-ai"), ("gone", "eu-ai")):
        assert client.put(f"{END_USERS}/{external_id}", json={"budget_id": budget_id}, headers=mlpa).status_code == 201
    client.patch(f"{END_USERS}/c", json={"blocked": True}, headers=mlpa)
    client.put(f"{END_USERS}/theirs", json={"budget_id": "eu-ai"}, headers=other)
    gone = db_session.query(User).filter(User.external_id == "gone").one()
    gone.deleted_at = datetime.now(UTC)
    db_session.commit()

    everyone = client.get(END_USERS, params={"include_total": True}, headers=mlpa)
    page = client.get(END_USERS, params={"skip": 1, "limit": 1}, headers=mlpa)
    on_ai = client.get(END_USERS, params={"budget_id": "eu-ai", "include_total": True}, headers=mlpa)
    blocked = client.get(END_USERS, params={"blocked": True}, headers=mlpa)

    assert everyone.status_code == 200, everyone.text
    assert [user["external_id"] for user in everyone.json()] == ["a", "b", "c"]
    assert everyone.headers["Otari-Total-Count"] == "3"
    assert [user["external_id"] for user in page.json()] == ["b"]
    assert "Otari-Total-Count" not in page.headers
    assert [user["external_id"] for user in on_ai.json()] == ["a", "c"]
    assert on_ai.headers["Otari-Total-Count"] == "2"
    assert [user["external_id"] for user in blocked.json()] == ["c"]


def test_a_service_key_counts_its_end_users_on_each_of_its_budgets(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _budgets(client, master_key_header)
    mlpa = _service_key(client, master_key_header, "mlpa")
    other = _service_key(client, master_key_header, "other-svc")
    assert _chat(client, mlpa, "fxa1").status_code == 200
    client.put(f"{END_USERS}/fxa2", json={"budget_id": "eu-ai"}, headers=mlpa)
    client.put(f"{END_USERS}/fxa3", json={"budget_id": "eu-memories"}, headers=mlpa)
    client.patch(f"{END_USERS}/fxa2", json={"blocked": True}, headers=mlpa)
    client.put(f"{END_USERS}/theirs", json={"budget_id": "eu-ai"}, headers=other)

    counted = client.get(END_USER_BUDGETS, headers=mlpa)

    assert counted.status_code == 200, counted.text
    assert counted.json() == [
        {"budget_id": "eu-ai", "is_default": True, "end_user_count": 2, "blocked_end_user_count": 1},
        {"budget_id": "eu-memories", "is_default": False, "end_user_count": 1, "blocked_end_user_count": 0},
    ]


def test_a_service_key_of_a_blocked_owner_manages_none_of_its_end_users(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _budgets(client, master_key_header)
    mlpa = _service_key(client, master_key_header, "mlpa")
    url = f"{END_USERS}/alice"
    assert client.put(url, json={"budget_id": "eu-ai"}, headers=mlpa).status_code == 201
    assert client.patch(url, json={"blocked": True}, headers=mlpa).status_code == 200
    assert client.patch(f"{API_ROOT}/users/mlpa", json={"blocked": True}, headers=master_key_header).status_code == 200

    requests = [
        ("GET", END_USERS, None),
        ("GET", url, None),
        ("PUT", url, {"budget_id": "eu-memories"}),
        ("PATCH", url, {"blocked": False, "budget_id": "eu-memories"}),
        ("GET", END_USER_BUDGETS, None),
        ("DELETE", url, None),
        ("DELETE", f"{END_USERS}?external_id=alice", None),
        ("DELETE", f"{END_USERS}?created_before=2999-01-01T00:00:00Z", None),
    ]
    for method, target, body in requests:
        refused = client.request(method, target, json=body, headers=mlpa)
        assert refused.status_code == 403, (method, target, refused.text)

    client.patch(f"{API_ROOT}/users/mlpa", json={"blocked": False}, headers=master_key_header)
    read = client.get(url, headers=mlpa)
    assert read.json()["blocked"] is True
    assert read.json()["budget_id"] == "eu-ai"


def test_a_key_with_a_cap_refuses_one_end_user_more(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _budgets(client, master_key_header)
    mlpa = _key(
        client,
        master_key_header,
        "mlpa",
        is_service_key=True,
        end_user_budget_ids=["eu-ai"],
        end_user_budget_id="eu-ai",
        max_end_users=2,
    )

    first = _chat(client, mlpa, "a")
    second = client.put(f"{END_USERS}/b", json={"budget_id": "eu-ai"}, headers=mlpa)
    by_request = _chat(client, mlpa, "c")
    by_put = client.put(f"{END_USERS}/c", json={"budget_id": "eu-ai"}, headers=mlpa)
    existing = _chat(client, mlpa, "a")

    assert first.status_code == 200, first.text
    assert second.status_code == 201, second.text
    assert by_request.status_code == 403
    assert by_request.headers["Otari-Error-Code"] == "end_user_limit_reached"
    assert by_put.status_code == 403
    assert by_put.json()["code"] == "end_user_limit_reached"
    # An end user the key already has keeps working when the cap is full.
    assert existing.status_code == 200, existing.text

    # A deleted end user frees its place, and reviving it takes one again.
    gone = db_session.query(User).filter(User.external_id == "b").one()
    gone.deleted_at = datetime.now(UTC)
    db_session.commit()
    assert _chat(client, mlpa, "c").status_code == 200
    assert _chat(client, mlpa, "b").status_code == 403


def test_two_keys_of_one_owner_count_their_own_budgets(client: TestClient, master_key_header: dict[str, str]) -> None:
    """A guest key and a signed-in key of one owner keep separate caps, each over the budgets it assigns."""
    _budgets(client, master_key_header)
    signed_in = _key(
        client,
        master_key_header,
        "mlpa",
        is_service_key=True,
        end_user_budget_ids=["eu-ai"],
        end_user_budget_id="eu-ai",
        max_end_users=1,
    )
    guests = client.post(
        f"{API_ROOT}/keys",
        json={
            "key_name": "mlpa-guest",
            "user_id": "mlpa",
            "is_service_key": True,
            "end_user_budget_ids": ["eu-memories"],
            "end_user_budget_id": "eu-memories",
            "max_end_users": 1,
        },
        headers=master_key_header,
    ).json()
    guest = {API_KEY_HEADER: f"Bearer {guests['key']}"}

    assert _chat(client, signed_in, "fxa1:ai").status_code == 200
    assert _chat(client, guest, "guest1:ai-guest").status_code == 200
    assert _chat(client, signed_in, "fxa2:ai").status_code == 403
    assert _chat(client, guest, "guest2:ai-guest").status_code == 403


def test_moving_an_end_user_onto_a_capped_keys_budget_takes_a_place(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """An end user another key of the owner started counts once it is moved onto this key's budgets."""
    _budgets(client, master_key_header)
    other = _key(
        client,
        master_key_header,
        "mlpa",
        is_service_key=True,
        end_user_budget_ids=["eu-ai"],
        end_user_budget_id="eu-ai",
    )
    capped = client.post(
        f"{API_ROOT}/keys",
        json={
            "key_name": "mlpa-guest",
            "user_id": "mlpa",
            "is_service_key": True,
            "end_user_budget_ids": ["eu-memories", "eu-other"],
            "end_user_budget_id": "eu-memories",
            "max_end_users": 1,
        },
        headers=master_key_header,
    ).json()
    guest = {API_KEY_HEADER: f"Bearer {capped['key']}"}
    for external_id in ("a", "b"):
        assert client.put(f"{END_USERS}/{external_id}", json={"budget_id": "eu-ai"}, headers=other).status_code == 201

    moved = client.put(f"{END_USERS}/a", json={"budget_id": "eu-memories"}, headers=guest)
    by_put = client.put(f"{END_USERS}/b", json={"budget_id": "eu-memories"}, headers=guest)
    by_patch = client.patch(f"{END_USERS}/b", json={"budget_id": "eu-memories"}, headers=guest)
    within = client.patch(f"{END_USERS}/a", json={"budget_id": "eu-other"}, headers=guest)

    assert moved.status_code == 200, moved.text
    assert by_put.status_code == 403
    assert by_put.json()["code"] == "end_user_limit_reached"
    assert by_patch.status_code == 403
    assert by_patch.json()["code"] == "end_user_limit_reached"
    # A move between the key's own budgets takes no new place.
    assert within.status_code == 200, within.text


def test_a_key_with_no_default_counts_end_users_on_no_budget(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """A request naming no budget starts an end user on none, and that still takes a place."""
    _budgets(client, master_key_header)
    mlpa = _key(client, master_key_header, "mlpa", is_service_key=True, end_user_budget_ids=["eu-ai"], max_end_users=1)

    first = _chat(client, mlpa, "a")
    second = _chat(client, mlpa, "b")

    assert first.status_code == 200, first.text
    assert second.status_code == 403
    assert second.headers["Otari-Error-Code"] == "end_user_limit_reached"


def test_the_cap_is_read_and_changed_through_the_keys_api(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _budgets(client, master_key_header)
    created = client.post(
        f"{API_ROOT}/keys",
        json={"key_name": "mlpa", "user_id": "mlpa", "is_service_key": True, "max_end_users": 10},
        headers=master_key_header,
    ).json()
    raised = client.patch(f"{API_ROOT}/keys/{created['id']}", json={"max_end_users": 20}, headers=master_key_header)
    cleared = client.patch(f"{API_ROOT}/keys/{created['id']}", json={"max_end_users": None}, headers=master_key_header)
    negative = client.patch(f"{API_ROOT}/keys/{created['id']}", json={"max_end_users": -1}, headers=master_key_header)

    assert created["max_end_users"] == 10
    assert raised.json()["max_end_users"] == 20
    assert cleared.json()["max_end_users"] is None
    assert negative.status_code == 422


def test_a_service_key_deletes_its_end_user_and_a_request_revives_it(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _budgets(client, master_key_header)
    mlpa = _service_key(client, master_key_header, "mlpa")
    assert _chat(client, mlpa, "guest1:ai").status_code == 200
    spent = client.get(f"{END_USERS}/guest1:ai", headers=mlpa).json()["current_requests"]

    deleted = client.delete(f"{END_USERS}/guest1:ai", headers=mlpa)
    listed = client.get(END_USERS, headers=mlpa)
    read = client.get(f"{END_USERS}/guest1:ai", headers=mlpa)
    again = client.delete(f"{END_USERS}/guest1:ai", headers=mlpa)

    assert deleted.status_code == 204, deleted.text
    assert listed.json() == []
    assert read.status_code == 404
    assert again.status_code == 404
    # Named again, it comes back with what it had used, so a delete cannot reset a budget.
    assert _chat(client, mlpa, "guest1:ai").status_code == 200
    assert client.get(f"{END_USERS}/guest1:ai", headers=mlpa).json()["current_requests"] == spent + 1


def test_the_query_form_deletes_any_id(client: TestClient, master_key_header: dict[str, str]) -> None:
    _budgets(client, master_key_header)
    mlpa = _service_key(client, master_key_header, "mlpa")
    external_id = "ios/key+id=="
    client.put(END_USERS, params={"external_id": external_id}, json={"budget_id": "eu-ai"}, headers=mlpa)

    deleted = client.delete(END_USERS, params={"external_id": external_id}, headers=mlpa)

    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {"deleted": 1}
    assert client.get(END_USERS, params={"external_id": external_id}, headers=mlpa).json() == []


def test_end_users_created_before_a_time_are_deleted_a_page_at_a_time(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _budgets(client, master_key_header)
    mlpa = _service_key(client, master_key_header, "mlpa")
    other = _service_key(client, master_key_header, "other-svc")
    for external_id in ("old1", "old2", "old3", "new"):
        client.put(f"{END_USERS}/{external_id}", json={"budget_id": "eu-ai"}, headers=mlpa)
    client.put(f"{END_USERS}/theirs-old", json={"budget_id": "eu-ai"}, headers=other)
    week_ago = datetime(2026, 1, 1, tzinfo=UTC)
    for user in db_session.query(User).filter(User.external_id.in_(["old1", "old2", "old3", "theirs-old"])):
        user.created_at = week_ago
    db_session.commit()
    cutoff = "2026-01-02T00:00:00Z"

    first = client.delete(END_USERS, params={"created_before": cutoff, "limit": 2}, headers=mlpa)
    second = client.delete(END_USERS, params={"created_before": cutoff, "limit": 2}, headers=mlpa)
    done = client.delete(END_USERS, params={"created_before": cutoff, "limit": 2}, headers=mlpa)

    assert first.json() == {"deleted": 2}
    assert second.json() == {"deleted": 1}
    assert done.json() == {"deleted": 0}
    assert [user["external_id"] for user in client.get(END_USERS, headers=mlpa).json()] == ["new"]
    # Another owner's end users are out of reach, however old.
    assert [user["external_id"] for user in client.get(END_USERS, headers=other).json()] == ["theirs-old"]


def test_a_delete_names_exactly_one_of_an_id_and_a_time(client: TestClient, master_key_header: dict[str, str]) -> None:
    _budgets(client, master_key_header)
    mlpa = _service_key(client, master_key_header, "mlpa")

    neither = client.delete(END_USERS, headers=mlpa)
    both = client.delete(END_USERS, params={"external_id": "a", "created_before": "2026-01-01T00:00:00Z"}, headers=mlpa)

    assert neither.status_code == 400
    assert both.status_code == 400


def test_a_service_key_cannot_delete_another_users_end_user(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _budgets(client, master_key_header)
    mlpa = _service_key(client, master_key_header, "mlpa")
    other = _service_key(client, master_key_header, "other-svc")
    client.put(f"{END_USERS}/theirs", json={"budget_id": "eu-ai"}, headers=other)

    refused = client.delete(f"{END_USERS}/theirs", headers=mlpa)

    assert refused.status_code == 404
    assert [user["external_id"] for user in client.get(END_USERS, headers=other).json()] == ["theirs"]


def test_deleting_an_end_user_erases_its_agent_traces(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _budgets(client, master_key_header)
    mlpa = _service_key(client, master_key_header, "mlpa")
    doomed = client.put(f"{END_USERS}/doomed", json={"budget_id": "eu-ai"}, headers=mlpa).json()["user_id"]
    kept = client.put(f"{END_USERS}/kept", json={"budget_id": "eu-ai"}, headers=mlpa).json()["user_id"]
    workspace = db_session.query(Workspace).first()
    assert workspace is not None
    now = datetime.now(UTC)
    for trace_id, user_id in (("s-doomed", doomed), ("s-kept", kept)):
        db_session.add(
            Trace(
                workspace_id=workspace.id,
                trace_id=trace_id,
                user_id=user_id,
                session_source="none",
                started_at=now,
                last_activity_at=now,
            )
        )
    db_session.commit()

    deleted = client.delete(f"{END_USERS}/doomed", headers=mlpa)

    assert deleted.status_code == 204, deleted.text
    db_session.expire_all()
    assert [row.trace_id for row in db_session.query(Trace).all()] == ["s-kept"]


def test_an_end_user_stays_live_when_its_telemetry_cannot_be_erased(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Erasure runs before the soft-delete, so a failed one deletes nothing and the call can be repeated."""
    _budgets(client, master_key_header)
    mlpa = _service_key(client, master_key_header, "mlpa")
    client.put(f"{END_USERS}/guest", json={"budget_id": "eu-ai"}, headers=mlpa)

    # Patched onto the class, so it is called bound and takes the instance.
    async def _boom(_self: DatabaseTelemetryStorageAdapter, *, user_id: str) -> int:
        raise RuntimeError("telemetry store unreachable")

    with patch.object(DatabaseTelemetryStorageAdapter, "purge_user", _boom):
        by_path = client.delete(f"{END_USERS}/guest", headers=mlpa)
        by_age = client.delete(END_USERS, params={"created_before": "2999-01-01T00:00:00Z"}, headers=mlpa)

    assert by_path.status_code == 500
    assert "telemetry" in by_path.json()["detail"].lower()
    assert by_age.status_code == 500
    assert [user["external_id"] for user in client.get(END_USERS, headers=mlpa).json()] == ["guest"]
    assert client.delete(f"{END_USERS}/guest", headers=mlpa).status_code == 204
