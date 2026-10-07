"""One service key that starts each end user on one of several budgets, chosen per request.

A key lists the budgets it may assign, a request names one in
``Otari-End-User-Budget``, budgets take ids the caller chooses, and end users
are managed by the id the service names them by.
"""

from typing import Any
from unittest.mock import patch

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.core.config import API_KEY_HEADER, API_ROOT
from gateway.models.budgets import Budget
from gateway.models.tenancy import Organization
from gateway.models.users import User
from gateway.repositories.budgets.budget_repository import BudgetRepository
from gateway.services.secret_box import generate_secret_key

from .conftest import MODEL_NAME

BUDGET_HEADER = "Otari-End-User-Budget"


async def _mock_acompletion(**_kwargs: Any) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-x",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _chat(client: TestClient, headers: dict[str, str], user: str, budget: str | None = None) -> Any:
    body = {"model": MODEL_NAME, "user": user, "messages": [{"role": "user", "content": "hi"}]}
    if budget is not None:
        headers = {**headers, BUDGET_HEADER: budget}
    with patch("gateway.api.routes.chat.acompletion") as mock:
        mock.side_effect = _mock_acompletion
        return client.post(f"{API_ROOT}/chat/completions", json=body, headers=headers)


def _put_budget(client: TestClient, master_key_header: dict[str, str], budget_id: str, **fields: Any) -> Any:
    return client.put(
        f"{API_ROOT}/budgets/{budget_id}",
        json={"budget_duration_sec": 86400, **fields},
        headers=master_key_header,
    )


def _service_key(client: TestClient, master_key_header: dict[str, str], owner: str, **settings: Any) -> Any:
    client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": MODEL_NAME, "input_price_per_million": 1.0, "output_price_per_million": 1.0},
        headers=master_key_header,
    )
    return client.post(
        f"{API_ROOT}/keys",
        json={"key_name": owner, "user_id": owner, "is_service_key": True, **settings},
        headers=master_key_header,
    )


def _mlpa(client: TestClient, master_key_header: dict[str, str]) -> tuple[str, dict[str, str]]:
    """A service key that may assign ``eu-ai`` (its default), ``eu-memories`` and ``eu-ai-dev``."""
    for budget_id in ("eu-ai", "eu-memories", "eu-ai-dev", "eu-other"):
        assert _put_budget(client, master_key_header, budget_id, request_limit=5).status_code == 201
    created = _service_key(
        client,
        master_key_header,
        "mlpa",
        end_user_budget_ids=["eu-ai", "eu-memories", "eu-ai-dev"],
        end_user_budget_id="eu-ai",
    )
    assert created.status_code == 200, created.text
    return created.json()["id"], {API_KEY_HEADER: f"Bearer {created.json()['key']}"}


def _end_user(client: TestClient, master_key_header: dict[str, str], key_id: str, external_id: str) -> Any:
    return client.get(f"{API_ROOT}/keys/{key_id}/end-users/{external_id}", headers=master_key_header)


# --- the key's list ---------------------------------------------------------


def test_a_key_lists_the_budgets_it_may_assign(client: TestClient, master_key_header: dict[str, str]) -> None:
    key_id, _ = _mlpa(client, master_key_header)

    fetched = client.get(f"{API_ROOT}/keys/{key_id}", headers=master_key_header).json()

    assert fetched["end_user_budget_ids"] == ["eu-ai", "eu-memories", "eu-ai-dev"]
    assert fetched["end_user_budget_id"] == "eu-ai"


def test_a_key_with_only_a_default_lists_the_default(client: TestClient, master_key_header: dict[str, str]) -> None:
    assert _put_budget(client, master_key_header, "solo", request_limit=1).status_code == 201

    created = _service_key(client, master_key_header, "svc", end_user_budget_id="solo")
    bare = _service_key(client, master_key_header, "svc-bare")

    assert created.json()["end_user_budget_ids"] == ["solo"]
    assert bare.json()["end_user_budget_ids"] == []


def test_a_default_off_the_list_is_refused(client: TestClient, master_key_header: dict[str, str]) -> None:
    for budget_id in ("one", "two"):
        _put_budget(client, master_key_header, budget_id)

    refused = _service_key(client, master_key_header, "svc", end_user_budget_ids=["one"], end_user_budget_id="two")

    assert refused.status_code == 400


def test_an_unknown_or_tenant_budget_on_the_list_is_refused(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    organization_id = db_session.query(col(Organization.id)).first()
    assert organization_id is not None
    tenant_budget = Budget(request_limit=1, organization_id=organization_id[0])
    db_session.add(tenant_budget)
    db_session.commit()
    _put_budget(client, master_key_header, "known")

    unknown = _service_key(client, master_key_header, "svc-a", end_user_budget_ids=["known", "no-such-budget"])
    tenant = _service_key(client, master_key_header, "svc-b", end_user_budget_ids=[tenant_budget.budget_id])

    assert unknown.status_code == 404
    assert tenant.status_code == 404


def test_updating_the_list_is_checked_against_the_stored_default(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    key_id, _ = _mlpa(client, master_key_header)

    refused = client.patch(
        f"{API_ROOT}/keys/{key_id}", json={"end_user_budget_ids": ["eu-memories"]}, headers=master_key_header
    )
    moved = client.patch(
        f"{API_ROOT}/keys/{key_id}",
        json={"end_user_budget_ids": ["eu-memories", "eu-memories"], "end_user_budget_id": "eu-memories"},
        headers=master_key_header,
    )
    cleared = client.patch(f"{API_ROOT}/keys/{key_id}", json={"end_user_budget_ids": None}, headers=master_key_header)

    assert refused.status_code == 400
    assert moved.status_code == 200, moved.text
    assert moved.json()["end_user_budget_ids"] == ["eu-memories"]
    assert cleared.json()["end_user_budget_ids"] == ["eu-memories"]


def test_deleting_a_budget_takes_it_off_every_list(client: TestClient, master_key_header: dict[str, str]) -> None:
    key_id, _ = _mlpa(client, master_key_header)

    deleted = client.delete(f"{API_ROOT}/budgets/eu-ai-dev", headers=master_key_header)
    fetched = client.get(f"{API_ROOT}/keys/{key_id}", headers=master_key_header).json()

    assert deleted.status_code == 204, deleted.text
    assert fetched["end_user_budget_ids"] == ["eu-ai", "eu-memories"]


def test_a_keys_default_end_user_budget_cannot_be_deleted(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Deleting it would leave the key's new end users uncapped, so the key's default has to move first."""
    key_id, headers = _mlpa(client, master_key_header)

    refused = client.delete(f"{API_ROOT}/budgets/eu-ai", headers=master_key_header)
    served = _chat(client, headers, "after-refusal")
    client.patch(f"{API_ROOT}/keys/{key_id}", json={"end_user_budget_id": "eu-memories"}, headers=master_key_header)
    deleted = client.delete(f"{API_ROOT}/budgets/eu-ai", headers=master_key_header)

    assert refused.status_code == 409, refused.text
    assert "mlpa" in refused.json()["detail"]
    assert served.headers[BUDGET_HEADER] == "eu-ai"
    assert deleted.status_code == 204, deleted.text


# --- the request names the budget -------------------------------------------


def test_a_request_starts_a_new_end_user_on_the_budget_it_names(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    key_id, headers = _mlpa(client, master_key_header)

    named = _chat(client, headers, "fxa1:memories", budget="eu-memories")
    unnamed = _chat(client, headers, "fxa1:ai")

    assert named.status_code == 200, named.text
    assert named.headers[BUDGET_HEADER] == "eu-memories"
    assert _end_user(client, master_key_header, key_id, "fxa1:memories").json()["budget_id"] == "eu-memories"
    assert unnamed.status_code == 200, unnamed.text
    assert unnamed.headers[BUDGET_HEADER] == "eu-ai"
    assert _end_user(client, master_key_header, key_id, "fxa1:ai").json()["budget_id"] == "eu-ai"


def test_an_existing_end_user_keeps_its_budget(client: TestClient, master_key_header: dict[str, str]) -> None:
    """A request cannot undo an operator's move, and the echoed header shows the budget actually applied."""
    _, headers = _mlpa(client, master_key_header)
    assert _chat(client, headers, "tester", budget="eu-ai-dev").status_code == 200

    again = _chat(client, headers, "tester", budget="eu-ai")

    assert again.status_code == 200
    assert again.headers[BUDGET_HEADER] == "eu-ai-dev"


def test_a_budget_off_the_list_is_refused_with_a_code(client: TestClient, master_key_header: dict[str, str]) -> None:
    key_id, headers = _mlpa(client, master_key_header)
    assert _chat(client, headers, "known").status_code == 200

    for user in ("new-user", "known"):
        refused = _chat(client, headers, user, budget="eu-other")
        assert refused.status_code == 403, user
        assert refused.headers["Otari-Error-Code"] == "end_user_budget_not_allowed"
        assert refused.json()["code"] == "end_user_budget_not_allowed"
    assert _end_user(client, master_key_header, key_id, "new-user").status_code == 404


def test_a_replay_repeats_the_budget_and_a_retry_naming_another_is_refused(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Stored responses are encrypted, so replaying one needs a key.
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _, headers = _mlpa(client, master_key_header)
    keyed = {**headers, "Idempotency-Key": "provision-once"}

    first = _chat(client, keyed, "fxa3:memories", budget="eu-memories")
    replayed = _chat(client, keyed, "fxa3:memories", budget="eu-memories")
    renamed = _chat(client, keyed, "fxa3:memories", budget="eu-ai")

    assert first.status_code == 200, first.text
    assert replayed.headers["Otari-Idempotent-Replayed"] == "true"
    assert replayed.headers[BUDGET_HEADER] == "eu-memories"
    assert renamed.status_code == 422, renamed.text


def test_each_end_user_is_held_to_the_budget_it_started_on(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    assert _put_budget(client, master_key_header, "tight", request_limit=1).status_code == 201
    assert _put_budget(client, master_key_header, "roomy", request_limit=5).status_code == 201
    created = _service_key(client, master_key_header, "svc", end_user_budget_ids=["tight", "roomy"])
    headers = {API_KEY_HEADER: f"Bearer {created.json()['key']}"}

    assert _chat(client, headers, "a:tight", budget="tight").status_code == 200
    assert _chat(client, headers, "a:tight", budget="tight").status_code == 403
    assert _chat(client, headers, "a:roomy", budget="roomy").status_code == 200
    assert _chat(client, headers, "a:roomy", budget="roomy").status_code == 200


def test_a_list_without_a_default_leaves_an_unnamed_end_user_unbudgeted(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _put_budget(client, master_key_header, "listed")
    created = _service_key(client, master_key_header, "svc", end_user_budget_ids=["listed"])
    headers = {API_KEY_HEADER: f"Bearer {created.json()['key']}"}

    unnamed = _chat(client, headers, "nobody")

    assert unnamed.status_code == 200
    assert BUDGET_HEADER not in unnamed.headers
    end_user = _end_user(client, master_key_header, created.json()["id"], "nobody").json()
    assert end_user["budget_id"] is None


# --- budgets with ids you choose ---------------------------------------------


def test_put_creates_a_budget_under_your_id_and_then_replaces_it(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    created = _put_budget(client, master_key_header, "end-user-budget-ai", max_budget=0.1, rpm_limit=40, tpm_limit=2000)
    replaced = client.put(f"{API_ROOT}/budgets/end-user-budget-ai", json={"max_budget": 0.2}, headers=master_key_header)
    fetched = client.get(f"{API_ROOT}/budgets/end-user-budget-ai", headers=master_key_header)

    assert created.status_code == 201, created.text
    assert created.json()["budget_id"] == "end-user-budget-ai"
    assert created.json()["rpm_limit"] == 40
    assert replaced.status_code == 200, replaced.text
    assert fetched.json()["max_budget"] == 0.2
    assert fetched.json()["rpm_limit"] is None
    assert fetched.json()["budget_duration_sec"] is None


def test_put_is_idempotent(client: TestClient, master_key_header: dict[str, str]) -> None:
    first = _put_budget(client, master_key_header, "same", request_limit=3)
    second = _put_budget(client, master_key_header, "same", request_limit=3)

    assert first.status_code == 201
    assert second.status_code == 200
    assert {k: v for k, v in first.json().items() if k != "updated_at"} == {
        k: v for k, v in second.json().items() if k != "updated_at"
    }


def test_put_refuses_an_id_that_is_not_a_slug(client: TestClient, master_key_header: dict[str, str]) -> None:
    for budget_id in ("-leading", "has space", "x" * 129):
        assert _put_budget(client, master_key_header, budget_id).status_code == 422, budget_id


def test_put_does_not_replace_an_organizations_budget(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    organization_id = db_session.query(col(Organization.id)).first()
    assert organization_id is not None
    db_session.add(Budget(budget_id="tenant-owned", request_limit=1, organization_id=organization_id[0]))
    db_session.commit()

    assert _put_budget(client, master_key_header, "tenant-owned").status_code == 409


# --- end users by your own id -------------------------------------------------


def test_put_places_an_end_user_on_a_budget_before_its_first_request(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    key_id, headers = _mlpa(client, master_key_header)
    url = f"{API_ROOT}/keys/{key_id}/end-users/tester:ai"

    created = client.put(url, json={"budget_id": "eu-ai-dev"}, headers=master_key_header)
    repeated = client.put(url, json={"budget_id": "eu-ai-dev"}, headers=master_key_header)
    served = _chat(client, headers, "tester:ai", budget="eu-ai")

    assert created.status_code == 201, created.text
    assert created.json()["external_id"] == "tester:ai"
    assert created.json()["owner_user_id"] == "mlpa"
    assert repeated.status_code == 200
    assert repeated.json()["budget_started_at"] == created.json()["budget_started_at"]
    assert served.headers[BUDGET_HEADER] == "eu-ai-dev"


def test_patch_blocks_and_moves_an_end_user(client: TestClient, master_key_header: dict[str, str]) -> None:
    key_id, headers = _mlpa(client, master_key_header)
    assert _chat(client, headers, "fxa2:ai").status_code == 200
    url = f"{API_ROOT}/keys/{key_id}/end-users/fxa2:ai"

    blocked = client.patch(url, json={"blocked": True}, headers=master_key_header)
    refused = _chat(client, headers, "fxa2:ai")
    moved = client.patch(url, json={"blocked": False, "budget_id": "eu-ai-dev"}, headers=master_key_header)
    off_list = client.patch(url, json={"budget_id": "eu-other"}, headers=master_key_header)

    assert blocked.status_code == 200, blocked.text
    assert blocked.json()["blocked"] is True
    assert refused.status_code == 403
    assert refused.headers["Otari-Error-Code"] == "user_blocked"
    assert moved.json()["blocked"] is False
    assert moved.json()["budget_id"] == "eu-ai-dev"
    assert off_list.status_code == 403
    assert off_list.json()["code"] == "end_user_budget_not_allowed"
    assert _chat(client, headers, "fxa2:ai").headers[BUDGET_HEADER] == "eu-ai-dev"


def test_end_users_are_scoped_to_the_keys_owner(client: TestClient, master_key_header: dict[str, str]) -> None:
    key_id, headers = _mlpa(client, master_key_header)
    assert _chat(client, headers, "alice").status_code == 200
    other = _service_key(client, master_key_header, "other-svc")
    plain = client.post(f"{API_ROOT}/keys", json={"user_id": "plain"}, headers=master_key_header)

    assert _end_user(client, master_key_header, key_id, "alice").status_code == 200
    assert _end_user(client, master_key_header, key_id, "nobody").status_code == 404
    assert _end_user(client, master_key_header, other.json()["id"], "alice").status_code == 404
    assert _end_user(client, master_key_header, plain.json()["id"], "alice").status_code == 400
    assert _end_user(client, master_key_header, "no-such-key", "alice").status_code == 404


def test_a_new_end_user_id_may_not_contain_a_slash(client: TestClient, master_key_header: dict[str, str]) -> None:
    key_id, headers = _mlpa(client, master_key_header)

    chat = _chat(client, headers, "org/alice")
    put = client.put(
        f"{API_ROOT}/keys/{key_id}/end-users/org/alice", json={"budget_id": "eu-ai"}, headers=master_key_header
    )

    assert chat.status_code == 400, chat.text
    assert put.status_code == 400, put.text
    assert _end_user(client, master_key_header, key_id, "org/alice").status_code == 404
    for dots in (".", ".."):
        assert _chat(client, headers, dots).status_code == 400, dots


def test_put_refuses_an_end_user_of_a_blocked_owner(client: TestClient, master_key_header: dict[str, str]) -> None:
    key_id, _ = _mlpa(client, master_key_header)
    assert client.patch(f"{API_ROOT}/users/mlpa", json={"blocked": True}, headers=master_key_header).status_code == 200

    put = client.put(
        f"{API_ROOT}/keys/{key_id}/end-users/early", json={"budget_id": "eu-ai"}, headers=master_key_header
    )

    assert put.status_code == 403, put.text
    assert _end_user(client, master_key_header, key_id, "early").status_code == 404


def test_an_end_user_created_with_a_slash_before_the_rule_still_works(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    key_id, headers = _mlpa(client, master_key_header)
    db_session.add(
        User(user_id="eu_legacy", alias="org/bob", parent_user_id="mlpa", external_id="org/bob", budget_id="eu-ai")
    )
    db_session.commit()

    served = _chat(client, headers, "org/bob")
    fetched = _end_user(client, master_key_header, key_id, "org/bob")
    blocked = client.patch(
        f"{API_ROOT}/keys/{key_id}/end-users/org/bob", json={"blocked": True}, headers=master_key_header
    )

    assert served.status_code == 200, served.text
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["user_id"] == "eu_legacy"
    assert blocked.json()["blocked"] is True


def test_put_replaces_a_budget_a_concurrent_put_created_first(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """Two provisioning runs racing on a new id both succeed, and the later body wins."""
    original = BudgetRepository.add_if_absent

    async def create_first(self: BudgetRepository, budget_id: str) -> bool:
        db_session.add(Budget(budget_id="raced", request_limit=1))
        db_session.commit()
        return await original(self, budget_id)

    with patch.object(BudgetRepository, "add_if_absent", create_first):
        response = _put_budget(client, master_key_header, "raced", request_limit=7)

    assert response.status_code == 200, response.text
    assert response.json()["request_limit"] == 7
