"""Deleting a key removes only the spend ceilings that cap that key."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT
from gateway.models.api_keys import APIKey
from gateway.repositories.api_keys import ApiKeyRepository


def _key(client: TestClient, headers: dict[str, str], name: str) -> str:
    response = client.post(f"{API_ROOT}/keys", headers=headers, json={"key_name": name})
    assert response.status_code == status.HTTP_200_OK, response.text
    return str(response.json()["id"])


def _budget(client: TestClient, headers: dict[str, str]) -> str:
    response = client.post(f"{API_ROOT}/budgets", headers=headers, json={"max_budget": 10.0})
    assert response.status_code == status.HTTP_200_OK, response.text
    return str(response.json()["budget_id"])


def _ceiling(
    client: TestClient,
    headers: dict[str, str],
    key_id: str,
    budget_id: str,
    provider: str | None = None,
) -> str:
    response = client.post(
        f"{API_ROOT}/scoped-budgets",
        headers=headers,
        json={"scope_type": "api_token", "scope_id": key_id, "budget_id": budget_id, "provider_key_id": provider},
    )
    assert response.status_code == status.HTTP_200_OK, response.text
    return str(response.json()["id"])


def test_deleting_a_key_removes_all_its_ceilings_and_releases_its_budget(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    key_id = _key(client, master_key_header, "Revoked")
    survivor = _key(client, master_key_header, "Survivor")
    budget_id = _budget(client, master_key_header)
    for provider in (None, "openai", "gemini"):
        _ceiling(client, master_key_header, key_id, budget_id, provider)
    retained = _ceiling(client, master_key_header, survivor, _budget(client, master_key_header))

    deleted = client.delete(f"{API_ROOT}/keys/{key_id}", headers=master_key_header)
    assert deleted.status_code == status.HTTP_204_NO_CONTENT, deleted.text
    listed = client.get(f"{API_ROOT}/scoped-budgets", headers=master_key_header)
    assert listed.status_code == status.HTTP_200_OK, listed.text
    assert {ceiling["id"] for ceiling in listed.json()} == {retained}
    assert client.get(f"{API_ROOT}/keys/{survivor}", headers=master_key_header).status_code == status.HTTP_200_OK

    removed_budget = client.delete(f"{API_ROOT}/budgets/{budget_id}", headers=master_key_header)
    assert removed_budget.status_code == status.HTTP_204_NO_CONTENT, removed_budget.text


@pytest.mark.parametrize("failure", ["listener", "delete"])
def test_a_failed_key_deletion_restores_the_key_and_its_ceilings(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A failure after the sweep must roll back both domains, with a safe public error."""
    from gateway.repositories.budgets import ScopedBudgetRepository

    key_id = _key(client, master_key_header, "Keep on failure")
    ceiling_id = _ceiling(client, master_key_header, key_id, _budget(client, master_key_header))
    if failure == "listener":
        sweep = ScopedBudgetRepository.delete_for_api_key

        async def fail_after_sweep(self: ScopedBudgetRepository, key_id: str) -> None:
            await sweep(self, key_id)
            raise TimeoutError("diagnostic detail must not reach the caller")

        monkeypatch.setattr(ScopedBudgetRepository, "delete_for_api_key", fail_after_sweep)
    else:
        delete = ApiKeyRepository.delete

        async def fail_after_delete(self: ApiKeyRepository, key: APIKey) -> None:
            await delete(self, key)
            raise TimeoutError("diagnostic detail must not reach the caller")

        monkeypatch.setattr(ApiKeyRepository, "delete", fail_after_delete)

    deleted = client.delete(f"{API_ROOT}/keys/{key_id}", headers=master_key_header)
    assert deleted.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR, deleted.text
    assert deleted.json() == {"detail": "Database error"}
    assert client.get(f"{API_ROOT}/keys/{key_id}", headers=master_key_header).status_code == status.HTTP_200_OK
    ceilings = client.get(f"{API_ROOT}/scoped-budgets", headers=master_key_header)
    assert {ceiling["id"] for ceiling in ceilings.json()} == {ceiling_id}


def test_deleting_a_key_without_ceilings_and_repeating_the_delete(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    key_id = _key(client, master_key_header, "No ceilings")
    deleted = client.delete(f"{API_ROOT}/keys/{key_id}", headers=master_key_header)
    assert deleted.status_code == status.HTTP_204_NO_CONTENT
    repeated = client.delete(f"{API_ROOT}/keys/{key_id}", headers=master_key_header)
    assert repeated.status_code == status.HTTP_404_NOT_FOUND
    assert repeated.json() == {"detail": f"API key with id '{key_id}' not found"}


@pytest.mark.parametrize("provider", [None, "openai"])
def test_duplicate_api_key_ceilings_keep_the_existing_conflict_response(
    client: TestClient, master_key_header: dict[str, str], provider: str | None
) -> None:
    key_id = _key(client, master_key_header, "Duplicate ceiling")
    budget_id = _budget(client, master_key_header)
    _ceiling(client, master_key_header, key_id, budget_id, provider)
    repeated = client.post(
        f"{API_ROOT}/scoped-budgets",
        headers=master_key_header,
        json={"scope_type": "api_token", "scope_id": key_id, "budget_id": budget_id, "provider_key_id": provider},
    )
    assert repeated.status_code == status.HTTP_409_CONFLICT
    assert repeated.json() == {"detail": "A budget already exists for this scope and provider"}


def test_api_key_ceilings_refuse_a_missing_key_or_budget(client: TestClient, master_key_header: dict[str, str]) -> None:
    budget_id = _budget(client, master_key_header)
    missing_key = client.post(
        f"{API_ROOT}/scoped-budgets",
        headers=master_key_header,
        json={"scope_type": "api_token", "scope_id": "missing", "budget_id": budget_id},
    )
    assert missing_key.status_code == status.HTTP_404_NOT_FOUND
    assert missing_key.json() == {"detail": "API key 'missing' not found"}
    key_id = _key(client, master_key_header, "Missing budget")
    missing_budget = client.post(
        f"{API_ROOT}/scoped-budgets",
        headers=master_key_header,
        json={"scope_type": "api_token", "scope_id": key_id, "budget_id": "missing"},
    )
    assert missing_budget.status_code == status.HTTP_404_NOT_FOUND
    assert missing_budget.json() == {"detail": "Budget 'missing' not found"}
