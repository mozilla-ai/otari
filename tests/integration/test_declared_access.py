"""Budgets and API keys declared in config.yml, applied at every start."""

import asyncio
import secrets
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from gateway.adapters.api_key_format_adapter import DefaultApiKeyFormatAdapter
from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.settings.api_keys import ApiKeyConfig
from gateway.core.settings.budgets import BudgetConfig
from gateway.exceptions.budget_exceptions import DeclaredAccessError
from gateway.main import apply_declared_access
from gateway.models.api_keys import APIKey
from gateway.models.budgets import SCOPE_API_TOKEN, Budget, ScopedBudget

from .conftest import _to_async_url, build_test_client

MASTER = {API_KEY_HEADER: "Bearer test-master-key"}


def _secret() -> str:
    return "tk-" + secrets.token_urlsafe(48)


SECRET = _secret()
OTHER_SECRET = _secret()

BUDGETS = {
    "end-user-budget-ai": BudgetConfig(max_budget=0.1, reset_alignment="calendar_day", rpm_limit=40, tpm_limit=2000),
    "end-user-budget-memories": BudgetConfig(max_budget=0.05, reset_alignment="calendar_day"),
    "mlpa-global": BudgetConfig(max_budget=100, reset_alignment="calendar_day"),
    "mlpa-global-2": BudgetConfig(max_budget=200, reset_alignment="calendar_month"),
}


def _key(**overrides: Any) -> ApiKeyConfig:
    fields: dict[str, Any] = {
        "secret": SecretStr(SECRET),
        "user_id": "mlpa",
        "is_service_key": True,
        "end_user_budget_ids": ["end-user-budget-ai"],
        "end_user_budget_id": "end-user-budget-ai",
        "ceiling": "mlpa-global",
    }
    fields.update(overrides)
    return ApiKeyConfig(**fields)


def _config(base: GatewayConfig, *, budgets: dict[str, BudgetConfig] | None = None, **keys: ApiKeyConfig) -> GatewayConfig:
    return base.model_copy(update={"budgets": BUDGETS if budgets is None else budgets, "api_keys": keys})


@contextmanager
def _boot(config: GatewayConfig) -> Iterator[TestClient]:
    """Start a gateway on ``config``, and stop it on the way out, as a restart would."""
    started: Generator[TestClient] = build_test_client(config)
    try:
        yield next(started)
    finally:
        started.close()


def _authenticates(client: TestClient, secret: str) -> bool:
    response = client.get(f"{API_ROOT}/models", headers={API_KEY_HEADER: f"Bearer {secret}"})
    assert response.status_code in (200, 401), response.text
    return response.status_code == 200


def _declared_keys(client: TestClient) -> list[dict[str, Any]]:
    response = client.get(f"{API_ROOT}/keys", headers=MASTER)
    assert response.status_code == 200, response.text
    return [key for key in response.json() if key["config_name"] is not None]


def _key_ceilings(client: TestClient, key_id: str) -> list[dict[str, Any]]:
    response = client.get(
        f"{API_ROOT}/scoped-budgets", params={"scope_type": SCOPE_API_TOKEN, "scope_id": key_id}, headers=MASTER
    )
    assert response.status_code == 200, response.text
    return list(response.json())


def test_declared_budgets_and_key_are_created_and_the_secret_authenticates(test_config: GatewayConfig) -> None:
    with _boot(_config(test_config, mlpa=_key())) as client:
        budget = client.get(f"{API_ROOT}/budgets/end-user-budget-ai", headers=MASTER).json()
        assert budget["max_budget"] == 0.1
        assert budget["reset_alignment"] == "calendar_day"
        assert budget["rpm_limit"] == 40
        assert budget["tpm_limit"] == 2000
        assert budget["origin"] == "config"

        (key,) = _declared_keys(client)
        assert key["config_name"] == "mlpa"
        assert key["key_name"] == "mlpa"
        assert key["user_id"] == "mlpa"
        assert key["is_service_key"] is True
        assert key["end_user_budget_id"] == "end-user-budget-ai"
        assert key["end_user_budget_ids"] == ["end-user-budget-ai"]
        assert key["key_prefix"] == SECRET[:10]
        assert key["key_suffix"] == SECRET[-4:]
        # The secret is never echoed back, in the key or anywhere else in the listing.
        assert SECRET not in client.get(f"{API_ROOT}/keys", headers=MASTER).text

        (ceiling,) = _key_ceilings(client, key["id"])
        assert ceiling["budget_id"] == "mlpa-global"

        assert _authenticates(client, SECRET)
        assert not _authenticates(client, OTHER_SECRET)


def test_a_restart_applies_the_same_config_without_a_second_row(test_config: GatewayConfig) -> None:
    config = _config(test_config, mlpa=_key())
    with _boot(config) as client:
        (first,) = _declared_keys(client)
        (first_ceiling,) = _key_ceilings(client, first["id"])
        # An edit made through the API, which config.yml overrides at the next start.
        assert client.patch(f"{API_ROOT}/budgets/mlpa-global", json={"max_budget": 5}, headers=MASTER).status_code == 200
        assert client.patch(f"{API_ROOT}/keys/{first['id']}", json={"is_service_key": False}, headers=MASTER).status_code == 200

    with _boot(config) as client:
        (second,) = _declared_keys(client)
        assert second["id"] == first["id"]
        assert second["is_service_key"] is True
        (second_ceiling,) = _key_ceilings(client, second["id"])
        assert second_ceiling["id"] == first_ceiling["id"]
        assert client.get(f"{API_ROOT}/budgets/mlpa-global", headers=MASTER).json()["max_budget"] == 100
        assert _authenticates(client, SECRET)


def test_changing_the_secret_in_config_changes_the_keys_secret(test_config: GatewayConfig) -> None:
    with _boot(_config(test_config, mlpa=_key())) as client:
        (before,) = _declared_keys(client)

    with _boot(_config(test_config, mlpa=_key(secret=SecretStr(OTHER_SECRET)))) as client:
        (after,) = _declared_keys(client)
        assert after["id"] == before["id"]
        assert after["key_suffix"] == OTHER_SECRET[-4:]
        assert _authenticates(client, OTHER_SECRET)
        assert not _authenticates(client, SECRET)


def test_changing_the_end_user_budgets_in_config_is_applied_at_the_next_start(test_config: GatewayConfig) -> None:
    with _boot(_config(test_config, mlpa=_key())):
        pass

    changed = _key(
        end_user_budget_ids=["end-user-budget-ai", "end-user-budget-memories"],
        end_user_budget_id="end-user-budget-memories",
    )
    with _boot(_config(test_config, mlpa=changed)) as client:
        (key,) = _declared_keys(client)
        assert key["end_user_budget_ids"] == ["end-user-budget-ai", "end-user-budget-memories"]
        assert key["end_user_budget_id"] == "end-user-budget-memories"


def test_changing_the_ceiling_in_config_moves_the_one_ceiling(test_config: GatewayConfig) -> None:
    with _boot(_config(test_config, mlpa=_key(ceiling=None))) as client:
        (key,) = _declared_keys(client)
        assert _key_ceilings(client, key["id"]) == []

    with _boot(_config(test_config, mlpa=_key())) as client:
        (attached,) = _key_ceilings(client, key["id"])
        assert attached["budget_id"] == "mlpa-global"

    with _boot(_config(test_config, mlpa=_key(ceiling="mlpa-global-2"))) as client:
        (moved,) = _key_ceilings(client, key["id"])
        assert moved["id"] == attached["id"]
        assert moved["budget_id"] == "mlpa-global-2"

    # Leaving the ceiling out does not remove it.
    with _boot(_config(test_config, mlpa=_key(ceiling=None))) as client:
        (kept,) = _key_ceilings(client, key["id"])
        assert kept["id"] == attached["id"]


def test_a_key_minted_through_the_api_is_taken_over_by_its_secret(test_config: GatewayConfig) -> None:
    with _boot(test_config) as client:
        minted = client.post(f"{API_ROOT}/keys", json={"key_name": "minted", "user_id": "mlpa"}, headers=MASTER).json()

    with _boot(_config(test_config, mlpa=_key(secret=SecretStr(minted["key"])))) as client:
        (key,) = _declared_keys(client)
        assert key["id"] == minted["id"]
        assert key["is_service_key"] is True
        assert _authenticates(client, minted["key"])


def test_a_declaration_removed_from_config_is_kept_and_no_longer_marked(test_config: GatewayConfig) -> None:
    with _boot(_config(test_config, mlpa=_key())) as client:
        (key,) = _declared_keys(client)

    with _boot(test_config) as client:
        assert _declared_keys(client) == []
        kept = client.get(f"{API_ROOT}/keys/{key['id']}", headers=MASTER).json()
        assert kept["config_name"] is None
        assert kept["is_service_key"] is True
        budget = client.get(f"{API_ROOT}/budgets/mlpa-global", headers=MASTER).json()
        assert budget["origin"] is None
        assert _authenticates(client, SECRET)


def test_a_key_naming_a_missing_budget_stops_the_start_and_writes_nothing(test_config: GatewayConfig) -> None:
    config = _config(test_config, mlpa=_key(ceiling="no-such-budget"))

    with pytest.raises(DeclaredAccessError, match="api_keys.mlpa"), _boot(config):
        pass

    with _boot(test_config) as client:
        for budget_id in BUDGETS:
            assert client.get(f"{API_ROOT}/budgets/{budget_id}", headers=MASTER).status_code == 404
        assert _declared_keys(client) == []
        assert not _authenticates(client, SECRET)


@pytest.mark.asyncio
async def test_replicas_starting_at_once_create_one_of_each(test_config: GatewayConfig, postgres_url: str) -> None:
    """Every replica applies the declarations; the database holds one key, one ceiling and one row per budget."""
    config = _config(test_config, mlpa=_key())
    engine = create_async_engine(_to_async_url(postgres_url), pool_size=4)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def replica() -> None:
        async with sessions() as session:
            await apply_declared_access(config, session, DefaultApiKeyFormatAdapter(session))

    try:
        await asyncio.gather(*(replica() for _ in range(4)))
        async with sessions() as session:
            keys = (await session.execute(select(APIKey).where(APIKey.config_name == "mlpa"))).scalars().all()
            assert len(keys) == 1
            ceilings = (
                await session.execute(
                    select(func.count())
                    .select_from(ScopedBudget)
                    .where(ScopedBudget.scope_type == SCOPE_API_TOKEN, ScopedBudget.scope_id == keys[0].id)
                )
            ).scalar_one()
            assert ceilings == 1
            budgets = (await session.execute(select(func.count()).select_from(Budget))).scalar_one()
            assert budgets == len(BUDGETS)
    finally:
        await engine.dispose()
