"""Moving the deployment providers stored through the dashboard onto hosted providers, over HTTP.

The deployment providers are stored as rows of the table the retired page
wrote, so the move reads what an operator actually stored. Model discovery is
stubbed, so no case dials a provider.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest
from any_llm.types.model import Model
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.providers import HostedProvider, ProviderCredential
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.secret_box import decrypt_secret, encrypt_secret, generate_secret_key

from .conftest import shared_test_config

PATH = f"{API_ROOT}/hosted-providers/migration"
DIAL = "gateway.services.providers._hosted_provider_service.test_provider_credentials"


@pytest.fixture(scope="module")
def test_config(postgres_url: str) -> GatewayConfig:
    """config.yml declares one provider, ``anthropic``, so a stored row can shadow it."""
    return shared_test_config(postgres_url, {"anthropic": {"api_key": "sk-config-7777"}})


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch, stored: Callable[..., None]) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


@pytest.fixture
def dials(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every dial, answered with one model unless its endpoint is a ``silent`` one."""
    calls: list[dict[str, Any]] = []

    async def _stub(impl_name: str, **kwargs: Any) -> ProviderDiscovery:
        calls.append({"provider": impl_name, **kwargs})
        if "silent" in str(kwargs.get("api_base")):
            return ProviderDiscovery(provider=impl_name, models=[], error="no listing", discovery_unsupported=True)
        return ProviderDiscovery(
            provider=impl_name, models=[Model(id="gpt-4o", object="model", created=0, owned_by=impl_name)]
        )

    monkeypatch.setattr(DIAL, _stub)
    return calls


@pytest.fixture
def stored(db_session_factory: Callable[[], Session]) -> Callable[..., None]:
    """Store a deployment provider as the retired Deployment providers page did."""

    def _store(instance: str, *, api_key: str | None = None, **fields: Any) -> None:
        with db_session_factory() as db:
            db.merge(
                ProviderCredential(
                    instance=instance,
                    encrypted_api_key=encrypt_secret(api_key) if api_key is not None else None,
                    last4=api_key[-4:] if api_key is not None else None,
                    client_args=fields.pop("client_args", None) or {},
                    updated_at=datetime.now(UTC),
                    **fields,
                )
            )
            db.commit()

    return _store


def _rows(response: Any) -> dict[str, dict[str, Any]]:
    assert response.status_code == 200, response.text
    return {row["instance"]: row for row in response.json()["rows"]}


def test_the_preview_plans_each_row_and_writes_nothing(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
    dials: list[dict[str, Any]],
    stored: Callable[..., None],
) -> None:
    stored("openai", api_key="sk-openai-1111")
    stored(
        "openai-eu",
        provider_type="openai",
        api_key="sk-eu-2222",
        api_base="https://eu.test/v1",
    )
    stored("my-vllm", provider_type="openai-compatible", api_base="http://vllm.test/v1")
    stored("groq", api_key="sk-groq-3333")
    taken = client.post(
        f"{API_ROOT}/hosted-providers",
        json={"provider": "groq", "api_key": "sk-hosted-4444"},
        headers=master_key_header,
    )
    assert taken.status_code == 201, taken.text
    dials.clear()

    rows = _rows(client.get(PATH, headers=master_key_header))

    assert {instance: row["outcome"] for instance, row in rows.items()} == {
        "groq": "conflict",
        "my-vllm": "migrate",
        "openai": "migrate",
        "openai-eu": "migrate",
    }
    assert rows["openai-eu"]["provider"] == "openai"
    assert rows["my-vllm"]["provider"] == "openai"
    assert "openai-compatible" in rows["my-vllm"]["notes"][0]
    assert "sk-" not in str(rows)
    assert dials == []
    assert len(db_session.execute(select(ProviderCredential)).scalars().all()) == 4
    assert len(db_session.execute(select(HostedProvider)).scalars().all()) == 1


def test_applying_moves_each_row_under_its_name_and_removes_it(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
    dials: list[dict[str, Any]],
    stored: Callable[..., None],
) -> None:
    stored("openai", api_key="sk-openai-1111", api_base="https://eu.test/v1")
    stored("vllm", api_base="http://vllm.test/v1", client_args={"timeout": 9})

    rows = _rows(client.post(PATH, headers=master_key_header))

    assert {instance: (row["outcome"], row["source_removed"], row["models"]) for instance, row in rows.items()} == {
        "openai": ("migrated", True, 1),
        "vllm": ("migrated", True, 1),
    }
    db_session.expire_all()
    assert db_session.execute(select(ProviderCredential)).scalars().all() == []
    hosted = {row.name: row for row in db_session.execute(select(HostedProvider)).scalars().all()}
    assert {name: (row.provider, row.api_base) for name, row in hosted.items()} == {
        "openai": ("openai", "https://eu.test/v1"),
        "vllm": ("vllm", "http://vllm.test/v1"),
    }
    assert decrypt_secret(hosted["openai"].encrypted_api_key or "") == "sk-openai-1111"
    assert hosted["vllm"].encrypted_api_key is None
    assert decrypt_secret(hosted["vllm"].encrypted_client_args or "") == '{"timeout": 9}'
    # The overlay forgets the moved rows at once, so dispatch stops serving them as deployment providers.
    listed = client.get(f"{API_ROOT}/providers", headers=master_key_header).json()["providers"]
    assert {row["instance"] for row in listed}.isdisjoint({"openai", "vllm"})


def test_a_renamed_provider_moves_under_its_name(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
    dials: list[dict[str, Any]],
    stored: Callable[..., None],
) -> None:
    """A request reaches a hosted provider by a name of its own, so the deployment provider goes."""
    stored(
        "openai-eu",
        provider_type="openai",
        api_key="sk-eu-2222",
        api_base="https://eu.test/v1",
    )

    [row] = _rows(client.post(PATH, headers=master_key_header)).values()

    assert (row["outcome"], row["source_removed"], row["models"]) == ("migrated", True, 1)
    db_session.expire_all()
    assert db_session.execute(select(ProviderCredential)).scalars().all() == []
    hosted = db_session.execute(select(HostedProvider)).scalar_one()
    assert (hosted.name, hosted.provider) == ("openai-eu", "openai")


def test_a_row_that_shadows_a_config_entry_says_the_entry_serves_again(
    client: TestClient, master_key_header: dict[str, str], dials: list[dict[str, Any]], stored: Callable[..., None]
) -> None:
    """Once the row goes, config.yml's entry of the same name answers ahead of the hosted provider."""
    stored("anthropic", api_key="sk-stored-6666")

    [row] = _rows(client.get(PATH, headers=master_key_header)).values()

    assert row["outcome"] == "migrate"
    assert any("config.yml declares a provider of the same name" in note for note in row["notes"])


def test_a_provider_that_lists_nothing_keeps_its_deployment_provider(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
    dials: list[dict[str, Any]],
    stored: Callable[..., None],
) -> None:
    """A hosted provider serves only its roster by default, so a row whose provider lists nothing stays for review."""
    stored("openai", api_key="sk-quiet-5555", api_base="https://silent.test")

    [row] = _rows(client.post(PATH, headers=master_key_header)).values()

    assert (row["outcome"], row["source_removed"], row["models"]) == ("migrated", False, 0)
    assert "listed no models" in row["notes"][-1]
    db_session.expire_all()
    assert [credential.instance for credential in db_session.execute(select(ProviderCredential)).scalars()] == [
        "openai"
    ]
    [hosted] = db_session.execute(select(HostedProvider)).scalars().all()
    assert (hosted.name, hosted.serves_unlisted_models) == ("openai", False)


def test_a_name_a_hosted_provider_cannot_take_says_how_to_move_it(
    client: TestClient, master_key_header: dict[str, str], dials: list[dict[str, Any]], stored: Callable[..., None]
) -> None:
    """An instance named after another implementation was legal here and is not for a hosted provider."""
    stored("groq", provider_type="openai", api_key="sk-groq-3333")

    [row] = _rows(client.get(PATH, headers=master_key_header)).values()

    assert row["outcome"] == "invalid"
    assert "names another provider" in row["detail"]
    assert "point its callers at that name" in row["detail"]


def test_a_row_no_configured_key_reads_is_left_where_it_is(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    dials: list[dict[str, Any]],
    stored: Callable[..., None],
) -> None:
    stored("openai", api_key="sk-openai-1111")
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    [row] = _rows(client.post(PATH, headers=master_key_header)).values()

    assert row["outcome"] == "unreadable"
    assert dials == []
    assert db_session.execute(select(HostedProvider)).scalars().all() == []


def test_a_row_stored_again_under_a_moved_name_is_a_conflict(
    client: TestClient, master_key_header: dict[str, str], dials: list[dict[str, Any]], stored: Callable[..., None]
) -> None:
    stored("openai", api_key="sk-quiet-5555", api_base="https://silent.test")
    client.post(PATH, headers=master_key_header)
    stored("openai", api_key="sk-again-6666")

    [row] = _rows(client.post(PATH, headers=master_key_header)).values()

    assert row["outcome"] == "conflict"


def test_a_row_the_reviewed_preview_did_not_show_is_left_where_it_is(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
    dials: list[dict[str, Any]],
    stored: Callable[..., None],
) -> None:
    """Another operator's row added after the preview loaded is not moved under this one's confirm."""
    stored("openai", api_key="sk-openai-1111")
    previewed = _rows(client.get(PATH, headers=master_key_header))
    reviewed = [{"instance": "openai", "updated_at": previewed["openai"]["updated_at"]}]

    # Another operator stores a deployment provider after this preview loaded.
    stored("groq", api_key="sk-groq-3333")

    rows = _rows(client.post(PATH, json={"rows": reviewed}, headers=master_key_header))

    assert (rows["openai"]["outcome"], rows["openai"]["source_removed"]) == ("migrated", True)
    assert rows["groq"]["outcome"] == "migrate"
    assert "not in the reviewed preview" in rows["groq"]["detail"]
    db_session.expire_all()
    assert [row.instance for row in db_session.execute(select(ProviderCredential)).scalars()] == ["groq"]


def test_a_row_edited_after_the_preview_is_left_where_it_is(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
    dials: list[dict[str, Any]],
    stored: Callable[..., None],
) -> None:
    """The operator reviewed one key and endpoint; a row since edited is not moved under that confirm."""
    stored("openai", api_key="sk-openai-1111")
    previewed = _rows(client.get(PATH, headers=master_key_header))
    reviewed = [{"instance": "openai", "updated_at": previewed["openai"]["updated_at"]}]
    stored("openai", api_key="sk-openai-9999")

    [row] = _rows(client.post(PATH, json={"rows": reviewed}, headers=master_key_header)).values()

    assert row["outcome"] == "migrate"
    assert "changed after the preview" in row["detail"]
    assert dials == []
    db_session.expire_all()
    assert db_session.execute(select(HostedProvider)).scalars().all() == []


def test_the_migration_is_operator_only(client: TestClient, api_key_header: dict[str, str]) -> None:
    response = client.post(PATH, headers=api_key_header)

    assert response.status_code in (401, 403, 404)


def test_a_row_the_move_cannot_take_can_be_discarded(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    db_session: Session,
    stored: Callable[..., None],
) -> None:
    stored("groq", provider_type="openai", api_key="sk-groq-3333")

    refused = client.delete(f"{PATH}/groq", headers=api_key_header)
    discarded = client.delete(f"{PATH}/groq", headers=master_key_header)
    again = client.delete(f"{PATH}/groq", headers=master_key_header)

    assert refused.status_code in (401, 403, 404)
    assert (discarded.status_code, again.status_code) == (204, 404)
    db_session.expire_all()
    assert db_session.execute(select(ProviderCredential)).scalars().all() == []
