"""Named hosted providers, keyless ones, and the test and re-encrypt routes, over HTTP.

A hosted provider's name is what the roster, the price list and every route key
on; its ``provider`` is the implementation it is dialed as. Model discovery and
the community dataset are stubbed, so no case dials a provider.
"""

from collections.abc import Iterator
from decimal import Decimal
from typing import Any

import pytest
from any_llm.types.model import Model
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT
from gateway.models.pricing import ModelPricing
from gateway.models.providers import HostedProvider, HostedProviderModel
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.pricing_service import configure_default_pricing
from gateway.services.secret_box import decrypt_secret, generate_secret_key

PATH = f"{API_ROOT}/hosted-providers"
KEY = "sk-live-hosted-1234"


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


@pytest.fixture
def dials(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every dial, answered with one model; the calls are returned for inspection."""
    calls: list[dict[str, Any]] = []

    async def _stub(impl_name: str, **kwargs: Any) -> ProviderDiscovery:
        calls.append({"provider": impl_name, **kwargs})
        return ProviderDiscovery(
            provider=impl_name, models=[Model(id="gpt-4o", object="model", created=0, owned_by=impl_name)]
        )

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _stub)
    return calls


@pytest.fixture
def looked_up(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[tuple[str | None, str | None]]]:
    """The community dataset, switched on and pricing gpt-4o under whatever it is asked.

    Yields each ``(provider, implementation)`` asked. Switched on after the
    client, whose start applies the configuration's own setting.
    """
    asked: list[tuple[str | None, str | None]] = []

    def _stub(
        provider: str | None, model: str, as_of: object, *, implementation: str | None = None
    ) -> ModelPricing | None:
        asked.append((provider, implementation))
        return ModelPricing(
            model_key=f"{provider}:{model}",
            input_price_per_million=Decimal("2.5"),
            output_price_per_million=Decimal("10"),
            pricing_tiers=[],
            unit="tokens",
        )

    monkeypatch.setattr("gateway.services.pricing_service.default_model_pricing", _stub)
    configure_default_pricing(True)
    yield asked
    configure_default_pricing(False)


def _create(client: TestClient, headers: dict[str, str], **body: Any) -> dict[str, Any]:
    response = client.post(PATH, json=body, headers=headers)
    assert response.status_code == 201, response.text
    created: dict[str, Any] = response.json()
    return created


def test_a_named_provider_is_dialed_as_its_implementation_and_keyed_by_its_name(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
    dials: list[dict[str, Any]],
    looked_up: list[tuple[str | None, str | None]],
) -> None:
    created = _create(
        client, master_key_header, provider="openai", name="openai-eu", api_key=KEY, api_base="https://eu.test/v1"
    )

    assert (created["name"], created["provider"]) == ("openai-eu", "openai")
    assert dials[0]["provider"] == "openai"
    assert dials[0]["api_base"] == "https://eu.test/v1"
    assert looked_up == [("openai-eu", "openai")]
    roster = db_session.execute(select(HostedProviderModel)).scalars().all()
    assert [(row.provider, row.model, row.enabled) for row in roster] == [("openai-eu", "gpt-4o", True)]
    priced = db_session.execute(select(ModelPricing.model_key)).scalars().all()
    assert "openai-eu:gpt-4o" in priced
    models = client.get(f"{PATH}/openai-eu/models", headers=master_key_header)
    assert models.status_code == 200, models.text
    assert [model["model"] for model in models.json()["data"]] == ["gpt-4o"]


def test_two_endpoints_of_one_implementation_coexist_under_their_names(
    client: TestClient, master_key_header: dict[str, str], dials: list[dict[str, Any]]
) -> None:
    _create(client, master_key_header, provider="openai", api_key=KEY)
    _create(client, master_key_header, provider="openai", name="openai-eu", api_key=KEY, api_base="https://eu.test/v1")

    listed = client.get(PATH, headers=master_key_header).json()["data"]
    assert [(row["name"], row["provider"], row["api_base"]) for row in listed] == [
        ("openai", "openai", None),
        ("openai-eu", "openai", "https://eu.test/v1"),
    ]
    again = client.post(
        PATH, json={"provider": "openai", "name": "openai-eu", "api_key": KEY}, headers=master_key_header
    )
    assert again.status_code == 409
    assert "openai-eu" in again.json()["detail"]


@pytest.mark.parametrize(
    ("name", "problem"),
    [("anthropic", "names another provider"), ("hosted", "is reserved"), ("eu:openai", "must not contain")],
)
def test_a_name_a_selector_could_not_address_is_refused(
    client: TestClient, master_key_header: dict[str, str], dials: list[dict[str, Any]], name: str, problem: str
) -> None:
    response = client.post(PATH, json={"provider": "openai", "name": name, "api_key": KEY}, headers=master_key_header)

    assert response.status_code == 400, response.text
    assert problem in response.json()["detail"]


@pytest.mark.parametrize("name", [None, "otari"])
def test_an_otari_provider_cannot_take_the_reserved_name(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
    dials: list[dict[str, Any]],
    name: str | None,
) -> None:
    """``otari`` prefixes the gateway's tool prices, so a provider under it would lose its own."""
    body: dict[str, Any] = {"provider": "otari", "api_base": "http://otari.test/v1"}
    if name is not None:
        body["name"] = name

    response = client.post(PATH, json=body, headers=master_key_header)

    assert response.status_code == 400, response.text
    assert "'otari' is reserved" in response.json()["detail"]
    assert dials == []
    assert db_session.execute(select(HostedProvider)).scalars().all() == []
    named = _create(client, master_key_header, provider="otari", name="upstream", api_base="http://otari.test/v1")
    assert (named["name"], named["provider"]) == ("upstream", "otari")


def test_an_endpoint_of_its_own_may_be_stored_without_a_key(
    client: TestClient, master_key_header: dict[str, str], db_session: Session, dials: list[dict[str, Any]]
) -> None:
    created = _create(client, master_key_header, provider="openai", name="my-vllm", api_base="http://vllm.test/v1")

    assert created["has_api_key"] is False
    assert created["api_key_last4"] is None
    row = db_session.execute(select(HostedProvider).where(HostedProvider.name == "my-vllm")).scalar_one()
    assert row.encrypted_api_key is None
    assert dials[0]["api_key"] is None


def test_a_keyed_provider_at_its_default_endpoint_needs_a_key(
    client: TestClient, master_key_header: dict[str, str], dials: list[dict[str, Any]]
) -> None:
    response = client.post(PATH, json={"provider": "openai"}, headers=master_key_header)

    assert response.status_code == 400, response.text
    assert "needs an API key" in response.json()["detail"]
    assert dials == []


def test_a_keyless_backend_needs_no_endpoint_either(
    client: TestClient, master_key_header: dict[str, str], dials: list[dict[str, Any]]
) -> None:
    created = _create(client, master_key_header, provider="ollama")

    assert created["has_api_key"] is False


def test_a_key_is_removed_only_where_the_provider_can_go_without(
    client: TestClient, master_key_header: dict[str, str], dials: list[dict[str, Any]]
) -> None:
    _create(client, master_key_header, provider="openai", api_key=KEY)
    _create(client, master_key_header, provider="openai", name="proxy", api_key=KEY, api_base="http://proxy.test/v1")

    refused = client.patch(f"{PATH}/openai", json={"api_key": None}, headers=master_key_header)
    removed = client.patch(f"{PATH}/proxy", json={"api_key": None}, headers=master_key_header)

    assert refused.status_code == 400, refused.text
    assert removed.status_code == 200, removed.text
    assert removed.json()["has_api_key"] is False


def test_a_provider_no_configured_key_reads_is_listed_as_undecryptable(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch, dials: list[dict[str, Any]]
) -> None:
    _create(client, master_key_header, provider="openai", api_key=KEY)

    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    [listed] = client.get(PATH, headers=master_key_header).json()["data"]

    assert listed["decryptable"] is False


def test_unsaved_credentials_are_tested_as_their_implementation(
    client: TestClient, master_key_header: dict[str, str], dials: list[dict[str, Any]]
) -> None:
    response = client.post(
        f"{PATH}/test",
        json={"provider": "openai", "name": "openai-eu", "api_key": KEY, "api_base": "https://eu.test/v1"},
        headers=master_key_header,
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"ok": True, "model_count": 1, "error": None, "discovery_unsupported": False}
    assert dials[0]["provider"] == "openai"
    assert dials[0]["api_key"] == KEY
    assert KEY not in response.text


def test_an_edit_is_tested_on_the_stored_key_without_typing_it(
    client: TestClient, master_key_header: dict[str, str], dials: list[dict[str, Any]]
) -> None:
    _create(client, master_key_header, provider="openai", name="openai-eu", api_key=KEY, api_base="https://eu.test/v1")
    dials.clear()

    response = client.post(f"{PATH}/test", json={"provider": "openai", "name": "openai-eu"}, headers=master_key_header)

    assert response.status_code == 200, response.text
    assert dials[0]["api_key"] == KEY
    assert dials[0]["api_base"] == "https://eu.test/v1"


def test_the_stored_key_is_not_tested_against_another_origin(
    client: TestClient, master_key_header: dict[str, str], dials: list[dict[str, Any]]
) -> None:
    _create(client, master_key_header, provider="openai", name="openai-eu", api_key=KEY, api_base="https://eu.test/v1")
    dials.clear()

    response = client.post(
        f"{PATH}/test",
        json={"provider": "openai", "name": "openai-eu", "api_base": "https://attacker.test/v1"},
        headers=master_key_header,
    )

    assert response.status_code == 400, response.text
    assert dials == []


def test_reencrypt_moves_keys_and_extras_onto_the_primary_key(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    dials: list[dict[str, Any]],
) -> None:
    old_key, new_key = generate_secret_key(), generate_secret_key()
    monkeypatch.setenv("OTARI_SECRET_KEY", old_key)
    _create(client, master_key_header, provider="bedrock", api_key=KEY, client_args={"region_name": "eu-west-1"})
    _create(client, master_key_header, provider="ollama")

    monkeypatch.setenv("OTARI_SECRET_KEY", f"{new_key},{old_key}")
    response = client.post(f"{PATH}/reencrypt", headers=master_key_header)

    assert response.status_code == 200, response.text
    assert response.json() == {"reencrypted": 1, "unreadable": 0, "skipped": 0}
    monkeypatch.setenv("OTARI_SECRET_KEY", new_key)
    db_session.expire_all()
    row = db_session.execute(select(HostedProvider).where(HostedProvider.name == "bedrock")).scalar_one()
    assert decrypt_secret(row.encrypted_api_key or "") == KEY
    assert decrypt_secret(row.encrypted_client_args or "") == '{"region_name": "eu-west-1"}'


def test_reencrypt_counts_what_no_configured_key_reads(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch, dials: list[dict[str, Any]]
) -> None:
    _create(client, master_key_header, provider="openai", api_key=KEY)

    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    response = client.post(f"{PATH}/reencrypt", headers=master_key_header)

    assert response.status_code == 200, response.text
    assert response.json() == {"reencrypted": 0, "unreadable": 1, "skipped": 0}
