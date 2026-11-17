"""A hosted provider's models grouped by vendor, a vendor switched off, and prices put back on the default.

A provider that routes to others serves models from many vendors. These tests
pin that the list names each model's vendor and counts each vendor across the
whole provider, that a vendor switched off withholds every one of its models at
dispatch and in the catalog while each model keeps its own switch, that a
vendor's models are switched together, and that an operator's price can be put
back on the default rate. Model discovery is stubbed; the provider call is
captured at the any-llm boundary and never made.
"""

from collections.abc import Iterator, Mapping
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.model import Model
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.pricing import SEED_ORIGIN, ModelPricing
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.providers._hosted_offers import HostedOffers
from gateway.services.secret_box import generate_secret_key

from .conftest import shared_test_config

PATH = f"{API_ROOT}/hosted-providers/together"
KEY = "sk-live-hosted-1234"
QWEN = "Qwen/Qwen3.5-9B"
CODER = "Qwen/Qwen3-Coder-480B"
DEEPSEEK = "deepseek-ai/DeepSeek-V3.2"
FLUX = "black-forest-labs/FLUX.1-schnell"
LISTED = (QWEN, CODER, DEEPSEEK, FLUX)


@pytest.fixture(scope="module")
def test_config(postgres_url: str) -> GatewayConfig:
    """The shared configuration declaring no provider, so nothing configured shadows the hosted one."""
    return shared_test_config(postgres_url, {})


class _MockCompletionError(Exception):
    """Raised to short-circuit the mocked provider call after capturing its kwargs."""


@pytest.fixture(autouse=True)
def _keys(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No ambient provider key, which would credential the ladder ahead of the port, and a real secret box."""
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


@pytest.fixture(autouse=True)
def _discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name,
            models=[Model(id=name, object="model", created=0, owned_by=impl_name) for name in LISTED],
        )

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _stub)


def _price(client: TestClient, headers: dict[str, str], model: str, input_price: float, output_price: float) -> None:
    response = client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": f"together:{model}",
            "input_price_per_million": input_price,
            "output_price_per_million": output_price,
        },
        headers=headers,
    )
    assert response.status_code == 200, response.text


@pytest.fixture
def together(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Together offering two Alibaba models, a DeepSeek one and an unpriced one, the priced ones switched on."""
    assert (
        client.post(f"{API_ROOT}/users", json={"user_id": "u1", "alias": "u1"}, headers=master_key_header).status_code
        == 200
    )
    _price(client, master_key_header, QWEN, 0.1, 0.1)
    _price(client, master_key_header, CODER, 2.0, 2.0)
    _price(client, master_key_header, DEEPSEEK, 0.6, 1.7)
    response = client.post(
        f"{API_ROOT}/hosted-providers", json={"provider": "together", "api_key": KEY}, headers=master_key_header
    )
    assert response.status_code == 201, response.text


def _models(client: TestClient, headers: dict[str, str], **params: Any) -> dict[str, Any]:
    response = client.get(f"{PATH}/models", params=params, headers=headers)
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def _vendors(body: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {facet["vendor"]: facet for facet in body["vendors"]}


def _set_vendor(client: TestClient, headers: dict[str, str], vendor: str, *, enabled: bool) -> dict[str, Any]:
    response = client.patch(f"{PATH}/vendors", json={"vendor": vendor, "enabled": enabled}, headers=headers)
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def _dispatched(client: TestClient, headers: dict[str, str], model: str) -> bool:
    """Whether a completion for ``together:<model>`` reached the provider on the hosted key."""
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> None:
        captured.update(kwargs)
        raise _MockCompletionError

    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": f"together:{model}", "messages": [{"role": "user", "content": "Hi"}], "user": "u1"},
            headers=headers,
        )
    return captured.get("api_key") == KEY


@pytest.mark.usefixtures("together")
def test_each_model_names_its_vendor_and_each_vendor_is_counted(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    body = _models(client, master_key_header)

    assert {row["model"]: row["vendor"] for row in body["data"]} == {
        QWEN: "Alibaba",
        CODER: "Alibaba",
        DEEPSEEK: "DeepSeek",
        FLUX: "black-forest-labs",
    }
    vendors = _vendors(body)
    assert [facet["vendor"] for facet in body["vendors"]] == ["Alibaba", "black-forest-labs", "DeepSeek"]
    assert vendors["Alibaba"] | {"vendor": None} == {
        "vendor": None,
        "enabled": True,
        "total": 2,
        "serving": 2,
        "unpriced": 0,
        "unavailable": 0,
        "custom_differs": 0,
    }
    assert vendors["black-forest-labs"]["unpriced"] == 1
    assert vendors["black-forest-labs"]["serving"] == 0


@pytest.mark.usefixtures("together")
def test_the_list_narrows_to_a_vendor_and_a_search_matches_the_vendor(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    by_vendor = _models(client, master_key_header, vendor="Alibaba")
    by_search = _models(client, master_key_header, search="alibaba")

    assert sorted(row["model"] for row in by_vendor["data"]) == [CODER, QWEN]
    assert sorted(row["model"] for row in by_search["data"]) == [CODER, QWEN]
    # The vendor counts still cover the whole provider.
    assert len(by_vendor["vendors"]) == 3


@pytest.mark.usefixtures("together")
def test_a_vendor_switched_off_withholds_its_models_and_keeps_their_own_switches(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    facet = _set_vendor(client, master_key_header, "Alibaba", enabled=False)

    assert facet["enabled"] is False
    assert facet["serving"] == 0
    body = _models(client, master_key_header)
    assert {row["model"]: row["enabled"] for row in body["data"] if row["vendor"] == "Alibaba"} == {
        QWEN: True,
        CODER: True,
    }
    served = _models(client, master_key_header, enabled="true")
    assert [row["model"] for row in served["data"]] == [DEEPSEEK]
    [listed] = client.get(f"{API_ROOT}/hosted-providers", headers=master_key_header).json()["data"]
    assert listed["models"] == {"total": 4, "serving": 1, "unpriced": 1, "vendors_off": 1}

    _set_vendor(client, master_key_header, "Alibaba", enabled=True)

    assert _vendors(_models(client, master_key_header))["Alibaba"]["serving"] == 2


@pytest.mark.usefixtures("together")
def test_a_vendor_switched_off_is_refused_at_dispatch_and_left_out_of_the_catalog(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    assert _dispatched(client, master_key_header, QWEN)

    _set_vendor(client, master_key_header, "Alibaba", enabled=False)

    assert not _dispatched(client, master_key_header, QWEN)
    assert _dispatched(client, master_key_header, DEEPSEEK)
    ids = {model["id"] for model in client.get(f"{API_ROOT}/models", headers=master_key_header).json()["data"]}
    assert not any("qwen" in model_id.lower() for model_id in ids)
    assert any("deepseek-v3.2" in model_id.lower() for model_id in ids)


@pytest.mark.usefixtures("together")
def test_a_vendor_the_provider_offers_nothing_by_is_a_404(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    switch = client.patch(f"{PATH}/vendors", json={"vendor": "Nobody", "enabled": False}, headers=master_key_header)
    models = client.post(
        f"{PATH}/models/switch", json={"vendor": "Nobody", "enabled": False}, headers=master_key_header
    )

    assert switch.status_code == 404
    assert models.status_code == 404


@pytest.mark.usefixtures("together")
def test_a_vendors_models_are_switched_together_and_an_unpriced_one_stays_off(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    withheld = client.post(
        f"{PATH}/models/switch", json={"vendor": "Alibaba", "enabled": False}, headers=master_key_header
    )
    served = client.post(
        f"{PATH}/models/switch", json={"vendor": "black-forest-labs", "enabled": True}, headers=master_key_header
    )

    assert withheld.json() == {"changed": 2, "skipped_unpriced": 0}
    assert served.json() == {"changed": 0, "skipped_unpriced": 1}
    rows = {row["model"]: row for row in _models(client, master_key_header)["data"]}
    assert rows[QWEN]["enabled"] is False
    assert rows[QWEN]["disabled_reason"] == "operator"
    assert rows[FLUX]["enabled"] is False

    again = client.post(f"{PATH}/models/switch", json={"vendor": "Alibaba", "enabled": True}, headers=master_key_header)

    assert again.json() == {"changed": 2, "skipped_unpriced": 0}


@pytest.fixture
def defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default rung prices Qwen differently from the operator, the Coder the same, and DeepSeek not at all."""
    table = {QWEN: (0.15, 0.2), CODER: (2.0, 2.0)}

    async def _defaults_for(
        self: HostedOffers, provider: str, models: Any, *, implementation: str | None = None
    ) -> dict[str, ModelPricing]:
        return {
            model: ModelPricing(
                model_key=f"{provider}:{model}",
                input_price_per_million=table[model][0],
                output_price_per_million=table[model][1],
                origin=SEED_ORIGIN,
            )
            for model in models
            if model in table
        }

    monkeypatch.setattr(HostedOffers, "defaults_for", _defaults_for)


@pytest.mark.usefixtures("together", "defaults")
def test_a_custom_price_the_default_now_prices_differently_is_flagged(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    body = _models(client, master_key_header)

    rows = {row["model"]: row for row in body["data"]}
    assert rows[QWEN]["default_rates"]["input_price_per_million"] == 0.15
    assert rows[CODER]["default_rates"] is None
    assert rows[DEEPSEEK]["default_rates"] is None
    assert _vendors(body)["Alibaba"]["custom_differs"] == 1


@pytest.mark.usefixtures("together", "defaults")
def test_a_dry_run_names_what_a_reset_would_change_and_writes_nothing(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    response = client.post(f"{PATH}/models/reset-prices", json={"dry_run": True}, headers=master_key_header)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["applied"] is False
    assert {change["model"]: change["matches"] for change in body["changes"]} == {CODER: True, QWEN: False}
    assert body["skipped_no_default"] == [DEEPSEEK]
    rows = {row["model"]: row for row in _models(client, master_key_header)["data"]}
    assert rows[QWEN]["price_source"] == "deployment"


@pytest.mark.usefixtures("together", "defaults")
def test_a_reset_puts_the_named_model_back_on_the_default_price(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    response = client.post(f"{PATH}/models/reset-prices", json={"models": [QWEN]}, headers=master_key_header)

    assert response.status_code == 200, response.text
    assert [change["model"] for change in response.json()["changes"]] == [QWEN]
    rows = {row["model"]: row for row in _models(client, master_key_header)["data"]}
    assert rows[QWEN]["price_source"] == "defaults"
    assert rows[QWEN]["input_price_per_million"] == 0.15
    assert rows[QWEN]["output_price_per_million"] == 0.2
    assert rows[CODER]["price_source"] == "deployment"


@pytest.mark.usefixtures("together", "defaults")
def test_a_reset_narrowed_to_a_vendor_leaves_the_others_alone(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    response = client.post(
        f"{PATH}/models/reset-prices", json={"vendor": "DeepSeek", "dry_run": True}, headers=master_key_header
    )

    assert response.json() == {"changes": [], "skipped_no_default": [DEEPSEEK], "applied": False}


@pytest.mark.usefixtures("together")
def test_removing_the_provider_takes_its_vendor_switches_with_it(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _set_vendor(client, master_key_header, "Alibaba", enabled=False)
    assert client.delete(PATH, headers=master_key_header).status_code == 200

    response = client.post(
        f"{API_ROOT}/hosted-providers", json={"provider": "together", "api_key": KEY}, headers=master_key_header
    )

    assert response.status_code == 201, response.text
    assert _vendors(_models(client, master_key_header))["Alibaba"]["enabled"] is True
