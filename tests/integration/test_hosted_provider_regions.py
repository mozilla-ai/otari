"""A hosted provider that serves from several regions, each a connection of its own under one name.

The provider call is captured at the any-llm boundary, so each test reads which
key and endpoint a request was sent with, and the usage row reads which region
it settled under and at what price.
"""

from collections.abc import Generator, Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.model import Model
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, API_ROOT
from gateway.core.error_codes import COUNTRY_NOT_SUPPORTED, ERROR_CODE_HEADER
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.secret_box import generate_secret_key

from .conftest import build_test_client, shared_test_config
from .test_routing_policies import _completion

HEADERS = {API_KEY_HEADER: "Bearer test-master-key"}
_COUNTRY_HEADER = "CF-IPCountry"
_PROVIDER = f"{API_ROOT}/hosted-providers/tokenhub"


@pytest.fixture(autouse=True)
def _environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    async def _stub(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name, models=[Model(id="gpt-4o", object="model", created=0, owned_by=impl_name)]
        )

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _stub)
    yield


@pytest.fixture
def client(postgres_url: str) -> Generator[TestClient]:
    """``tokenhub``, an OpenAI-compatible provider served from the US, priced at 2.5 / 10."""
    config = shared_test_config(postgres_url, {}).model_copy(update={"client_country_header": _COUNTRY_HEADER})
    for client in build_test_client(config):
        _ok(client.post(f"{API_ROOT}/users", json={"user_id": "u1"}, headers=HEADERS))
        _ok(
            client.post(
                f"{API_ROOT}/pricing",
                json={"model_key": "tokenhub:gpt-4o", "input_price_per_million": 2.5, "output_price_per_million": 10},
                headers=HEADERS,
            )
        )
        _ok(
            client.post(
                f"{API_ROOT}/hosted-providers",
                json={
                    "provider": "openai",
                    "name": "tokenhub",
                    "api_key": "sk-us",
                    "api_base": "https://us.tokenhub.example/v1",
                    "data_location": "US",
                },
                headers=HEADERS,
            )
        )
        yield client


def _ok(response: Any) -> Any:
    assert response.status_code in (200, 201), response.text
    return response.json()


def _add_singapore(client: TestClient, **fields: Any) -> Any:
    body = {
        "slug": "sg",
        "label": "Southeast Asia",
        "api_key": "sk-sg",
        "api_base": "https://sg.tokenhub.example/v1",
        "data_location": "sg",
        **fields,
    }
    return client.post(f"{_PROVIDER}/regions", json=body, headers=HEADERS)


def _chat(client: TestClient, model: str, country: str = "US") -> tuple[Any, dict[str, Any]]:
    captured: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _completion("gpt-4o")

    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=fake)):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}], "user": "u1"},
            headers={**HEADERS, _COUNTRY_HEADER: country},
        )
    return response, captured


def _last_usage(client: TestClient) -> Any:
    rows = _ok(client.get(f"{API_ROOT}/usage", headers=HEADERS))
    return rows[0]


def test_a_provider_with_no_regions_serves_from_its_own_connection(client: TestClient) -> None:
    provider = _ok(client.get(f"{API_ROOT}/hosted-providers", headers=HEADERS))["data"][0]
    response, sent = _chat(client, "tokenhub:gpt-4o")

    assert (provider["regions"], provider["data_location"]) == ([], "US")
    assert response.status_code == 200, response.text
    assert (sent["api_key"], sent["api_base"]) == ("sk-us", "https://us.tokenhub.example/v1")
    assert _last_usage(client)["provider"] == "tokenhub"


def test_the_first_region_keeps_the_provider_connection_ahead_of_it(client: TestClient) -> None:
    provider = _ok(_add_singapore(client))

    regions = [(each["slug"], each["label"], each["selector"], each["position"]) for each in provider["regions"]]
    assert regions == [("us", "US", "tokenhub@us", 0), ("sg", "Southeast Asia", "tokenhub@sg", 1)]
    us, sg = provider["regions"]
    assert (us["has_api_key"], us["api_base"]) == (False, None), "the first uses the provider's connection"
    assert us["data_location"] == "US", "the provider's location is copied onto it"
    assert (sg["has_api_key"], sg["api_key_last4"], sg["data_location"]) == (True, "k-sg", "SG")


def test_a_request_is_served_by_the_first_region_and_settles_under_it(client: TestClient) -> None:
    _ok(_add_singapore(client))

    response, sent = _chat(client, "tokenhub:gpt-4o")

    assert response.status_code == 200, response.text
    assert sent["api_key"] == "sk-us"
    usage = _last_usage(client)
    assert usage["provider"] == "tokenhub@us"
    assert usage["data_location"] == "US"
    assert float(usage["cost"]) == pytest.approx(10 * 2.5e-6 + 5 * 10e-6), "a region with no price is the provider's"


def test_a_country_one_region_leaves_out_is_served_by_the_next(client: TestClient) -> None:
    _ok(_add_singapore(client, supported_countries=["SG"]))
    _ok(client.patch(f"{_PROVIDER}/regions/us", json={"supported_countries": ["US"]}, headers=HEADERS))

    from_singapore, sent_sg = _chat(client, "tokenhub:gpt-4o", "SG")
    from_france, _ = _chat(client, "tokenhub:gpt-4o", "FR")

    assert from_singapore.status_code == 200, from_singapore.text
    assert (sent_sg["api_key"], sent_sg["api_base"]) == ("sk-sg", "https://sg.tokenhub.example/v1")
    assert from_france.status_code == 403
    assert from_france.headers[ERROR_CODE_HEADER] == COUNTRY_NOT_SUPPORTED


def test_a_pinned_region_is_served_alone_and_priced_on_its_own_rate(client: TestClient) -> None:
    _ok(_add_singapore(client))
    _ok(
        client.post(
            f"{API_ROOT}/pricing",
            json={"model_key": "tokenhub@sg:gpt-4o", "input_price_per_million": 5, "output_price_per_million": 20},
            headers=HEADERS,
        )
    )

    response, sent = _chat(client, "tokenhub@sg:gpt-4o")
    models = _ok(client.get(f"{_PROVIDER}/models", headers=HEADERS))["data"]

    assert response.status_code == 200, response.text
    assert sent["api_key"] == "sk-sg"
    usage = _last_usage(client)
    assert usage["provider"] == "tokenhub@sg"
    assert float(usage["cost"]) == pytest.approx(10 * 5e-6 + 5 * 20e-6)
    [gpt] = models
    assert set(gpt["region_rates"]) == {"sg"}
    assert gpt["region_rates"]["sg"]["input_price_per_million"] == 5


def test_a_region_switched_off_is_skipped_and_cannot_be_pinned(client: TestClient) -> None:
    _ok(_add_singapore(client))
    _ok(client.patch(f"{_PROVIDER}/regions/us", json={"enabled": False}, headers=HEADERS))

    unpinned, sent = _chat(client, "tokenhub:gpt-4o")
    pinned, pinned_sent = _chat(client, "tokenhub@us:gpt-4o")

    assert unpinned.status_code == 200, unpinned.text
    assert sent["api_key"] == "sk-sg"
    assert pinned.status_code == 424, pinned.text
    assert not pinned_sent, "a region switched off is never called"


def test_moving_a_region_changes_which_one_serves_first(client: TestClient) -> None:
    _ok(_add_singapore(client))

    provider = _ok(client.patch(f"{_PROVIDER}/regions/sg", json={"position": 0}, headers=HEADERS))
    _, sent = _chat(client, "tokenhub:gpt-4o")

    assert [(region["slug"], region["position"]) for region in provider["regions"]] == [("sg", 0), ("us", 1)]
    assert sent["api_key"] == "sk-sg"


def test_a_region_given_back_the_provider_key_inherits_it(client: TestClient) -> None:
    _ok(_add_singapore(client, api_base=None))

    provider = _ok(client.patch(f"{_PROVIDER}/regions/sg", json={"api_key": None}, headers=HEADERS))
    _, sent = _chat(client, "tokenhub@sg:gpt-4o")

    assert provider["regions"][1]["has_api_key"] is False
    assert sent["api_key"] == "sk-us"


def test_a_region_may_not_send_the_provider_key_to_another_endpoint(client: TestClient) -> None:
    response = _add_singapore(client, api_key=None)

    assert response.status_code == 400, response.text
    assert _ok(client.get(f"{API_ROOT}/hosted-providers", headers=HEADERS))["data"][0]["regions"] == []


def test_removing_every_region_returns_the_provider_to_its_own_connection(client: TestClient) -> None:
    _ok(_add_singapore(client))

    _ok(client.delete(f"{_PROVIDER}/regions/sg", headers=HEADERS))
    provider = _ok(client.delete(f"{_PROVIDER}/regions/us", headers=HEADERS))
    _, sent = _chat(client, "tokenhub:gpt-4o")

    assert provider["regions"] == []
    assert sent["api_key"] == "sk-us"
    assert _last_usage(client)["provider"] == "tokenhub"


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"slug": "Bad Slug", "label": "x", "api_key": "sk"}, 400),
        ({"slug": "us", "label": "again", "api_key": "sk", "primary_slug": "us"}, 409),
        ({"slug": "eu", "label": "x", "api_key": "sk", "data_location": "Mars"}, 422),
    ],
)
def test_a_region_that_cannot_be_stored_is_refused(client: TestClient, body: dict[str, Any], status: int) -> None:
    response = client.post(f"{_PROVIDER}/regions", json=body, headers=HEADERS)

    assert response.status_code == status, response.text


def test_an_unknown_region_answers_404(client: TestClient) -> None:
    response = client.patch(f"{_PROVIDER}/regions/nowhere", json={"enabled": False}, headers=HEADERS)

    assert response.status_code == 404, response.text


def test_the_last_region_switched_on_cannot_be_switched_off(client: TestClient) -> None:
    _ok(_add_singapore(client))
    _ok(client.patch(f"{_PROVIDER}/regions/us", json={"enabled": False}, headers=HEADERS))

    response = client.patch(f"{_PROVIDER}/regions/sg", json={"enabled": False}, headers=HEADERS)
    removed = client.delete(f"{_PROVIDER}/regions/sg", headers=HEADERS)

    assert (response.status_code, removed.status_code) == (400, 400), (response.text, removed.text)


def test_the_provider_endpoint_may_not_move_under_a_region_with_a_key_of_its_own(client: TestClient) -> None:
    _ok(_add_singapore(client, api_base=None))

    response = client.patch(
        _PROVIDER, json={"api_base": "https://elsewhere.example/v1", "api_key": "sk-new"}, headers=HEADERS
    )

    assert response.status_code == 400, response.text
    assert "Region 'sg'" in response.json()["detail"]


def test_a_region_on_another_endpoint_does_not_take_the_provider_extras(client: TestClient) -> None:
    _ok(client.patch(_PROVIDER, json={"client_args": {"default_headers": {"X-Team": "a"}}}, headers=HEADERS))
    _ok(_add_singapore(client, client_args={"timeout": 5}))

    _, sent_us = _chat(client, "tokenhub@us:gpt-4o")
    _, sent_sg = _chat(client, "tokenhub@sg:gpt-4o")

    assert sent_us["client_args"] == {"default_headers": {"X-Team": "a"}}
    assert sent_sg["client_args"] == {"timeout": 5}


def test_a_stored_provider_may_not_take_a_name_that_reads_as_a_region(client: TestClient) -> None:
    created = client.post(
        f"{API_ROOT}/hosted-providers",
        json={"provider": "openai", "name": "tokenhub@jp", "api_key": "sk-jp"},
        headers=HEADERS,
    )

    assert created.status_code == 400, "a stored provider is refused the separator"


def test_a_region_endpoint_moved_without_its_key_is_refused(client: TestClient) -> None:
    _ok(_add_singapore(client))

    moved = client.patch(f"{_PROVIDER}/regions/sg", json={"api_base": "https://elsewhere.example/v1"}, headers=HEADERS)
    same_origin = client.patch(
        f"{_PROVIDER}/regions/sg", json={"api_base": "https://sg.tokenhub.example/v2"}, headers=HEADERS
    )

    assert moved.status_code == 400, moved.text
    assert same_origin.status_code == 200, same_origin.text


def test_a_region_endpoint_carrying_a_key_must_be_https(client: TestClient) -> None:
    response = _add_singapore(client, api_base="http://sg.tokenhub.example/v1")

    assert response.status_code == 400, response.text


def test_a_region_keeps_its_data_location_when_the_provider_changes(client: TestClient) -> None:
    _ok(_add_singapore(client, data_location=None))

    _ok(client.patch(_PROVIDER, json={"data_location": "DE"}, headers=HEADERS))
    provider = _ok(client.get(f"{API_ROOT}/hosted-providers", headers=HEADERS))["data"][0]
    _chat(client, "tokenhub@sg:gpt-4o")

    us, sg = provider["regions"]
    assert (us["data_location"], sg["data_location"]) == ("US", None), "copied, never inherited"
    assert _last_usage(client)["data_location"] is None, "a region declaring nothing records nothing"


def test_a_region_left_without_a_location_copies_the_provider_one(client: TestClient) -> None:
    body = {"slug": "sg", "label": "Southeast Asia", "api_key": "sk-sg", "api_base": "https://sg.tokenhub.example/v1"}

    provider = _ok(client.post(f"{_PROVIDER}/regions", json=body, headers=HEADERS))

    assert provider["regions"][1]["data_location"] == "US"
