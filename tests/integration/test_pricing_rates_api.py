"""A price set or read through ``/pricing`` carries ``rates``, the cost block it is."""

from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT

pytestmark = pytest.mark.usefixtures("unpriced_builtin_fetch")

Headers = dict[str, str]
_ENDPOINT = f"{API_ROOT}/pricing"
_IMAGE = {"image": {"output": {"price": 0.04, "unit": "image"}}}


def _set(client: TestClient, headers: dict[str, str], **body: Any) -> Any:
    return client.post(_ENDPOINT, json={"model_key": "openai:gpt-image-1", **body}, headers=headers)


def test_a_price_given_as_rates_reads_back_with_its_columns(client: TestClient, master_key_header: Headers) -> None:
    response = _set(client, master_key_header, rates=_IMAGE)

    assert response.status_code == status.HTTP_200_OK, response.text
    row = response.json()
    assert (row["unit"], row["input_price_per_million"], row["output_price_per_million"]) == ("image", 40000.0, 0.0)
    assert row["rates"] == _IMAGE
    listed = client.get(_ENDPOINT, headers=master_key_header).json()
    assert next(item["rates"] for item in listed if item["model_key"] == "openai:gpt-image-1") == _IMAGE


def test_a_price_given_as_columns_reads_back_with_its_rates(client: TestClient, master_key_header: Headers) -> None:
    response = _set(
        client,
        master_key_header,
        input_price_per_million=3,
        output_price_per_million=15,
        cache_read_price_per_million=0.3,
    )

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json()["rates"] == {
        "text": {
            "input": {"price": 3.0, "unit": "token", "cache": {"read": 0.3}},
            "output": {"price": 15.0, "unit": "token"},
        }
    }


def test_rates_leave_no_cache_rate_to_inherit(client: TestClient, master_key_header: Headers) -> None:
    """A later version given as a block carries the block's cache prices alone, not the earlier version's."""
    assert (
        _set(
            client,
            master_key_header,
            input_price_per_million=3,
            output_price_per_million=15,
            cache_read_price_per_million=0.3,
        ).status_code
        == status.HTTP_200_OK
    )
    later = _set(client, master_key_header, rates={"text": {"input": {"price": 5}, "output": {"price": 20}}})

    assert later.status_code == status.HTTP_200_OK, later.text
    row = later.json()
    assert (row["cache_read_price_per_million"], row["rates"]["text"]["input"].get("cache")) == (None, None)


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (
            {"rates": _IMAGE, "input_price_per_million": 3, "output_price_per_million": 15},
            "remove input_price_per_million",
        ),
        ({"rates": _IMAGE, "unit": "image"}, "remove unit"),
        ({"rates": _IMAGE, "price_per_request": 0.01}, "remove price_per_request"),
        ({"rates": {"image": {"output": {"prize": 0.04}}}}, "prize"),
        ({"rates": {"text": {"input": {"price": 3, "unit": "tokens"}}}}, "unit"),
        ({"rates": {}}, "price nothing"),
        ({"rates": {"text": {"input": {"price": -1}}}}, "greater than or equal to 0"),
    ],
)
def test_a_block_the_api_cannot_take_is_refused(
    client: TestClient, master_key_header: Headers, body: dict[str, Any], reason: str
) -> None:
    response = _set(client, master_key_header, **body)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, response.text
    assert reason in response.text


def test_a_search_price_given_as_rates_is_per_request(client: TestClient, master_key_header: Headers) -> None:
    response = client.post(
        _ENDPOINT,
        json={"model_key": "exa", "kind": "search", "rates": {"text": {"input": {"price": 0.005, "unit": "request"}}}},
        headers=master_key_header,
    )

    assert response.status_code == status.HTTP_200_OK, response.text
    row = response.json()
    assert (row["unit"], row["price_per_request"], row["input_price_per_million"]) == ("request", 0.005, 5000.0)
