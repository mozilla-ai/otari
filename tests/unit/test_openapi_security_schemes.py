"""The OpenAPI document advertises every way the gateway accepts an API key (#846).

The gateway honors Otari-Key, x-api-key and Authorization: Bearer. An interactive
docs reader or a generated client can only offer what the spec's security schemes
describe, so all three are published and every authenticated operation lists them.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI

from gateway.core.config import API_KEY_HEADER, GatewayConfig, X_API_KEY_HEADER
from gateway.main import create_app
from gateway.services.secret_box import generate_secret_key


@pytest.fixture(scope="module")
def monkeypatch_module() -> Iterator[pytest.MonkeyPatch]:
    with pytest.MonkeyPatch.context() as patch:
        yield patch


@pytest.fixture(scope="module")
def spec(tmp_path_factory: pytest.TempPathFactory, monkeypatch_module: pytest.MonkeyPatch) -> dict[str, Any]:
    database = Path(tmp_path_factory.mktemp("openapi-security")) / "routes.db"
    monkeypatch_module.setenv("OTARI_SECRET_KEY", generate_secret_key())
    app: FastAPI = create_app(
        GatewayConfig(database_url=f"sqlite:///{database}", bootstrap=None, bootstrap_api_key=False)
    )
    document: dict[str, Any] = app.openapi()
    return document


def test_all_three_key_schemes_are_published(spec: dict[str, Any]) -> None:
    schemes = spec["components"]["securitySchemes"]
    assert schemes["ApiKeyAuth"] == {
        "type": "apiKey",
        "in": "header",
        "name": API_KEY_HEADER,
        "description": schemes["ApiKeyAuth"]["description"],
    }
    assert schemes["XApiKeyAuth"]["in"] == "header"
    assert schemes["XApiKeyAuth"]["name"] == X_API_KEY_HEADER
    assert schemes["BearerAuth"] == {
        "type": "http",
        "scheme": "bearer",
        "description": schemes["BearerAuth"]["description"],
    }


def test_every_authenticated_operation_offers_all_three_schemes(spec: dict[str, Any]) -> None:
    expected = [{"ApiKeyAuth": []}, {"XApiKeyAuth": []}, {"BearerAuth": []}]
    authenticated = 0
    for path_item in spec["paths"].values():
        for operation in path_item.values():
            if isinstance(operation, dict) and operation.get("security"):
                assert operation["security"] == expected
                authenticated += 1
    assert authenticated, "no authenticated operations found"


def test_public_operations_stay_unsecured(spec: dict[str, Any]) -> None:
    """A public endpoint carries an explicit empty requirement, so no scheme is suggested for it."""
    empties = [
        operation
        for path_item in spec["paths"].values()
        for operation in path_item.values()
        if isinstance(operation, dict) and operation.get("security") == []
    ]
    assert empties, "expected at least one public operation with security: []"
