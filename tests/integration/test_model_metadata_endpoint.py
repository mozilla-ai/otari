"""Tests for GET /v1/models/metadata (models.dev enrichment, mocked fetch)."""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import GatewayConfig
from gateway.services import model_catalog_service as mcs

from .conftest import build_test_client

CATALOG: dict[str, Any] = {
    "openai": {
        "id": "openai",
        "name": "OpenAI",
        "models": {
            "gpt-4o": {
                "id": "gpt-4o",
                "name": "GPT-4o",
                "tool_call": True,
                "reasoning": False,
                "modalities": {"input": ["text", "image"], "output": ["text"]},
                "limit": {"context": 128000, "output": 16384},
                "knowledge": "2023-09",
            },
        },
    },
}


def _make_client(config: GatewayConfig) -> Generator[TestClient]:
    mcs.clear_catalog_cache()
    try:
        yield from build_test_client(config)
    finally:
        mcs.clear_catalog_cache()


def _config(postgres_url: str, **overrides: Any) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        providers={"openai": {"api_key": "sk-fake"}},
        **overrides,
    )


@pytest.fixture
def master_header() -> dict[str, str]:
    return {"Authorization": "Bearer test-master-key"}


def test_metadata_enriches_configured_models(postgres_url: str, master_header: dict[str, str]) -> None:
    client_gen = _make_client(_config(postgres_url))
    client = next(client_gen)
    try:
        with patch.object(mcs, "_fetch", new=AsyncMock(return_value=CATALOG)):
            resp = client.get("/v1/models/metadata", headers=master_header)
        assert resp.status_code == 200
        body = resp.json()
        assert body["available"] is True
        assert body["source"] == "models.dev"
        entry = body["models"]["openai:gpt-4o"]
        assert entry["tool_call"] is True
        assert entry["input_modalities"] == ["text", "image"]
        assert entry["context_window"] == 128000
        assert entry["knowledge_cutoff"] == "2023-09"
    finally:
        client_gen.close()


def test_metadata_unavailable_when_fetch_fails(postgres_url: str, master_header: dict[str, str]) -> None:
    client_gen = _make_client(_config(postgres_url))
    client = next(client_gen)
    try:
        # A failed fetch degrades to available=false and an empty map.
        with patch.object(mcs, "_fetch", new=AsyncMock(return_value=None)):
            resp = client.get("/v1/models/metadata", headers=master_header)
        assert resp.status_code == 200
        body = resp.json()
        assert body["available"] is False
        assert body["models"] == {}
    finally:
        client_gen.close()


def test_metadata_disabled_makes_no_fetch(postgres_url: str, master_header: dict[str, str]) -> None:
    client_gen = _make_client(_config(postgres_url, models_dev_metadata=False))
    client = next(client_gen)
    try:
        fetch_mock = AsyncMock(return_value=CATALOG)
        with patch.object(mcs, "_fetch", new=fetch_mock):
            resp = client.get("/v1/models/metadata", headers=master_header)
        assert resp.status_code == 200
        assert resp.json()["available"] is False
        # The outbound call is never made when enrichment is disabled.
        fetch_mock.assert_not_awaited()
    finally:
        client_gen.close()


def test_metadata_requires_master_key(postgres_url: str) -> None:
    client_gen = _make_client(_config(postgres_url))
    client = next(client_gen)
    try:
        resp = client.get("/v1/models/metadata")
        assert resp.status_code in (401, 403)
    finally:
        client_gen.close()
