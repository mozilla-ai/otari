"""Unit tests for the models.dev catalog parsing and mapping (no network)."""

from typing import Any

import pytest

from gateway.core.config import GatewayConfig
from gateway.services import model_catalog_service as mcs
from gateway.services.model_catalog_service import build_metadata_map, parse_entry


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    mcs.clear_catalog_cache()


def _config(providers: dict[str, dict[str, Any]]) -> GatewayConfig:
    return GatewayConfig(master_key="test", providers=providers)


GPT_4O = {
    "id": "gpt-4o",
    "name": "GPT-4o",
    "description": "General reasoning model.",
    "family": "gpt",
    "reasoning": False,
    "tool_call": True,
    "structured_output": True,
    "attachment": True,
    "temperature": True,
    "knowledge": "2023-09",
    "release_date": "2024-05-13",
    "last_updated": "2024-08-06",
    "modalities": {"input": ["text", "image", "pdf"], "output": ["text"]},
    "open_weights": False,
    "limit": {"context": 128000, "output": 16384},
    "cost": {"input": 2.5, "output": 10, "cache_read": 1.25},
}

CATALOG: dict[str, Any] = {
    "openai": {"id": "openai", "name": "OpenAI", "models": {"gpt-4o": GPT_4O}},
}


def test_parse_entry_reads_all_fields() -> None:
    entry = parse_entry(GPT_4O)
    assert entry.name == "GPT-4o"
    assert entry.input_modalities == ["text", "image", "pdf"]
    assert entry.output_modalities == ["text"]
    assert entry.tool_call is True
    assert entry.reasoning is False
    assert entry.structured_output is True
    assert entry.context_window == 128000
    assert entry.max_output_tokens == 16384
    assert entry.knowledge_cutoff == "2023-09"
    assert entry.release_date == "2024-05-13"
    assert entry.cost_input == 2.5
    assert entry.cost_output == 10.0
    assert entry.deprecated is False


def test_parse_entry_tolerates_missing_fields() -> None:
    entry = parse_entry({"id": "bare"})
    assert entry.name is None
    assert entry.input_modalities == []
    assert entry.context_window is None
    assert entry.tool_call is False
    assert entry.deprecated is False


def test_parse_entry_marks_deprecated_status() -> None:
    assert parse_entry({"id": "old", "status": "deprecated"}).deprecated is True


def test_build_metadata_map_keys_by_instance() -> None:
    config = _config({"openai": {"api_key": "sk-x"}})
    result = build_metadata_map(config, CATALOG)
    assert set(result) == {"openai:gpt-4o"}
    assert result["openai:gpt-4o"].tool_call is True


def test_build_metadata_map_uses_provider_type_for_named_instance() -> None:
    # A named instance backed by openai joins to openai's models under its own key.
    config = _config({"my-oai": {"provider_type": "openai", "api_key": "sk-x"}})
    result = build_metadata_map(config, CATALOG)
    assert set(result) == {"my-oai:gpt-4o"}


def test_build_metadata_map_skips_unknown_providers() -> None:
    config = _config({"mystery": {"api_key": "x"}})
    assert build_metadata_map(config, CATALOG) == {}


def test_build_metadata_map_empty_when_catalog_missing() -> None:
    config = _config({"openai": {"api_key": "sk-x"}})
    assert build_metadata_map(config, None) == {}
