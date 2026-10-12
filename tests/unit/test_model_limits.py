"""Unit tests for a model's token limits, read from models.dev and then genai-prices."""

from typing import Any

from gateway.core.config import GatewayConfig
from gateway.services.catalog._limits import ModelLimits, apply_model_limits, model_limits
from gateway.services.merged_catalog_service import ModelObject

# Limits that differ from genai-prices' wherever both know the model, so a test can tell which one answered.
MODELS_DEV: dict[str, Any] = {
    "openai": {"models": {"gpt-4o": {"limit": {"context": 64000, "output": 16384}}}},
    "nebius": {"models": {"Qwen/Qwen3-30B-A3B-Instruct-2507": {"limit": {"context": 262144, "output": 8192}}}},
    "google": {"models": {"gemini-3.5-flash-lite": {"limit": {"context": 1048576, "output": 65536}}}},
    # An entry with no window, so the dataset below it is asked.
    "anthropic": {"models": {"claude-sonnet-4-5": {"limit": {"output": 64000}}}},
}


def _config(**providers: dict[str, Any]) -> GatewayConfig:
    return GatewayConfig(master_key="test", providers=providers)


def test_models_dev_answers_first() -> None:
    """Where both datasets know a model, models.dev's numbers win over genai-prices' 128000."""
    assert model_limits(_config(), MODELS_DEV, "openai:gpt-4o") == ModelLimits(64000, 16384)


def test_a_model_only_models_dev_knows_reports_both_limits() -> None:
    """The issue's model: genai-prices has no entry, models.dev has both limits."""
    limits = model_limits(_config(), MODELS_DEV, "nebius:Qwen/Qwen3-30B-A3B-Instruct-2507")

    assert limits == ModelLimits(262144, 8192)


def test_genai_prices_fills_a_window_models_dev_lacks() -> None:
    assert model_limits(_config(), MODELS_DEV, "anthropic:claude-sonnet-4-5") == ModelLimits(200000, 64000)
    # No models.dev at all: the window genai-prices knows, and no output limit, which it never lists.
    assert model_limits(_config(), None, "openai:gpt-4o") == ModelLimits(128000, None)


def test_a_model_neither_dataset_knows_has_no_limits() -> None:
    assert model_limits(_config(), MODELS_DEV, "nebius:made-up-model-xyz") == ModelLimits(None, None)


def test_models_dev_is_looked_up_by_the_instance_provider_type() -> None:
    """A named instance reads its implementation's entries, the way the grouped catalog does."""
    config = _config(home_lab={"provider_type": "openai", "api_key": "t"})

    assert model_limits(config, MODELS_DEV, "home_lab:gpt-4o") == ModelLimits(64000, 16384)


def test_a_gemini_listing_id_finds_its_entry() -> None:
    """Gemini lists ``models/<id>``, where models.dev keys the bare id."""
    assert model_limits(_config(), MODELS_DEV, "gemini:models/gemini-3.5-flash-lite") == ModelLimits(1048576, 65536)


def test_a_bare_key_is_answered_by_genai_prices_alone() -> None:
    """A bare name carries no provider to look models.dev up under."""
    assert model_limits(_config(), MODELS_DEV, "gpt-4o") == ModelLimits(128000, None)


def test_apply_model_limits_reads_the_selector_it_is_given() -> None:
    """An alias's id is a display name; its limits come from the target."""
    alias = ModelObject(id="house-qwen", created=0, owned_by="otari")

    apply_model_limits(_config(), MODELS_DEV, alias, limits_selector="nebius:Qwen/Qwen3-30B-A3B-Instruct-2507")

    assert (alias.context_window, alias.max_output_tokens) == (262144, 8192)
