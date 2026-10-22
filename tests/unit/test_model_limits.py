"""Unit tests for a model's token limits and capabilities, from the live models.dev catalog and the pricing snapshot."""

from typing import Any

from gateway.core.config import GatewayConfig
from gateway.services.catalog._limits import (
    ModelArchitecture,
    ModelCapabilities,
    ModelLimits,
    apply_model_metadata,
    model_capabilities,
    model_limits,
)
from gateway.services.merged_catalog_service import ModelObject

# Limits that differ from the pricing snapshot's wherever both know the model, so a test can tell which one answered.
MODELS_DEV: dict[str, Any] = {
    "openai": {
        "models": {
            "gpt-4o": {
                "name": "GPT-4o",
                "tool_call": True,
                "structured_output": True,
                "modalities": {"input": ["text", "image"], "output": ["text"]},
                "limit": {"context": 64000, "output": 16384},
            },
        },
    },
    "nebius": {"models": {"Qwen/Qwen3-30B-A3B-Instruct-2507": {"limit": {"context": 262144, "output": 8192}}}},
    "google": {"models": {"gemini-3.5-flash-lite": {"limit": {"context": 1048576, "output": 65536}}}},
    # An entry with no window, so the dataset below it is asked.
    "anthropic": {"models": {"claude-sonnet-4-5": {"limit": {"output": 64000}}}},
}


def _config(**providers: dict[str, Any]) -> GatewayConfig:
    return GatewayConfig(master_key="test", providers=providers)


def test_models_dev_answers_first() -> None:
    """Where both know a model, the live catalog's numbers win over the snapshot's 128000."""
    assert model_limits(_config(), MODELS_DEV, "openai:gpt-4o") == ModelLimits(64000, 16384)


def test_a_model_only_models_dev_knows_reports_both_limits() -> None:
    """The issue's model: the snapshot has no entry, the live catalog has both limits."""
    limits = model_limits(_config(), MODELS_DEV, "nebius:Qwen/Qwen3-30B-A3B-Instruct-2507")

    assert limits == ModelLimits(262144, 8192)


def test_the_snapshot_fills_a_window_the_live_catalog_lacks() -> None:
    assert model_limits(_config(), MODELS_DEV, "anthropic:claude-sonnet-4-5") == ModelLimits(200000, 64000)
    # No live catalog at all: the snapshot's window, and no output limit, which it never keeps.
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


def test_a_bare_key_has_no_limits() -> None:
    """A bare name carries no provider to look either the live catalog or the snapshot up under."""
    assert model_limits(_config(), MODELS_DEV, "gpt-4o") == ModelLimits(None, None)


def test_apply_model_metadata_reads_the_selector_it_is_given() -> None:
    """An alias's id is a display name; its limits come from the target."""
    alias = ModelObject(id="house-qwen", created=0, owned_by="otari")

    apply_model_metadata(_config(), MODELS_DEV, alias, metadata_selector="nebius:Qwen/Qwen3-30B-A3B-Instruct-2507")

    assert (alias.context_window, alias.max_output_tokens) == (262144, 8192)


def test_capabilities_map_models_dev_flags_to_openrouter_fields() -> None:
    """A tool-calling, structured-output model lists the parameters those flags stand for."""
    assert model_capabilities(_config(), MODELS_DEV, "openai:gpt-4o") == ModelCapabilities(
        name="GPT-4o",
        supported_parameters=["tools", "tool_choice", "response_format"],
        input_modalities=["text", "image"],
        output_modalities=["text"],
    )


def test_an_entry_without_flags_lists_no_parameters() -> None:
    """Known to models.dev but flagged for nothing: an empty list, which is not the same as unknown."""
    capabilities = model_capabilities(_config(), MODELS_DEV, "nebius:Qwen/Qwen3-30B-A3B-Instruct-2507")

    assert capabilities is not None
    assert capabilities.supported_parameters == []


def test_a_model_models_dev_does_not_describe_has_no_capabilities() -> None:
    assert model_capabilities(_config(), MODELS_DEV, "nebius:made-up-model-xyz") is None
    assert model_capabilities(_config(), None, "openai:gpt-4o") is None
    assert model_capabilities(_config(), MODELS_DEV, "gpt-4o") is None


def test_apply_model_metadata_fills_capabilities() -> None:
    model = ModelObject(id="openai:gpt-4o", created=0, owned_by="openai")

    apply_model_metadata(_config(), MODELS_DEV, model)

    assert model.name == "GPT-4o"
    assert model.supported_parameters == ["tools", "tool_choice", "response_format"]
    assert model.architecture == ModelArchitecture(input_modalities=["text", "image"], output_modalities=["text"])


def test_an_alias_takes_its_targets_capabilities_but_not_its_name() -> None:
    """The target's display name would say what the alias hides."""
    alias = ModelObject(id="house-4o", created=0, owned_by="otari")

    apply_model_metadata(_config(), MODELS_DEV, alias, metadata_selector="openai:gpt-4o", with_name=False)

    assert alias.name is None
    assert alias.supported_parameters == ["tools", "tool_choice", "response_format"]


def test_an_undescribed_model_reports_unknown_capabilities() -> None:
    model = ModelObject(id="nebius:made-up-model-xyz", created=0, owned_by="nebius")

    apply_model_metadata(_config(), MODELS_DEV, model)

    assert (model.name, model.supported_parameters, model.architecture) == (None, None, None)
