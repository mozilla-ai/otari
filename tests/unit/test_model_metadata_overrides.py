"""Unit tests for how a model metadata override sits over what its server and models.dev say."""

import uuid
from unittest.mock import MagicMock

import pytest

from gateway.core.config import GatewayConfig
from gateway.models.catalog import ModelMetadataOverride
from gateway.services.catalog import _overrides, _served
from gateway.services.catalog._limits import ModelLimits, model_capabilities, model_limits, served_context_window
from gateway.services.catalog._overrides import MetadataOverride
from gateway.services.catalog._served import ServedFacts
from gateway.types.models_dev import parse_catalog

MODELS_DEV = parse_catalog(
    {
        "huggingface": {
            "models": {
                "Qwen/Qwen3-32B": {
                    "name": "Qwen3 32B",
                    "tool_call": True,
                    "reasoning": True,
                    "structured_output": True,
                    "modalities": {"input": ["text"], "output": ["text"]},
                    "limit": {"context": 524288, "output": 300000},
                },
            },
        },
        "amazon-bedrock": {
            "models": {
                "meta.llama4": {"name": "Llama 4", "tool_call": True, "limit": {"context": 1000, "output": 100}},
            },
        },
    }
)

ACME = uuid.UUID(int=1)


def _config() -> GatewayConfig:
    return GatewayConfig(
        master_key="test",
        providers={
            "cairn": {"provider_type": "vllm", "api_base": "http://x/v1"},
            "bedrock": {"provider_type": "bedrock"},
        },
    )


@pytest.fixture(autouse=True)
def _empty_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_served, "_served", {})
    monkeypatch.setattr(_overrides, "_overrides", {})


def _serve(provider: str, model_id: str, facts: ServedFacts) -> None:
    _served._served.setdefault((None, provider), {})[model_id] = facts


def _override(
    provider: str, model_id: str, override: MetadataOverride, organization_id: uuid.UUID | None = None
) -> None:
    _overrides._overrides.setdefault((organization_id, provider), {})[model_id] = override


def test_an_override_describes_a_model_nothing_else_does() -> None:
    _override(
        "cairn",
        "glm",
        MetadataOverride(
            context_window=1048576, max_output_tokens=65536, tool_call=True, input_modalities=("text", "image")
        ),
    )

    capabilities = model_capabilities(_config(), MODELS_DEV, "cairn:glm")

    assert model_limits(_config(), MODELS_DEV, "cairn:glm") == ModelLimits(1048576, 65536)
    assert capabilities is not None
    assert capabilities.supported_parameters == ["tools", "tool_choice"]
    assert capabilities.input_modalities == ["text", "image"]


def test_an_override_beats_the_server_and_models_dev_field_by_field() -> None:
    _serve("cairn", "cairn-qwen", ServedFacts(engine="vllm", context_window=262144, root="Qwen/Qwen3-32B"))
    _override("cairn", "cairn-qwen", MetadataOverride(context_window=131072, tool_call=False))

    capabilities = model_capabilities(_config(), MODELS_DEV, "cairn:cairn-qwen")

    # models.dev's output limit still applies, under the overridden window.
    assert model_limits(_config(), MODELS_DEV, "cairn:cairn-qwen") == ModelLimits(131072, 131072)
    assert capabilities is not None
    assert capabilities.supported_parameters == ["response_format", "reasoning"]


def test_an_override_turns_off_what_the_server_s_parsers_report() -> None:
    _serve(
        "cairn",
        "glm",
        ServedFacts(
            engine="vllm",
            settings_reported=True,
            tool_call_parser="hermes",
            reasoning_parser="qwen3",
            input_modalities=("text", "image"),
        ),
    )
    _override("cairn", "glm", MetadataOverride(tool_call=False, reasoning=False, input_modalities=("text",)))

    capabilities = model_capabilities(_config(), MODELS_DEV, "cairn:glm")

    assert capabilities is not None
    assert capabilities.supported_parameters == []
    assert capabilities.input_modalities == ["text"]


def test_an_override_s_output_limit_replaces_models_dev_s() -> None:
    _serve("cairn", "cairn-qwen", ServedFacts(engine="vllm", context_window=262144, root="Qwen/Qwen3-32B"))
    _override("cairn", "cairn-qwen", MetadataOverride(max_output_tokens=4096))

    assert model_limits(_config(), MODELS_DEV, "cairn:cairn-qwen") == ModelLimits(262144, 4096)


def test_an_organization_without_its_own_override_reads_the_deployment_s() -> None:
    _override("byo", "glm", MetadataOverride(context_window=8192))

    assert model_limits(_config(), MODELS_DEV, "byo:glm", ACME).context_window == 8192
    assert served_context_window("byo", "glm", ACME) == 8192


@pytest.mark.asyncio
@pytest.mark.parametrize("selector", ["glm", "cairn:", ":glm"])
async def test_a_selector_missing_a_provider_or_a_model_is_refused_before_any_write(selector: str) -> None:
    repository = MagicMock()
    service = _overrides.ModelMetadataOverrideService(MagicMock(), overrides=repository)

    with pytest.raises(ValueError, match="is not a provider:model selector"):
        await service.set(selector, MetadataOverride(context_window=8192))
    with pytest.raises(ValueError, match="is not a provider:model selector"):
        await service.clear(selector)
    assert repository.method_calls == []


def test_an_override_cannot_offer_what_any_llm_will_not_send() -> None:
    _override("bedrock", "meta.llama4", MetadataOverride(structured_output=True))

    capabilities = model_capabilities(_config(), MODELS_DEV, "bedrock:meta.llama4")

    assert capabilities is not None
    assert "response_format" not in capabilities.supported_parameters


def test_an_organization_s_override_is_read_only_by_its_own_callers() -> None:
    _override("byo", "glm", MetadataOverride(context_window=8192), ACME)

    assert model_limits(_config(), MODELS_DEV, "byo:glm", ACME).context_window == 8192
    assert model_limits(_config(), MODELS_DEV, "byo:glm").context_window is None
    assert model_limits(_config(), MODELS_DEV, "byo:glm", uuid.UUID(int=2)).context_window is None


def test_an_organization_never_overrides_a_configured_instance() -> None:
    _override("cairn", "glm", MetadataOverride(context_window=4096))
    _override("cairn", "glm", MetadataOverride(context_window=8192), ACME)

    assert model_limits(_config(), MODELS_DEV, "cairn:glm", ACME).context_window == 4096


def test_the_trace_view_reads_the_overridden_window() -> None:
    _serve("cairn", "glm", ServedFacts(engine="vllm", context_window=262144))
    _override("cairn", "glm", MetadataOverride(context_window=131072))
    _override("cairn", "glm", MetadataOverride(context_window=65536), ACME)

    assert served_context_window("cairn", "glm") == 131072
    assert served_context_window("cairn", "glm", ACME) == 65536


def test_a_reload_read_before_a_local_write_does_not_undo_it() -> None:
    _overrides.replace_overrides([])
    stale = ModelMetadataOverride(provider="cairn", model="qwen", organization_id=None, context_window=8192)

    generation = _overrides.overrides_generation()
    _overrides.ModelMetadataOverrideService.forget_provider("cairn")
    _overrides.replace_overrides([stale], generation)
    assert _overrides.metadata_override("cairn", "qwen") is None

    _overrides.replace_overrides([stale], _overrides.overrides_generation())
    assert _overrides.metadata_override("cairn", "qwen") == MetadataOverride(context_window=8192)
    _overrides.replace_overrides([])
