"""What kind of model a provider's listing and models.dev say a model is, and the refusals that follow."""

from collections.abc import Iterator
from typing import Any

import pytest
from any_llm.types.model import Model

from gateway.core.config import GatewayConfig
from gateway.services.catalog import model_type, not_chat_detail, not_embedding_detail
from gateway.services.catalog._model_type import type_from_listing
from gateway.services.model_discovery_service import get_model_cache
from gateway.types.models_dev import parse_catalog


@pytest.fixture(autouse=True)
def _empty_cache() -> Iterator[None]:
    get_model_cache().clear()
    yield
    get_model_cache().clear()


def _listed(provider: str, model_id: str, **fields: Any) -> None:
    get_model_cache().set(provider, [Model(id=model_id, object="model", created=0, owned_by=provider, **fields)])


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        # Together
        ({"type": "chat"}, "chat"),
        ({"type": "language"}, "chat"),
        ({"type": "embedding"}, "embedding"),
        ({"type": "rerank"}, "rerank"),
        ({"type": "audio"}, "speech"),
        ({"type": "transcribe"}, "transcription"),
        # DeepInfra
        ({"metadata": {"tags": ["embed"]}}, "embedding"),
        ({"metadata": {"tags": ["vlm", "chat", "vision"]}}, "chat"),
        ({"metadata": {"tags": ["stt"]}}, "transcription"),
        # DeepSeek, OpenRouter
        ({"output_modalities": ["text"]}, "chat"),
        ({"architecture": {"output_modalities": ["embeddings"]}}, "embedding"),
        # OVHcloud: a model that writes no completion tokens is no chat model, of no kind it names.
        ({"max_completion_tokens": 0, "context_length": 8192}, "other"),
        ({"max_completion_tokens": 131072, "context_length": 131072}, None),
        ({"max_completion_tokens": False}, None),
        ({"architecture": {}, "max_completion_tokens": 0}, "other"),
        # Nothing said
        ({}, None),
        ({"type": "something-new"}, None),
        ({"metadata": {"tags": []}}, None),
    ],
)
def test_a_listing_entry_names_the_kind_in_its_providers_own_field(
    fields: dict[str, Any], expected: str | None
) -> None:
    assert type_from_listing(fields) == expected


def test_the_providers_listing_wins_over_models_dev() -> None:
    _listed("together", "BAAI/bge-x", type="embedding")
    models_dev = parse_catalog(
        {"togetherai": {"models": {"BAAI/bge-x": {"modalities": {"input": ["text"], "output": ["text"]}}}}}
    )

    assert model_type(GatewayConfig(), models_dev, "together:BAAI/bge-x") == "embedding"


def test_models_dev_names_an_embedding_model_by_its_family_not_its_text_output() -> None:
    models_dev = parse_catalog(
        {
            "openai": {
                "models": {
                    "text-embedding-3-small": {
                        "family": "text-embedding",
                        "modalities": {"input": ["text"], "output": ["text"]},
                    },
                    "gpt-4o": {"family": "gpt", "modalities": {"input": ["text", "image"], "output": ["text"]}},
                }
            }
        }
    )

    assert model_type(GatewayConfig(), models_dev, "openai:text-embedding-3-small") == "embedding"
    assert model_type(GatewayConfig(), models_dev, "openai:gpt-4o") == "chat"
    assert model_type(GatewayConfig(), models_dev, "openai:unknown") is None


def test_a_chat_request_is_refused_only_when_every_candidate_cannot_chat() -> None:
    _listed("deepinfra", "BAAI/bge-m3", metadata={"tags": ["embed"]})
    get_model_cache().set(
        "together",
        [
            Model.model_validate(
                {"id": "BAAI/bge-m3", "object": "model", "created": 0, "owned_by": "together", "type": "embedding"}
            ),
            Model.model_validate(
                {"id": "Qwen/Qwen3-8B", "object": "model", "created": 0, "owned_by": "together", "type": "chat"}
            ),
        ],
    )
    config = GatewayConfig()

    detail = not_chat_detail(config, None, ["deepinfra:BAAI/bge-m3", "together:BAAI/bge-m3"], "bge-m3")
    assert detail == "'bge-m3' is an embedding model, which cannot serve a chat request. Send it to /api/v1/embeddings."
    assert not_chat_detail(config, None, ["deepinfra:BAAI/bge-m3", "together:Qwen/Qwen3-8B"], "mixed") is None
    # A candidate of unknown kind may chat, so the request goes through.
    assert not_chat_detail(config, None, ["deepinfra:BAAI/bge-m3", "ovhcloud:bge-m3"], "bge-m3") is None
    assert not_chat_detail(config, None, [], "nothing") is None


def test_an_embeddings_request_is_refused_only_when_the_provider_lists_another_kind() -> None:
    _listed("together", "Qwen/Qwen3-8B", type="chat")

    assert not_embedding_detail("together", "Qwen/Qwen3-8B", "qwen3-8b") == (
        "'qwen3-8b' is a chat model, which cannot create embeddings."
    )
    # models.dev is never asked: it gives an embedding model text output as it does a chat model.
    assert not_embedding_detail("openai", "text-embedding-3-small", "text-embedding-3-small") is None


def test_a_failed_listing_forgets_the_kinds_it_held() -> None:
    _listed("together", "BAAI/bge-x", type="embedding")
    get_model_cache().clear("together")

    assert get_model_cache().listed_model("together", "BAAI/bge-x") is None


def test_a_model_known_only_as_no_chat_model_is_refused_chat_and_allowed_embeddings() -> None:
    _listed("ovhcloud", "bge-multilingual-gemma2", max_completion_tokens=0)

    assert not_chat_detail(GatewayConfig(), None, ["ovhcloud:bge-multilingual-gemma2"], "bge-multilingual-gemma2") == (
        "'bge-multilingual-gemma2' is not a chat model, so it cannot serve a chat request."
    )
    assert not_embedding_detail("ovhcloud", "bge-multilingual-gemma2", "bge-multilingual-gemma2") is None


def test_a_transcription_model_is_refused_chat_and_sent_to_transcriptions() -> None:
    _listed("deepinfra", "openai/whisper-large-v3", metadata={"tags": ["stt"]})
    _listed("together", "meta-llama/Llama-Guard-4-12B", type="moderation")
    config = GatewayConfig()

    assert not_chat_detail(config, None, ["deepinfra:openai/whisper-large-v3"], "whisper") == (
        "'whisper' is a transcription model, which cannot serve a chat request. "
        "Send it to /api/v1/audio/transcriptions."
    )
    # A moderation model such as Llama Guard is called through chat completions.
    assert not_chat_detail(config, None, ["together:meta-llama/Llama-Guard-4-12B"], "guard") is None
