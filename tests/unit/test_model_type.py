"""What kind of model a provider's listing and models.dev say a model is, and the refusals that follow."""

from collections.abc import Iterator
from typing import Any

import pytest
from any_llm.types.model import Model

from gateway.core.config import GatewayConfig
from gateway.services.catalog import catalog_model_type, model_type, not_chat_detail, not_embedding_detail
from gateway.services.catalog._model_type import type_from_listing
from gateway.services.model_discovery_service import get_model_cache
from gateway.types.models_dev import parse_catalog, parse_catalog_v2


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


@pytest.mark.parametrize(
    ("models_dev_type", "expected"),
    [
        ("chat", "chat"),
        ("image", "image"),
        ("video", "video"),
        ("embedding", "embedding"),
        ("reranking", "rerank"),
        ("decision", "decision"),
        ("transcription", "transcription"),
        ("speech", "speech"),
        ("realtime", "realtime"),
        # Not one of the nine values models.dev filters by, but in its data.
        ("audio", "speech"),
        ("extraction", "other"),
        ("something-new", "other"),
    ],
)
def test_models_dev_names_the_kind_in_its_v2_type(models_dev_type: str, expected: str) -> None:
    # The modalities say chat, as they do for an embedding model, so only the type can say otherwise.
    model = {"type": models_dev_type, "family": "gpt", "modalities": {"input": ["text"], "output": ["text"]}}
    models_dev = parse_catalog_v2({"openai": {"models": {"m": model}}})

    assert model_type(GatewayConfig(), models_dev, "openai:m") == expected
    assert model_type(GatewayConfig(), models_dev, "openai:unknown") is None


def test_a_v2_model_with_no_type_is_a_chat_model() -> None:
    models_dev = parse_catalog_v2({"openai": {"models": {"m": {}}}})

    assert model_type(GatewayConfig(), models_dev, "openai:m") == "chat"


def test_a_v1_model_has_no_kind_from_models_dev() -> None:
    """v1 has no type, and its family and modalities are not read as one."""
    models_dev = parse_catalog(
        {
            "openai": {
                "models": {
                    "text-embedding-3-small": {
                        "family": "text-embedding",
                        "modalities": {"input": ["text"], "output": ["text"]},
                    },
                    "tts-1": {"modalities": {"input": ["text"], "output": ["audio"]}},
                }
            }
        }
    )

    assert model_type(GatewayConfig(), models_dev, "openai:text-embedding-3-small") is None
    assert model_type(GatewayConfig(), models_dev, "openai:tts-1") is None


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


@pytest.mark.parametrize(
    ("offerings", "vendor", "expected"),
    [
        # The kind most offerings give.
        ([("together", "embedding"), ("deepinfra", "embedding"), ("openai", "chat")], "OpenAI", "embedding"),
        # An offering of unknown kind does not vote.
        ([("together", None), ("deepinfra", None), ("openai", "chat")], None, "chat"),
        ([("together", None)], None, None),
        ([], None, None),
        # A tie goes to the vendor's own provider, wherever it is in the list.
        ([("openrouter", "chat"), ("openai", "embedding")], "OpenAI", "embedding"),
        ([("openai", "embedding"), ("openrouter", "chat")], "OpenAI", "embedding"),
        # With no offering from the vendor, the first offering's kind.
        ([("openrouter", "chat"), ("together", "embedding")], "OpenAI", "chat"),
        ([("together", "embedding"), ("openrouter", "chat")], None, "embedding"),
    ],
)
def test_a_catalog_model_has_the_kind_most_of_its_offerings_give(
    offerings: list[tuple[str, str | None]], vendor: str | None, expected: str | None
) -> None:
    assert catalog_model_type(offerings, vendor) == expected  # type: ignore[arg-type]
