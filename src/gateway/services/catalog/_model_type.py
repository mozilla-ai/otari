"""What kind of model a ``provider:model`` is: a chat model, an embedding model, and so on.

The provider's own model listing is read first, because it describes the model
as that provider serves it, then models.dev's v2 ``type``. Each provider says it
in its own field:
Together a ``type``, DeepInfra ``metadata.tags``, DeepSeek ``output_modalities``.
OVHcloud says less: a ``max_completion_tokens`` of 0 on every model that writes
no text, which makes it ``other``, a model that is not a chat model and nothing
more is known of. A listing that says nothing leaves the kind unknown rather
than guessed.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.services.catalog._metadata import metadata_entry
from gateway.services.model_identity import own_providers_for_vendor_slug, vendor_slug
from gateway.types.model_type import ModelType
from gateway.types.models_dev import ModelsDevCatalog

# The kinds a chat completion, a message or a response can never be served by.
# Image, audio and video models are left out: an aggregator such as OpenRouter
# serves some of them through chat completions. So are moderation models, which
# Llama Guard and Qwen3Guard are, and which are called through chat completions,
# and decision and realtime models, which some hosts may serve through them.
NOT_CHAT: frozenset[ModelType] = frozenset({"embedding", "rerank", "transcription", "other"})

# Where a model of a kind no chat endpoint serves is sent instead.
_ENDPOINT_FOR: dict[ModelType, str] = {
    "embedding": "embeddings",
    "rerank": "rerank",
    "transcription": "audio/transcriptions",
}

# Together's ``type``. ``language`` is a base model and ``code`` a code model, both completed as chat.
_TOGETHER_TYPES: dict[str, ModelType] = {
    "chat": "chat",
    "language": "chat",
    "code": "chat",
    "embedding": "embedding",
    "image": "image",
    "audio": "speech",
    "transcribe": "transcription",
    "rerank": "rerank",
    "moderation": "moderation",
    "video": "video",
}

# DeepInfra's ``metadata.tags``, checked in this order after ``chat``, which a vision model carries beside ``vlm``.
_DEEPINFRA_TAGS: tuple[tuple[str, ModelType], ...] = (
    ("embed", "embedding"),
    ("tts", "speech"),
    ("stt", "transcription"),
    ("image-gen", "image"),
    ("video-gen", "video"),
)


def _from_output_modalities(modalities: object) -> ModelType | None:
    if not isinstance(modalities, list):
        return None
    if "embeddings" in modalities or "embedding" in modalities:
        return "embedding"
    if "text" in modalities:
        return "chat"
    return None


def type_from_listing(fields: dict[str, Any]) -> ModelType | None:
    """The kind a provider's listing entry gives a model, or ``None`` when it gives none."""
    kind = fields.get("type")
    if isinstance(kind, str) and kind in _TOGETHER_TYPES:
        return _TOGETHER_TYPES[kind]
    metadata = fields.get("metadata")
    tags = metadata.get("tags") if isinstance(metadata, dict) else None
    if isinstance(tags, list):
        if "chat" in tags:
            return "chat"
        for tag, tagged in _DEEPINFRA_TAGS:
            if tag in tags:
                return tagged
    architecture = fields.get("architecture")
    if isinstance(architecture, dict):
        from_architecture = _from_output_modalities(architecture.get("output_modalities"))
        if from_architecture is not None:
            return from_architecture
    from_modalities = _from_output_modalities(fields.get("output_modalities"))
    if from_modalities is not None:
        return from_modalities
    # OVHcloud: an embedding, speech, transcription or image model may write no completion tokens.
    limit = fields.get("max_completion_tokens")
    if isinstance(limit, int) and not isinstance(limit, bool) and limit == 0:
        return "other"
    return None


def listed_model_type(instance: str, model_id: str) -> ModelType | None:
    """The kind ``instance``'s last model listing gives ``model_id``, or ``None``.

    Read from the discovery cache, so it never dials and knows nothing of a
    provider not listed since this worker started.
    """
    # Imported here: discovery imports provider_kwargs, which imports this package.
    from gateway.services.model_discovery_service import get_model_cache  # noqa: PLC0415

    listed = get_model_cache().listed_model(instance, model_id)
    if listed is None:
        return None
    return type_from_listing(listed.model_extra or {})


def offering_model_type(
    models_dev: ModelsDevCatalog | None, instance: str, provider_type: str, model_id: str
) -> ModelType | None:
    """The kind of ``model_id`` on ``instance``: its listing's word for it, then models.dev's for ``provider_type``."""
    listed = listed_model_type(instance, model_id)
    if listed is not None:
        return listed
    entry = metadata_entry(models_dev, provider_type, model_id)
    return entry.type if entry is not None else None


def model_type(config: GatewayConfig, models_dev: ModelsDevCatalog | None, selector: str) -> ModelType | None:
    """The kind of the ``provider:model`` ``selector``: its listing's word for it, then models.dev's."""
    instance, separator, model_id = selector.partition(":")
    if not separator:
        return None
    return offering_model_type(models_dev, instance, config.provider_instance_type(instance), model_id)


def catalog_model_type(offerings: Sequence[tuple[str, ModelType | None]], vendor: str | None) -> ModelType | None:
    """The kind of a catalog model: the kind most of its offerings give, ``None`` when none gives one.

    ``offerings`` are each offering's provider type and kind, in the catalog's
    order. Offerings can disagree, because one provider's listing may say
    ``embedding`` where another's models.dev entry says ``chat``. A tie goes to
    the vendor's own provider, then to the first offering.
    """
    known = [(provider_type, kind) for provider_type, kind in offerings if kind is not None]
    if not known:
        return None
    counts = Counter(kind for _provider_type, kind in known)
    top = max(counts.values())
    tied = [kind for _provider_type, kind in known if counts[kind] == top]
    own = own_providers_for_vendor_slug(vendor_slug(vendor)) if vendor else frozenset()
    return next((kind for provider_type, kind in known if kind in tied and provider_type in own), tied[0])


def _a(kind: ModelType) -> str:
    return f"an {kind}" if kind[0] in "aeiou" else f"a {kind}"


def not_chat_detail(
    config: GatewayConfig, models_dev: ModelsDevCatalog | None, candidates: Iterable[str], selector: str
) -> str | None:
    """Why a chat request for ``selector`` is refused, or ``None`` when one of its candidates may serve it.

    ``candidates`` are the ``provider:model`` selectors the request would try.
    Refused only when every one is a kind no chat endpoint serves, so a
    candidate of unknown kind, or one plan entry of the right kind, lets the
    request through.
    """
    kinds = [model_type(config, models_dev, candidate) for candidate in candidates]
    refused = [kind for kind in kinds if kind is not None and kind in NOT_CHAT]
    if not refused or len(refused) < len(kinds):
        return None
    kind = refused[0]
    if kind not in _ENDPOINT_FOR:
        return f"'{selector}' is not a chat model, so it cannot serve a chat request."
    return (
        f"'{selector}' is {_a(kind)} model, which cannot serve a chat request. "
        f"Send it to {API_ROOT}/{_ENDPOINT_FOR[kind]}."
    )


def not_embedding_detail(instance: str, model_id: str, selector: str) -> str | None:
    """Why an embeddings request for ``selector`` is refused, or ``None`` unless its provider lists another kind.

    Only the provider's listing is trusted here: models.dev gives an embedding
    model text output, as it does a chat model, so its word would refuse one.
    """
    kind = listed_model_type(instance, model_id)
    # ``other`` may well be an embedding model: OVHcloud says no more of one.
    if kind is None or kind in ("embedding", "other"):
        return None
    return f"'{selector}' is {_a(kind)} model, which cannot create embeddings."
