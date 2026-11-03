"""What kind of model a model is: a chat model, an embedding model, and so on."""

from typing import Literal

ModelType = Literal[
    "chat",
    "embedding",
    "image",
    "speech",
    "transcription",
    "rerank",
    "moderation",
    "video",
    "decision",
    "realtime",
    "other",
]

# models.dev's v2 ``type``. ``reranking`` is ``rerank``, the word ``/v1/models``
# already uses. ``audio`` is in the data though not among the values models.dev
# filters by; it is a model that writes audio, as a ``speech`` model does.
_MODELS_DEV_TYPES: dict[str, ModelType] = {
    "chat": "chat",
    "image": "image",
    "video": "video",
    "embedding": "embedding",
    "reranking": "rerank",
    "decision": "decision",
    "transcription": "transcription",
    "speech": "speech",
    "realtime": "realtime",
    "audio": "speech",
}


def models_dev_model_type(value: str | None) -> ModelType | None:
    """The kind models.dev's ``type`` names, ``other`` for one it has no word for, ``None`` when it gives none."""
    if value is None:
        return None
    return _MODELS_DEV_TYPES.get(value, "other")
