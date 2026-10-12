"""Which provider-specific embedding options a dispatch may carry."""

from typing import Any

from any_llm import LLMProvider

from gateway.exceptions.inference_exceptions import EmbeddingOptionUnsupportedError

# The any-llm providers whose embedding call reads ``input_type``. Every other
# provider either ignores the keyword or rejects it as an unexpected argument.
_INPUT_TYPE_PROVIDERS = frozenset({LLMProvider.COHERE, LLMProvider.VOYAGE})


def embedding_provider_options(provider: LLMProvider, *, selector: str, input_type: str | None) -> dict[str, Any]:
    """The provider-specific keywords to add to an embedding call for ``provider``.

    Raises:
        EmbeddingOptionUnsupportedError: an option was sent that ``provider``
            does not take. ``selector`` is the model the caller asked for,
            which the error names.
    """
    if input_type is None:
        return {}
    if provider not in _INPUT_TYPE_PROVIDERS:
        raise EmbeddingOptionUnsupportedError("input_type", selector)
    return {"input_type": input_type}
