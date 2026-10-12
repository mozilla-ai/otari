"""Which provider-specific embedding options a dispatch may carry."""

import pytest
from any_llm import LLMProvider
from fastapi import status

from gateway.exceptions.inference_exceptions import EmbeddingOptionUnsupportedError
from gateway.services.inference import InferenceService

SERVICE = InferenceService


@pytest.mark.parametrize("provider", [LLMProvider.COHERE, LLMProvider.VOYAGE])
def test_input_type_is_forwarded_to_a_provider_that_reads_it(provider: LLMProvider) -> None:
    assert SERVICE.embedding_provider_options(provider, selector="m", input_type="search_query") == {
        "input_type": "search_query"
    }


@pytest.mark.parametrize("provider", [LLMProvider.COHERE, LLMProvider.OPENAI])
def test_no_option_sent_adds_nothing(provider: LLMProvider) -> None:
    assert SERVICE.embedding_provider_options(provider, selector="m", input_type=None) == {}


def test_input_type_for_another_provider_is_a_400_naming_the_requested_selector() -> None:
    with pytest.raises(EmbeddingOptionUnsupportedError) as caught:
        SERVICE.embedding_provider_options(LLMProvider.OPENAI, selector="my-alias", input_type="search_query")
    assert caught.value.status_code == status.HTTP_400_BAD_REQUEST
    assert caught.value.message == "'input_type' is not supported for model 'my-alias'"
