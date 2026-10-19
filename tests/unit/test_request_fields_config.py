"""``providers.<instance>.request_fields``: which names an operator may let through, and how they travel."""

import pytest
from pydantic import ValidationError

from gateway.core.config import GatewayConfig
from gateway.services.provider_kwargs import get_provider_kwargs, with_request_fields


def _config(**entry: object) -> GatewayConfig:
    return GatewayConfig(providers={"exa": {"provider_type": "openai", "api_key": "k", **entry}})


def test_a_listed_field_goes_in_extra_body() -> None:
    config = _config(request_fields=["text", "livecrawl"])

    fields = with_request_fields(config, "exa", {"temperature": 0.2}, {"text": True, "other": 1})

    assert fields == {"temperature": 0.2, "extra_body": {"text": True}}


def test_nothing_listed_or_sent_leaves_the_fields_alone() -> None:
    fields = {"temperature": 0.2}

    assert with_request_fields(_config(), "exa", fields, {"text": True}) is fields
    assert with_request_fields(_config(request_fields=["text"]), "exa", fields, {}) is fields
    assert with_request_fields(_config(request_fields=["text"]), "missing", fields, {"text": True}) is fields


@pytest.mark.parametrize("name", ["model", "messages", "stream", "api_key", "api_base", "extra_body", "temperature", "user"])
def test_a_field_the_gateway_sets_or_a_request_carries_cannot_be_listed(name: str) -> None:
    with pytest.raises(ValidationError, match=name):
        _config(request_fields=[name])


def test_request_fields_must_be_a_list_of_names() -> None:
    with pytest.raises(ValidationError, match="list of field names"):
        _config(request_fields="text")


def test_the_list_never_reaches_the_provider_call() -> None:
    from any_llm import LLMProvider

    kwargs = get_provider_kwargs(_config(request_fields=["text"]), LLMProvider.OPENAI, instance="exa")

    assert "request_fields" not in kwargs
