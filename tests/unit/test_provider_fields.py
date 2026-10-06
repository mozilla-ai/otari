"""Provider extras copied under ``provider_specific_fields``, where LiteLLM-era clients read them."""

import json

from any_llm.types.completion import ChatCompletion, ChatCompletionChunk

from gateway.provider_fields import surface_provider_fields

_CITATIONS = [{"url": "https://example.com", "title": "Example"}]


def _completion(message: dict[str, object]) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "c1",
            "object": "chat.completion",
            "created": 0,
            "model": "exa",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", **message}}],
        }
    )


def test_exa_citations_are_copied_under_provider_specific_fields() -> None:
    result = surface_provider_fields(_completion({"content": "answer", "citations": _CITATIONS}))

    message = json.loads(result.model_dump_json())["choices"][0]["message"]
    assert message["citations"] == _CITATIONS
    assert message["provider_specific_fields"] == {"citations": _CITATIONS}


def test_a_field_the_provider_already_nests_wins() -> None:
    result = surface_provider_fields(
        _completion({"content": "a", "citations": [], "provider_specific_fields": {"citations": _CITATIONS}})
    )

    message = json.loads(result.model_dump_json())["choices"][0]["message"]
    assert message["provider_specific_fields"] == {"citations": _CITATIONS}


def test_a_plain_message_gains_nothing() -> None:
    result = surface_provider_fields(_completion({"content": "a"}))

    assert "provider_specific_fields" not in json.loads(result.model_dump_json())["choices"][0]["message"]


def test_stream_deltas_carry_them_too() -> None:
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "exa",
            "choices": [{"index": 0, "delta": {"content": "a", "citations": _CITATIONS}}],
        }
    )

    delta = json.loads(surface_provider_fields(chunk).model_dump_json())["choices"][0]["delta"]
    assert delta["provider_specific_fields"] == {"citations": _CITATIONS}
