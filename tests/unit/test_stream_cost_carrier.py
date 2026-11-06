"""Which streamed chunk is the usage report and the cost carrier, across provider shapes."""

from __future__ import annotations

from any_llm.types.completion import ChatCompletionChunk, ChoiceDelta, ChunkChoice, CompletionUsage

from gateway.services.inference import is_chat_stream_cost_carrier, is_usage_only_chunk

_USAGE = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def _chunk(choices: list[ChunkChoice], usage: CompletionUsage | None = _USAGE) -> ChatCompletionChunk:
    """A streamed chunk with the given choices, reporting ``usage`` (the shared one by default)."""
    return ChatCompletionChunk(
        id="c", choices=choices, created=0, model="m", object="chat.completion.chunk", usage=usage
    )


def _choice(*, content: str | None = None, finish: str | None = None, role: str | None = "assistant") -> ChunkChoice:
    """A choice whose delta holds only the given role and content."""
    return ChunkChoice(delta=ChoiceDelta(role=role, content=content), finish_reason=finish, index=0)  # type: ignore[arg-type]


def test_openai_include_usage_chunk_is_the_carrier() -> None:
    """OpenAI ``include_usage`` sends usage on a chunk with no choices: the report and the carrier."""
    chunk = _chunk([])
    assert is_usage_only_chunk(chunk)
    assert is_chat_stream_cost_carrier(chunk)


def test_bedrock_usage_on_an_empty_choice_is_the_carrier() -> None:
    """any-llm's Bedrock adapter sends usage on a choice with only a role: the report and the carrier."""
    chunk = _chunk([_choice()])
    assert is_usage_only_chunk(chunk)
    assert is_chat_stream_cost_carrier(chunk)


def test_usage_on_the_finish_chunk_is_the_carrier_but_not_usage_only() -> None:
    """Usage on the finish chunk carries the cost, but the chunk is content, so the tool loop must not defer it."""
    chunk = _chunk([_choice(finish="stop")])
    assert not is_usage_only_chunk(chunk)
    assert is_chat_stream_cost_carrier(chunk)


def test_a_content_chunk_with_running_usage_is_neither() -> None:
    """Running usage on a content chunk does not make it the report or the carrier."""
    chunk = _chunk([_choice(content="Hel")])
    assert not is_usage_only_chunk(chunk)
    assert not is_chat_stream_cost_carrier(chunk)


def test_a_chunk_without_usage_is_neither() -> None:
    """A chunk with no usage is never the report or the carrier."""
    chunk = _chunk([_choice()], usage=None)
    assert not is_usage_only_chunk(chunk)
    assert not is_chat_stream_cost_carrier(chunk)


def test_a_usage_chunk_with_any_other_payload_is_not_usage_only() -> None:
    """An OpenAI-compatible endpoint may put audio, or a field this code does not know, next to the usage.
    That chunk carries a payload, so it must reach the client rather than be deferred as a usage report."""
    audio = ChunkChoice.model_validate(
        {"delta": {"role": "assistant", "audio": {"id": "a1", "data": "AAA"}}, "finish_reason": None, "index": 0}
    )
    unknown = ChunkChoice.model_validate(
        {"delta": {"role": "assistant", "some_new_field": "x"}, "finish_reason": None, "index": 0}
    )
    for choice in (audio, unknown):
        chunk = _chunk([choice])
        assert not is_usage_only_chunk(chunk)
        assert not is_chat_stream_cost_carrier(chunk)


def test_an_empty_content_string_still_counts_as_an_empty_choice() -> None:
    """An empty content string is no payload, so the choice still counts as empty."""
    chunk = _chunk([_choice(content="")])
    assert is_usage_only_chunk(chunk)
