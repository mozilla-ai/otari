"""Unit tests for Codex-specific Responses API metadata handling."""

from typing import Any

from any_llm import LLMProvider

from gateway.api.routes.responses import _codex_extra_body, _with_codex_extra_body


def test_codex_extra_body_preserves_responses_payload() -> None:
    payload: list[dict[str, Any]] = [
        {
            "type": "message",
            "role": "developer",
            "content": [
                {
                    "type": "input_text",
                    "text": "Follow these instructions.",
                    "internal_chat_message_metadata_passthrough": {"turn_id": "nested"},
                }
            ],
            "internal_chat_message_metadata_passthrough": {"turn_id": "developer"},
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
            "internal_chat_message_metadata_passthrough": {"turn_id": "user"},
        },
    ]

    clean_input, extra_body = _codex_extra_body(payload, {"session_id": "session_123"})

    assert clean_input == [
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "Follow these instructions."}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
        },
    ]
    assert extra_body == {
        "client_metadata": {"session_id": "session_123"},
        "input": payload,
    }
    assert payload[0]["internal_chat_message_metadata_passthrough"] == {"turn_id": "developer"}


def test_codex_extra_body_is_forwarded_only_to_openai() -> None:
    fields = {
        "input_data": "hi",
        "_codex_extra_body": {"client_metadata": {"session_id": "session_123"}},
    }

    assert _with_codex_extra_body(fields, LLMProvider.OPENAI) == {
        "input_data": "hi",
        "extra_body": {"client_metadata": {"session_id": "session_123"}},
    }
    assert _with_codex_extra_body(fields, LLMProvider.FIREWORKS) == {"input_data": "hi"}
