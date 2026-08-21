"""Per-format request accessors for the Reprise v0 fingerprint (otari-ai#1647).

``client_system_hash`` is a fingerprint input and ``first_user_message_hash`` is
a payload field #1488 conditions a pile's agreement on. Both come off the
caller's request before the gateway rewrites it, and all three wire formats
carry them somewhere else, so the accessors have to agree on one normalized
string: one nightly automation driven through two SDKs must land in one pile,
not two half-sized ones.
"""

from typing import Any

import pytest

from gateway.api.routes import chat, messages, responses
from gateway.api.routes._pipeline import FormatAdapter
from gateway.core.observation import text_hash

_SYSTEM = "You are a release-notes bot."
_OPENING = "Summarize yesterday's merged PRs."
_LATER = "Now group them by area."


def _chat_kwargs(**overrides: Any) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _OPENING},
        ],
        **overrides,
    }


def _messages_kwargs(**overrides: Any) -> dict[str, Any]:
    return {
        "system": _SYSTEM,
        "messages": [{"role": "user", "content": _OPENING}],
        **overrides,
    }


def _responses_kwargs(**overrides: Any) -> dict[str, Any]:
    return {
        "instructions": _SYSTEM,
        "input_data": [{"role": "user", "content": _OPENING}],
        **overrides,
    }


_ADAPTERS: list[tuple[FormatAdapter[Any, Any], dict[str, Any]]] = [
    (chat._ADAPTER, _chat_kwargs()),
    (messages._ADAPTER, _messages_kwargs()),
    (responses._ADAPTER, _responses_kwargs()),
]


def test_one_client_system_hash_across_the_three_formats() -> None:
    hashes = {text_hash(adapter.client_system_prompt(kwargs)) for adapter, kwargs in _ADAPTERS}

    assert len(hashes) == 1


def test_one_first_user_message_hash_across_the_three_formats() -> None:
    hashes = {text_hash(adapter.first_user_message(kwargs)) for adapter, kwargs in _ADAPTERS}

    assert len(hashes) == 1


@pytest.mark.parametrize(
    ("adapter", "kwargs"),
    [
        pytest.param(chat._ADAPTER, _chat_kwargs(), id="chat"),
        pytest.param(messages._ADAPTER, _messages_kwargs(), id="messages"),
        pytest.param(responses._ADAPTER, _responses_kwargs(), id="responses"),
    ],
)
def test_absent_system_prompt_normalizes_like_an_empty_one(
    adapter: FormatAdapter[Any, Any], kwargs: dict[str, Any]
) -> None:
    without = {k: v for k, v in kwargs.items() if k not in {"system", "instructions"}}
    if "messages" in without:
        without["messages"] = [m for m in without["messages"] if m["role"] != "system"]

    assert adapter.client_system_prompt(without) == ""


def test_system_string_and_single_text_block_agree() -> None:
    as_string = messages._ADAPTER.client_system_prompt(_messages_kwargs())
    as_blocks = messages._ADAPTER.client_system_prompt(
        _messages_kwargs(system=[{"type": "text", "text": _SYSTEM}])
    )

    assert as_string == as_blocks


def test_developer_led_chat_request_agrees_with_a_system_led_one() -> None:
    """Both roles carry the caller's own prompt, so both feed the same hash."""
    as_system = chat._ADAPTER.client_system_prompt(_chat_kwargs())
    as_developer = chat._ADAPTER.client_system_prompt(
        _chat_kwargs(
            messages=[
                {"role": "developer", "content": _SYSTEM},
                {"role": "user", "content": _OPENING},
            ]
        )
    )

    assert as_system == as_developer


def test_cache_control_markers_do_not_change_the_system_hash() -> None:
    """A cache marker moves money, not meaning, so it must not split a pile."""
    plain = messages._ADAPTER.client_system_prompt(
        _messages_kwargs(system=[{"type": "text", "text": _SYSTEM}])
    )
    marked = messages._ADAPTER.client_system_prompt(
        _messages_kwargs(
            system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}]
        )
    )

    assert plain == marked


def test_multi_block_system_prompt_joins_on_a_blank_line() -> None:
    prompt = messages._ADAPTER.client_system_prompt(
        _messages_kwargs(
            system=[
                {"type": "text", "text": "You are a release-notes bot."},
                {"type": "text", "text": "Answer in bullet points.", "cache_control": {"type": "ephemeral"}},
            ]
        )
    )

    assert prompt == "You are a release-notes bot.\n\nAnswer in bullet points."


@pytest.mark.parametrize(
    ("adapter", "with_later_turn"),
    [
        pytest.param(
            chat._ADAPTER,
            _chat_kwargs(
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _OPENING},
                    {"role": "assistant", "content": "Here they are."},
                    {"role": "user", "content": _LATER},
                ]
            ),
            id="chat",
        ),
        pytest.param(
            messages._ADAPTER,
            _messages_kwargs(
                messages=[
                    {"role": "user", "content": _OPENING},
                    {"role": "assistant", "content": "Here they are."},
                    {"role": "user", "content": _LATER},
                ]
            ),
            id="messages",
        ),
        pytest.param(
            responses._ADAPTER,
            _responses_kwargs(
                input_data=[
                    {"role": "user", "content": _OPENING},
                    {"role": "assistant", "content": "Here they are."},
                    {"role": "user", "content": _LATER},
                ]
            ),
            id="responses",
        ),
    ],
)
def test_a_later_turn_does_not_change_the_first_user_message(
    adapter: FormatAdapter[Any, Any], with_later_turn: dict[str, Any]
) -> None:
    assert adapter.first_user_message(with_later_turn) == _OPENING


def test_responses_accepts_a_bare_string_input() -> None:
    """The Responses ``input`` is a string or a list of items; both are the opening turn."""
    assert responses._ADAPTER.first_user_message(_responses_kwargs(input_data=_OPENING)) == _OPENING


def test_responses_flattens_input_text_parts() -> None:
    assert (
        responses._ADAPTER.first_user_message(
            _responses_kwargs(
                input_data=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": _OPENING}],
                    }
                ]
            )
        )
        == _OPENING
    )


@pytest.mark.parametrize(
    ("adapter", "kwargs"),
    [
        pytest.param(chat._ADAPTER, _chat_kwargs(messages=[]), id="chat"),
        pytest.param(messages._ADAPTER, _messages_kwargs(messages=[]), id="messages"),
        pytest.param(responses._ADAPTER, _responses_kwargs(input_data=[]), id="responses"),
    ],
)
def test_a_request_with_no_user_turn_normalizes_to_empty(
    adapter: FormatAdapter[Any, Any], kwargs: dict[str, Any]
) -> None:
    assert adapter.first_user_message(kwargs) == ""
