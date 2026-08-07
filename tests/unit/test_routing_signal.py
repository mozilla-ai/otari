"""Unit tests for the request-side inputs a policy's router reads.

Two layers, both pure: the header contract (``Otari-Router``,
``Otari-Conversation-Id``, ``Otari-Router-Task``) and the prompt-text signals the
endpoints flatten out of their own wire formats. Between them they are everything
a backend sees about a request, so a change in either is a change in routing
behavior for every endpoint at once.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from gateway.api.routes._helpers import (
    conversation_opening_text,
    first_user_text,
    latest_user_text,
    routing_opted_out,
    routing_signal_from_messages,
    routing_signal_from_text,
)


def _request(headers: dict[str, str]) -> Request:
    """A real Starlette Request carrying just the given headers."""
    raw = [(key.lower().encode(), value.encode()) for key, value in headers.items()]
    return Request({"type": "http", "headers": raw})


# -- the Otari-Router opt-out ----------------------------------------------


@pytest.mark.parametrize("value", ["off", "OFF", "Off", "false", "0", "no", "none", "disabled"])
def test_off_values_opt_out(value: str) -> None:
    assert routing_opted_out(_request({"Otari-Router": value})) is True


@pytest.mark.parametrize("value", ["on", "ON", "true", "1", "yes", "auto", "default"])
def test_on_values_keep_routing(value: str) -> None:
    assert routing_opted_out(_request({"Otari-Router": value})) is False


def test_absent_header_keeps_routing() -> None:
    assert routing_opted_out(_request({})) is False


def test_surrounding_whitespace_is_tolerated() -> None:
    assert routing_opted_out(_request({"Otari-Router": "  off  "})) is True


def test_invalid_value_raises_400() -> None:
    # Ignoring an unrecognized value would leave a client believing it had opted
    # out of routing when it had not.
    with pytest.raises(HTTPException) as exc:
        routing_opted_out(_request({"Otari-Router": "maybe"}))
    assert exc.value.status_code == 400
    assert "Otari-Router" in exc.value.detail


# -- conversation id and task partition ------------------------------------


def test_conversation_id_and_task_are_extracted_and_trimmed() -> None:
    signal = routing_signal_from_messages(
        [{"role": "user", "content": "hi"}],
        _request({"Otari-Conversation-Id": "  conv-42  ", "Otari-Router-Task": "  support-bot  "}),
        has_tools=False,
    )
    assert signal.conversation_id == "conv-42"
    assert signal.task_id == "support-bot"


@pytest.mark.parametrize(
    "headers",
    [{}, {"Otari-Conversation-Id": "", "Otari-Router-Task": ""}, {"Otari-Conversation-Id": "   "}],
)
def test_absent_or_blank_headers_are_none(headers: dict[str, str]) -> None:
    # Blank is treated as absent: the router falls back to hashing the opener, and
    # to the user's default pool, rather than keying on an empty identity.
    signal = routing_signal_from_messages([{"role": "user", "content": "hi"}], _request(headers), has_tools=False)
    assert signal.conversation_id is None
    assert signal.task_id is None


# -- the prompt signals ----------------------------------------------------


_CONVO: list[dict[str, Any]] = [
    {"role": "system", "content": "you are terse"},
    {"role": "user", "content": "OPENER"},
    {"role": "assistant", "content": "sure"},
    {"role": "user", "content": "LATEST"},
]


def test_signals_separate_this_turn_from_the_conversation() -> None:
    assert latest_user_text(_CONVO) == "LATEST"
    assert first_user_text(_CONVO) == "OPENER"
    # The anchor stops at the first assistant reply, so it does not grow as the
    # conversation does: appending turns must not change the trace identity.
    assert conversation_opening_text(_CONVO) == "you are terse\nOPENER"


def test_anchor_is_stable_as_turns_are_appended() -> None:
    grown = [*_CONVO, {"role": "assistant", "content": "more"}, {"role": "user", "content": "and more"}]
    assert conversation_opening_text(grown) == conversation_opening_text(_CONVO)


def test_content_part_lists_are_flattened() -> None:
    messages = [{"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "image_url"}]}]
    assert latest_user_text(messages) == "part one"


def test_a_conversation_with_no_user_turn_falls_back_to_the_last_message() -> None:
    messages = [{"role": "system", "content": "only a system turn"}]
    assert first_user_text(messages) == "only a system turn"


def test_continuation_is_detected_from_an_assistant_turn() -> None:
    fresh = routing_signal_from_messages([{"role": "user", "content": "hi"}], _request({}), has_tools=False)
    assert fresh.is_continuation is False
    assert routing_signal_from_messages(_CONVO, _request({}), has_tools=True).is_continuation is True


def test_the_responses_signal_includes_instructions() -> None:
    """`instructions` is part of the task, so it has to be in the signal.

    The same `input` under "answer in one word" and "write a proof" are different
    jobs with different quality bars. Embedding only `input` gave both the same
    routing decision and, under trace-sticky granularity, the same conversation
    identity. Guardrails read `input` alone because they screen what the user sent;
    routing reads what the model was asked to do.
    """
    from types import SimpleNamespace

    from gateway.api.routes.responses import _routing_text

    terse = SimpleNamespace(instructions="Answer in one word.", input="What is 2 plus 2?")
    proof = SimpleNamespace(instructions="Write a rigorous proof.", input="What is 2 plus 2?")
    assert _routing_text(terse) != _routing_text(proof)
    assert "Answer in one word." in _routing_text(terse)
    assert "What is 2 plus 2?" in _routing_text(terse)
    # No instructions is still the input alone, with no stray separator.
    assert _routing_text(SimpleNamespace(instructions=None, input="hi")) == "hi"


def test_text_form_uses_one_blob_for_every_signal() -> None:
    # The responses API has no turn structure to draw an opener from.
    signal = routing_signal_from_text("do the thing", _request({}), has_tools=False)
    assert signal.task_signal == signal.trace_signal == signal.trace_anchor == "do the thing"
    assert signal.is_continuation is False
