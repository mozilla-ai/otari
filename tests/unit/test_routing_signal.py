"""Unit tests for the request-side inputs a policy's router reads.

The header contract (``Otari-Router`` and ``Otari-Conversation-Id``) is everything
routing reads about a request, so a change here is a change in routing behavior
for every endpoint at once. Also the prompt flattening guardrails read.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from gateway.api.routes._helpers import latest_user_text, routing_opted_out, routing_signal


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


# -- the conversation id ---------------------------------------------------


def test_the_conversation_id_is_extracted_and_trimmed() -> None:
    signal = routing_signal(_request({"Otari-Conversation-Id": "  conv-42  ", "Otari-Router": "off"}))
    assert signal.conversation_id == "conv-42"
    assert signal.opted_out is True


@pytest.mark.parametrize("headers", [{}, {"Otari-Conversation-Id": ""}, {"Otari-Conversation-Id": "   "}])
def test_an_absent_or_blank_conversation_id_is_none(headers: dict[str, str]) -> None:
    # Blank is treated as absent rather than keying on an empty identity.
    assert routing_signal(_request(headers)).conversation_id is None


# -- the prompt text guardrails read ----------------------------------------


def test_the_latest_user_turn_is_read() -> None:
    convo: list[dict[str, Any]] = [
        {"role": "system", "content": "you are terse"},
        {"role": "user", "content": "OPENER"},
        {"role": "assistant", "content": "sure"},
        {"role": "user", "content": "LATEST"},
    ]
    assert latest_user_text(convo) == "LATEST"


def test_content_part_lists_are_flattened() -> None:
    messages = [{"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "image_url"}]}]
    assert latest_user_text(messages) == "part one"


def test_a_conversation_with_no_user_turn_falls_back_to_the_last_message() -> None:
    messages = [{"role": "system", "content": "only a system turn"}]
    assert latest_user_text(messages) == "only a system turn"
