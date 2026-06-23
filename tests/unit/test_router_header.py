"""Unit tests for the per-request routing headers (`Otari-Router`, `Otari-Conversation-Id`)."""

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from gateway.api.routes.chat import _conversation_id, _router_task, _routing_opted_out


def _request(headers: dict[str, str]) -> Request:
    """A real Starlette Request carrying just the given headers."""
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "headers": raw})


@pytest.mark.parametrize("value", ["off", "OFF", "Off", "false", "0", "no", "none", "disabled"])
def test_off_values_opt_out(value: str) -> None:
    assert _routing_opted_out(_request({"Otari-Router": value})) is True


@pytest.mark.parametrize("value", ["on", "ON", "true", "1", "yes", "auto", "default"])
def test_on_values_keep_routing(value: str) -> None:
    assert _routing_opted_out(_request({"Otari-Router": value})) is False


def test_absent_header_keeps_routing() -> None:
    assert _routing_opted_out(_request({})) is False


def test_blank_and_whitespace_are_tolerated() -> None:
    # Surrounding whitespace is stripped; "  off  " still opts out.
    assert _routing_opted_out(_request({"Otari-Router": "  off  "})) is True


def test_invalid_value_raises_400() -> None:
    with pytest.raises(HTTPException) as exc:
        _routing_opted_out(_request({"Otari-Router": "maybe"}))
    assert exc.value.status_code == 400
    assert "Otari-Router" in exc.value.detail


def test_conversation_id_is_extracted_and_trimmed() -> None:
    assert _conversation_id(_request({"Otari-Conversation-Id": "  conv-42  "})) == "conv-42"


@pytest.mark.parametrize("headers", [{}, {"Otari-Conversation-Id": ""}, {"Otari-Conversation-Id": "   "}])
def test_absent_or_blank_conversation_id_is_none(headers: dict[str, str]) -> None:
    # A blank id is treated as absent so the router falls back to the opener hash.
    assert _conversation_id(_request(headers)) is None


def test_router_task_is_extracted_and_trimmed() -> None:
    assert _router_task(_request({"Otari-Router-Task": "  support-bot  "})) == "support-bot"


@pytest.mark.parametrize("headers", [{}, {"Otari-Router-Task": ""}, {"Otari-Router-Task": "   "}])
def test_absent_or_blank_router_task_is_none(headers: dict[str, str]) -> None:
    # A blank task is treated as absent so the request routes over the whole pool.
    assert _router_task(_request(headers)) is None
