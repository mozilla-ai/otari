"""Which Responses call fields continue state stored on the provider's account."""

from __future__ import annotations

import pytest

from gateway.api.routes._pipeline import names_stored_provider_state


@pytest.mark.parametrize(
    ("call_kwargs", "field"),
    [
        ({"previous_response_id": "resp_abc"}, "previous_response_id"),
        ({"conversation": "conv_abc"}, "conversation"),
        ({"conversation": {"id": "conv_abc"}}, "conversation"),
        ({"input_data": [{"type": "item_reference", "id": "msg_abc"}]}, "item_reference"),
        ({"extra_body": {"input": [{"type": "item_reference", "id": "msg_abc"}]}}, "item_reference"),
    ],
    ids=["previous-response", "conversation-id", "conversation-object", "item-reference", "codex-input"],
)
def test_a_stored_id_is_named(call_kwargs: dict[str, object], field: str) -> None:
    assert names_stored_provider_state(call_kwargs) == field


@pytest.mark.parametrize(
    "call_kwargs",
    [
        {},
        {"previous_response_id": None, "conversation": None},
        {"input_data": "Hi"},
        {"input_data": [{"type": "message", "role": "user", "content": "Hi", "id": "msg_abc"}]},
        {"extra_body": {"input": "Hi"}},
    ],
    ids=["none", "null", "text-input", "echoed-item", "codex-text-input"],
)
def test_nothing_else_names_stored_state(call_kwargs: dict[str, object]) -> None:
    assert names_stored_provider_state(call_kwargs) is None
