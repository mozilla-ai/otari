"""The container vocabulary a request speaks, and which of it names a container on the provider's account."""

from __future__ import annotations

import pytest

from gateway.services.code_execution import CONTAINER_AUTO, names_provider_container, requested_container


def test_what_a_container_field_asks_for_is_read_in_both_vocabularies() -> None:
    """An id resumes one, ``auto`` asks for one, and nothing asks for nothing.

    OpenAI spells the ask as the object ``{"type": "auto"}`` on a
    ``code_interpreter`` entry, which is why the string spelling is not the only
    one: the dialects with no object form (Anthropic's top-level field, the
    gateway's own entry) use ``"auto"``.
    """
    assert requested_container("otari_cntr_1") == "otari_cntr_1"
    assert requested_container({"id": "otari_cntr_2"}) == "otari_cntr_2"
    assert requested_container("auto") == CONTAINER_AUTO
    assert requested_container(" AUTO ") == CONTAINER_AUTO, "case and padding are the client's, not the meaning"
    assert requested_container({"type": "auto"}) == CONTAINER_AUTO
    # An object naming an id means that id, whatever its type says.
    assert requested_container({"type": "auto", "id": "otari_cntr_4"}) == "otari_cntr_4"

    # Nothing asked for: the request holds no sandbox past itself.
    assert requested_container(None) is None
    assert requested_container("") is None
    assert requested_container("   ") is None
    assert requested_container({}) is None
    assert requested_container({"type": "something_else"}) is None
    assert requested_container(7) is None


@pytest.mark.parametrize(
    "call_kwargs",
    [
        {"container": "container_abc"},
        {"tools": [{"type": "code_interpreter", "container": "cntr_abc"}]},
        {"tools": [{"type": "code_interpreter", "container": {"id": "cntr_abc"}}]},
    ],
    ids=["anthropic-id", "openai-id", "openai-object-id"],
)
def test_a_providers_container_id_is_named(call_kwargs: dict[str, object]) -> None:
    assert names_provider_container(call_kwargs) is True


@pytest.mark.parametrize(
    "call_kwargs",
    [
        {},
        {"container": None},
        {"container": "auto"},
        {"container": "otari_cntr_1"},
        {"tools": [{"type": "code_interpreter", "container": {"type": "auto"}}]},
        {"tools": [{"type": "code_interpreter", "container": "otari_cntr_1"}]},
        {"tools": [{"type": "function", "container": "cntr_abc"}]},
    ],
    ids=["none", "null", "auto", "gateway-id", "openai-auto", "openai-gateway-id", "not-code-interpreter"],
)
def test_nothing_else_names_a_providers_container(call_kwargs: dict[str, object]) -> None:
    assert names_provider_container(call_kwargs) is False
