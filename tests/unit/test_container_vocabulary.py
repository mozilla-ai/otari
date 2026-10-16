"""The container vocabulary a request speaks, and which of it a managed credential admits."""

from __future__ import annotations

import pytest

from gateway.exceptions.tools_exceptions import ContainerOnManagedCredentialError
from gateway.services.code_execution import CONTAINER_AUTO, check_container_on_credential, requested_container


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


@pytest.mark.parametrize("container", ["otari_cntr_1", "container_abc", {"id": "cntr_1"}, {}, ""])
def test_a_container_id_is_refused_on_a_managed_credential(container: object) -> None:
    with pytest.raises(ContainerOnManagedCredentialError) as refused:
        check_container_on_credential(container, managed_credential=True)

    assert refused.value.status_code == 400


@pytest.mark.parametrize("container", ["auto", " AUTO ", {"type": "auto"}])
def test_auto_passes_on_a_managed_credential(container: object) -> None:
    check_container_on_credential(container, managed_credential=True)


@pytest.mark.parametrize("container", ["otari_cntr_1", "container_abc", {"id": "cntr_1"}, "auto"])
def test_any_container_passes_on_the_callers_own_credential(container: object) -> None:
    check_container_on_credential(container, managed_credential=False)
