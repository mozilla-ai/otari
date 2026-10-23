"""A span's content is sealed for its row with OTARI_SECRET_KEY and opens only there."""

import uuid

import pytest

from gateway.services.secret_box import generate_secret_key
from gateway.services.traces._content import open_content, seal_content

_WORKSPACE = uuid.UUID("00000000-0000-0000-0000-000000000001")
_CONTENT = {"command": "cat ~/.aws/credentials"}


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> str:
    value = generate_secret_key()
    monkeypatch.setenv("OTARI_SECRET_KEY", value)
    return value


def test_sealed_content_hides_what_it_holds_and_opens_on_its_row(key: str) -> None:
    sealed = seal_content(_CONTENT, workspace_id=_WORKSPACE, span_id="tool-1")

    assert sealed is not None and "credentials" not in sealed
    assert open_content(sealed, workspace_id=_WORKSPACE, span_id="tool-1") == _CONTENT


@pytest.mark.parametrize(
    ("workspace_id", "span_id"),
    [(_WORKSPACE, "tool-2"), (uuid.UUID("00000000-0000-0000-0000-000000000002"), "tool-1")],
)
def test_content_copied_onto_another_row_opens_as_nothing(key: str, workspace_id: uuid.UUID, span_id: str) -> None:
    sealed = seal_content(_CONTENT, workspace_id=_WORKSPACE, span_id="tool-1")

    assert open_content(sealed, workspace_id=workspace_id, span_id=span_id) == {}


def test_without_a_key_nothing_is_sealed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)

    assert seal_content(_CONTENT, workspace_id=_WORKSPACE, span_id="tool-1") is None


def test_content_opens_under_a_rotated_key_list_and_not_once_its_key_is_gone(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed = seal_content(_CONTENT, workspace_id=_WORKSPACE, span_id="tool-1")
    newer = generate_secret_key()

    monkeypatch.setenv("OTARI_SECRET_KEY", f"{newer},{key}")
    assert open_content(sealed, workspace_id=_WORKSPACE, span_id="tool-1") == _CONTENT
    monkeypatch.setenv("OTARI_SECRET_KEY", newer)
    assert open_content(sealed, workspace_id=_WORKSPACE, span_id="tool-1") == {}


def test_no_content_seals_to_nothing(key: str) -> None:
    assert seal_content({}, workspace_id=_WORKSPACE, span_id="tool-1") is None
    assert open_content(None, workspace_id=_WORKSPACE, span_id="tool-1") == {}
