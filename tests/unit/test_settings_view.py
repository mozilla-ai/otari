"""The settings view is declared on the fields and laid out from them."""

from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from gateway.core.settings_view import OMITTED, SECRET, SettingsGroup, Shown, derive_view, view_of


class _Declared(BaseModel):
    later: Annotated[int, Shown(SettingsGroup.GENERAL)] = 1
    first: Annotated[str, Shown(SettingsGroup.SERVER)] = "a"
    second: Annotated[str, Shown(SettingsGroup.SERVER)] = Field(default="b", min_length=1)
    block: Annotated[dict[str, str], OMITTED] = {}
    token: Annotated[str | None, SECRET] = None


def test_the_layout_follows_group_order_then_declaration_order() -> None:
    layout = derive_view(_Declared)

    assert layout.shown == (("Server & database", ("first", "second")), ("General", ("later",)))
    assert layout.hidden == ("block", "token")


def test_an_empty_group_is_absent_rather_than_empty() -> None:
    labels = [label for label, _ in derive_view(_Declared).shown]
    assert "Files" not in labels


def test_a_field_without_a_view_is_refused() -> None:
    """The point of putting the view on the field: a new setting cannot be forgotten."""

    class _Unmarked(BaseModel):
        marked: Annotated[int, OMITTED] = 0
        forgotten: int = 0

    with pytest.raises(TypeError, match="'forgotten' must carry exactly one"):
        derive_view(_Unmarked)


def test_a_field_with_two_views_is_refused() -> None:
    class _Twice(BaseModel):
        both: Annotated[int, OMITTED, Shown(SettingsGroup.FILES)] = 0

    with pytest.raises(TypeError, match="'both' must carry exactly one"):
        view_of("both", _Twice.model_fields["both"])


def test_other_annotated_metadata_is_left_alone() -> None:
    """Pydantic constraints share the metadata list; only the view is read."""
    assert view_of("second", _Declared.model_fields["second"]) == Shown(SettingsGroup.SERVER)
