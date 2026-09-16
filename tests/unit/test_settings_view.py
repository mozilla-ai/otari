"""The settings view is declared on the fields and laid out from them."""

from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from gateway.core.config import GatewayConfig
from gateway.core.settings_view import OMITTED, SECRET, SettingsGroup, Shown, derive_view, view_of

# Every credential ``GatewayConfig`` holds. The view never carries one, and
# ``SECRET`` is what says so on the field: ``OMITTED`` would hide it just as
# well today, which is exactly why the rule needs asserting rather than
# describing. A new credential belongs here on the day it is added.
CREDENTIALS = (
    "master_key",
    "smtp_user",
    "smtp_password",
    "oauth_google_client_secret",
    "oauth_github_client_secret",
    "web_search_provider_api_key",
    "web_search_backend_token",
)


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


@pytest.mark.parametrize("name", CREDENTIALS)
def test_every_credential_is_marked_secret_rather_than_merely_hidden(name: str) -> None:
    """The rule the module docstring states, as something that can fail.

    Both markers hide a field, so a credential marked ``OMITTED`` would be as
    invisible and the distinction would mean nothing. This is what makes
    downgrading one a test failure.
    """
    assert view_of(name, GatewayConfig.model_fields[name]) == SECRET


def test_no_credential_is_shown_in_the_view() -> None:
    """The end the marker protects: none of them reaches the endpoint's roster."""
    shown = {key for _, keys in derive_view(GatewayConfig).shown for key in keys}

    assert not shown & set(CREDENTIALS)
