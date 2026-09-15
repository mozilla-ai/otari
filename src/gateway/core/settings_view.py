"""How each gateway setting appears in the operator's settings view.

Every ``GatewayConfig`` field carries exactly one of these in its ``Annotated``
metadata, and the settings endpoint derives its view from them. A field is
shown in a group, kept out of the view, or a secret. Nothing here is a default:
a field with no annotation is an error, so a new setting has to say where it
belongs.

``Omitted`` covers two different facts. A structured block (a dict or a nested
model) cannot be rendered, because the view carries scalars and string lists
only, and has a page of its own instead. A scalar may simply not have been asked
for yet; moving one to ``Shown`` needs no more justification than somebody
wanting to read it. ``Secret`` is a rule rather than a default: the view never
carries a credential, because whether one works is answered by using it, not by
echoing it back to whoever opened the page.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

from pydantic import BaseModel
from pydantic.fields import FieldInfo


class SettingsGroup(StrEnum):
    """The groups of the settings view, in display order."""

    SERVER = "Server & database"
    METERING = "Metering & budgets"
    MODELS = "Models & discovery"
    RATE_LIMITING = "Rate limiting & CORS"
    FILES = "Files"
    VISION = "Vision & file understanding"
    TOOLS = "Tools & network access"
    MAIL = "Email delivery"
    GENERAL = "General"


@dataclass(frozen=True)
class Shown:
    """Shown in the view, under ``group``, in declaration order within it."""

    group: SettingsGroup


@dataclass(frozen=True)
class Omitted:
    """Kept out of the view: a structured block, or a scalar no page has asked for."""


@dataclass(frozen=True)
class Secret:
    """A credential. Never shown."""


SettingsView = Shown | Omitted | Secret

OMITTED = Omitted()
SECRET = Secret()


class SettingsViewLayout(NamedTuple):
    """The view a settings model declares: what is shown, grouped, and what is not."""

    shown: tuple[tuple[str, tuple[str, ...]], ...]
    """Group label and the field names under it, in display order; empty groups are absent."""
    hidden: tuple[str, ...]
    """Every field kept out of the view, omitted and secret alike."""


def view_of(name: str, field: FieldInfo) -> SettingsView:
    """The view annotation a settings field carries.

    Raises:
        TypeError: the field carries no view annotation, or more than one.

    """
    views = [item for item in field.metadata if isinstance(item, Shown | Omitted | Secret)]
    if len(views) != 1:
        raise TypeError(
            f"setting {name!r} must carry exactly one settings view annotation "
            f"(Shown, OMITTED or SECRET), found {len(views)}"
        )
    return views[0]


def derive_view(model: type[BaseModel]) -> SettingsViewLayout:
    """Lay out the settings view a model declares on its fields."""
    grouped: dict[SettingsGroup, list[str]] = {group: [] for group in SettingsGroup}
    hidden: list[str] = []
    for name, field in model.model_fields.items():
        match view_of(name, field):
            case Shown(group=group):
                grouped[group].append(name)
            case Omitted() | Secret():
                hidden.append(name)
    return SettingsViewLayout(
        shown=tuple((group.value, tuple(names)) for group, names in grouped.items() if names),
        hidden=tuple(hidden),
    )
