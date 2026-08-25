"""The slug a created organization gets, which is derived and never sent.

Pure string work with no database, so it is a unit test: the endpoint that uses
it is covered in `tests/integration/test_tenancy_api.py`.
"""

import re

from gateway.services.tenancy.organization_service import _generated_slug
from gateway.services.tenancy.provisioning_service import DEFAULT_ORGANIZATION_SLUG

_SUFFIXED = re.compile(r"^(?P<stem>[a-z0-9-]*?)-(?P<suffix>[0-9a-f]{8})$")


def _parts(name: str) -> tuple[str, str]:
    match = _SUFFIXED.match(_generated_slug(name))
    assert match is not None, f"{name!r} produced a slug in an unexpected shape"
    return match.group("stem"), match.group("suffix")


def test_the_stem_is_the_name_reduced_to_url_safe_characters() -> None:
    assert _parts("Research & Development")[0] == "research-development"
    assert _parts("  Spaced  Out  ")[0] == "spaced-out"
    assert _parts("ACME")[0] == "acme"


def test_a_name_with_no_url_safe_characters_still_produces_a_slug() -> None:
    """The stem can reduce to nothing, and a slug of only a suffix reads as an error."""
    assert _parts("...")[0] == "organization"
    assert _parts("日本語")[0] == "organization"


def test_the_stem_is_bounded_so_a_long_name_cannot_produce_an_unreadable_slug() -> None:
    stem, _ = _parts("a" * 400)

    assert len(stem) == 64


def test_two_organizations_with_one_name_get_different_slugs() -> None:
    """What lets a name repeat: the slug is unique, the name is not."""
    assert _generated_slug("Research") != _generated_slug("Research")


def test_a_created_slug_is_never_the_one_first_boot_adopts() -> None:
    """``provisioning_service`` adopts ``default``; the suffix is what keeps this off it.

    Without that, an organization somebody named "Default" would be adopted as
    the provisioned one by a deployment whose bootstrap marker had been lost.
    """
    assert _generated_slug("Default") != DEFAULT_ORGANIZATION_SLUG
    assert _generated_slug("default") != DEFAULT_ORGANIZATION_SLUG
