"""Shared helpers for the tests that walk the Alembic chain."""

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory


def parent_of(revision: str, *, alembic_dir: Path) -> str:
    """The revision immediately below ``revision``, read from the chain itself.

    Derived rather than written down. Every migration that lands on main re-points the
    ``down_revision`` of whatever sat at the end of the chain, so a literal in a test is a second
    place to remember, and the one that goes stale without failing.
    """
    config = Config()
    config.set_main_option("script_location", str(alembic_dir))
    parent = ScriptDirectory.from_config(config).get_revision(revision).down_revision
    assert isinstance(parent, str), f"{revision} should have exactly one parent, got {parent!r}"
    return parent
