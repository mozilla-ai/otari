"""A command timeout reaches callers as a database error, not a bare one.

`db_command_timeout` is a client-side ceiling, and asyncpg signals it with a
bare `asyncio.TimeoutError`. SQLAlchemy's asyncpg dialect translates only
asyncpg's own exception classes, so untranslated it would walk straight past
every `except SQLAlchemyError` arm in the codebase: the one that turns a
database failure into a 503 rather than a 500, the one that releases a budget
hold rather than leaking it, and the ones that roll a usage-log write back
rather than dropping it.
"""

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import event
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS, init_db, reset_db, translate_timeout_error


class _Context:
    """The one attribute the translating listener reads."""

    def __init__(self, exc: BaseException) -> None:
        self.original_exception = exc


def _dispatch(exc: BaseException) -> None:
    translate_timeout_error(_Context(exc))


def test_database_errors_covers_both_shapes() -> None:
    # A connect timeout is outside the listener's reach: SQLAlchemy only wraps
    # DBAPI errors raised while opening a connection. Handlers on the request
    # path catch this tuple rather than SQLAlchemyError alone.
    assert SQLAlchemyError in DATABASE_ERRORS
    assert TimeoutError in DATABASE_ERRORS


@pytest.fixture
def engine(tmp_path: Path) -> Any:
    reset_db()
    init_db(
        GatewayConfig(
            database_url=f"sqlite+aiosqlite:///{tmp_path / 'otari.db'}",
            auto_migrate=False,
        )
    )
    from gateway.core import database

    yield database._engine
    reset_db()


def test_the_listener_is_installed_on_every_engine(engine: Any) -> None:
    assert event.contains(engine.sync_engine, "handle_error", translate_timeout_error)


def test_a_timeout_is_reported_as_a_database_error() -> None:
    with pytest.raises(OperationalError, match="db_command_timeout"):
        _dispatch(TimeoutError("command timed out"))


def test_an_ordinary_database_error_is_left_alone() -> None:
    # Already a SQLAlchemyError, and re-wrapping would lose its type: the
    # handlers that distinguish an IntegrityError from an OperationalError
    # would stop being able to.
    original = OperationalError("SELECT 1", {}, Exception("gone"))
    _dispatch(original)  # returns without raising


def test_an_unrelated_error_is_left_alone() -> None:
    _dispatch(ValueError("not a database problem"))
