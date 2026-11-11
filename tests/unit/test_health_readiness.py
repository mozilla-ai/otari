"""The readiness probe logs a database outage when it starts and ends, not on every probe."""

from collections.abc import Iterator
from typing import Any, cast

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import OperationalError

from gateway.api.routes import health
from gateway.log_config import logger


class _Database:
    """A session whose ``execute`` raises ``error`` when one is set."""

    def __init__(self) -> None:
        self.error: BaseException | None = None

    async def execute(self, _statement: Any) -> None:
        if self.error is not None:
            raise self.error


@pytest.fixture(autouse=True)
def _no_outage() -> Iterator[None]:
    health._DatabaseOutage.ongoing = False
    yield
    health._DatabaseOutage.ongoing = False


def _capture(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    logged: list[tuple[str, str]] = []
    for level in ("debug", "info", "warning", "error"):
        monkeypatch.setattr(logger, level, lambda message, *args, _level=level: logged.append((_level, message % args)))
    return logged


async def _probe(db: _Database) -> int:
    try:
        await health.health_readiness(cast(Any, db))
    except HTTPException as exc:
        return exc.status_code
    return 200


@pytest.mark.asyncio
async def test_an_outage_is_logged_once_when_it_starts_and_once_when_it_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    logged = _capture(monkeypatch)
    db = _Database()
    db.error = OperationalError("SELECT 1", {}, Exception("password=hunter2 refused"))

    assert [await _probe(db) for _ in range(3)] == [503, 503, 503]
    db.error = None
    assert [await _probe(db) for _ in range(2)] == [200, 200]

    assert [level for level, _message in logged] == ["warning", "debug", "debug", "info"]
    assert logged[0][1] == "Readiness check: database unavailable (OperationalError Exception)"
    assert "hunter2" not in str(logged)


@pytest.mark.asyncio
async def test_an_error_that_is_not_the_databases_is_not_reported_as_an_outage() -> None:
    db = _Database()
    db.error = RuntimeError("a bug in the check")

    with pytest.raises(RuntimeError):
        await _probe(db)
    assert health._DatabaseOutage.ongoing is False
