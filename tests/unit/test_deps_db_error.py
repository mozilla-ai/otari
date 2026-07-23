"""Unit tests for auth-lookup DB-error handling in ``gateway.api.deps``.

Regression coverage for issue #106: under SQLite lock contention the auth
lookup query (``db.execute(select(APIKey)...)``) could raise a transient
``SQLAlchemyError``, which propagated as an unhandled HTTP 500. The handler
must instead surface a clean, retryable response (503 with a JSON detail),
never a bare 500, so callers can distinguish "key not found" (401) from
"could not check the key right now" (503).
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, status
from sqlalchemy.exc import OperationalError

from gateway.api import deps
from gateway.api.deps import _verify_and_update_api_key, is_valid_master_key
from gateway.auth.models import generate_api_key, hash_key
from gateway.core.config import GatewayConfig
from gateway.services.master_key_service import generate_master_key


@pytest.mark.asyncio
async def test_lookup_db_error_raises_503_not_500() -> None:
    token = generate_api_key()
    db: Any = AsyncMock()
    db.execute.side_effect = OperationalError("SELECT", {}, Exception("database is locked"))

    with pytest.raises(HTTPException) as exc_info:
        await _verify_and_update_api_key(db, token)

    assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert isinstance(exc_info.value.detail, str)
    assert exc_info.value.detail


@pytest.mark.asyncio
async def test_generated_master_key_lookup_db_error_raises_503_not_500() -> None:
    db: Any = AsyncMock()
    db.get.side_effect = OperationalError("SELECT", {}, Exception("database is locked"))

    with pytest.raises(HTTPException) as exc_info:
        await is_valid_master_key(generate_master_key(), GatewayConfig(), db)

    assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert exc_info.value.detail == "Authentication temporarily unavailable, please retry"


@pytest.mark.asyncio
async def test_failed_last_used_bump_logs_warning_and_does_not_poison_request_session() -> None:
    """A failed ``last_used_at`` bump must be logged and must not fail the request.

    The bump runs on a separate short-lived session, so the request session is
    never committed or rolled back by auth, and a ``SQLAlchemyError`` from the
    bump is swallowed after a warning rather than surfaced to the caller.
    """
    token = generate_api_key()
    api_key: Any = SimpleNamespace(
        id="key-abc123",
        key_hash=hash_key(token),
        is_active=True,
        expires_at=None,
        last_used_at=None,
    )

    db: Any = AsyncMock()
    lookup_result = MagicMock()
    lookup_result.scalar_one_or_none.return_value = api_key
    db.execute.return_value = lookup_result

    class _FailingSession:
        async def execute(self, *args: Any, **kwargs: Any) -> None:
            raise OperationalError("UPDATE", {}, Exception("database is locked"))

        async def commit(self) -> None:  # pragma: no cover - not reached
            raise AssertionError("commit should not be reached")

    class _FailingSessionCM:
        async def __aenter__(self) -> "_FailingSession":
            return _FailingSession()

        async def __aexit__(self, *args: Any) -> bool:
            return False

    with (
        patch.object(deps, "create_session", lambda: _FailingSessionCM()),
        patch("gateway.api.deps.logger.warning") as mock_warning,
    ):
        result = await _verify_and_update_api_key(db, token)

    assert result is api_key
    mock_warning.assert_called_once()
    assert "key-abc123" in mock_warning.call_args.args
    db.commit.assert_not_called()
    db.rollback.assert_not_called()
