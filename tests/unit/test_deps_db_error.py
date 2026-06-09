"""Unit tests for auth-lookup DB-error handling in ``gateway.api.deps``.

Regression coverage for issue #106: under SQLite lock contention the auth
lookup query (``db.execute(select(APIKey)...)``) could raise a transient
``SQLAlchemyError``, which propagated as an unhandled HTTP 500. The handler
must instead surface a clean, retryable response (503 with a JSON detail),
never a bare 500, so callers can distinguish "key not found" (401) from
"could not check the key right now" (503).
"""

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, status
from sqlalchemy.exc import OperationalError

from gateway.api.deps import _verify_and_update_api_key
from gateway.auth.models import generate_api_key


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
