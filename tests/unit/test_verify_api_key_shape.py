"""Unit tests for shape-independent API key verification (issue #646).

``hash_key`` used to validate the ``gw-`` format before hashing, and
``_verify_and_update_api_key`` turned that ``ValueError`` into a 401. A key
minted by otari-ai (``tk_...``) failed the prefix and charset checks, so a
migrated row could never authenticate even though the two products compute the
same unsalted SHA-256 digest. Verification now hashes whatever was presented and
decides on the row lookup alone; format stays a mint-time concern.
"""

import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, status

from gateway.api.deps import _verify_and_update_api_key
from gateway.metrics import REGISTRY

# A key with the shape otari-ai mints: neither the ``gw-``/``gw_`` prefix nor the
# ``gw[-_][A-Za-z0-9_-]+`` charset the old check enforced.
MIGRATED_KEY = "tk_live.migrated-platform-key-0123456789abcdefghij"


def _platform_digest(api_key: str) -> str:
    """The digest otari-ai stored for ``api_key``, computed without ``hash_key``.

    Standing in for the other product's hasher keeps the row independent of the
    function under test: pre-fix, ``hash_key`` raised on this key, so a fixture
    built through it would fail during setup instead of on the auth behavior the
    test is about.
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def _sample(labels: dict[str, str]) -> float:
    return REGISTRY.get_sample_value("gateway_auth_failures_total", labels) or 0.0


def _db_returning(api_key: Any) -> Any:
    db: Any = AsyncMock()
    lookup_result = MagicMock()
    lookup_result.scalar_one_or_none.return_value = api_key
    db.execute.return_value = lookup_result
    return db


@pytest.mark.asyncio
async def test_migrated_key_authenticates_when_its_hash_is_on_a_row() -> None:
    api_key: Any = SimpleNamespace(
        id="key-migrated",
        key_hash=_platform_digest(MIGRATED_KEY),
        is_active=True,
        expires_at=None,
        # Recent enough that the throttled ``last_used_at`` bump is skipped, so
        # the test never reaches a real session.
        last_used_at=datetime.now(UTC),
    )

    verified = await _verify_and_update_api_key(_db_returning(api_key), MIGRATED_KEY)

    assert verified is api_key


@pytest.mark.asyncio
async def test_unknown_migrated_shape_key_gets_the_ordinary_invalid_key_401() -> None:
    before_invalid_key = _sample({"reason": "invalid_key"})
    before_invalid_format = _sample({"reason": "invalid_format"})

    with pytest.raises(HTTPException) as exc_info:
        await _verify_and_update_api_key(_db_returning(None), MIGRATED_KEY)

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail == "Invalid API key"
    assert _sample({"reason": "invalid_key"}) - before_invalid_key == 1.0
    # The wrong-shape key must not be reported as a format failure any more.
    assert _sample({"reason": "invalid_format"}) == before_invalid_format
