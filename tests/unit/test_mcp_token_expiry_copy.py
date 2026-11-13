"""A token read keeps the expiry it read before a renewal released the session that loaded it."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from sqlalchemy.exc import OperationalError

from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.tools import McpUserConnectionRepository
from gateway.services.secret_box import encrypt_secret, generate_secret_key
from gateway.services.tools import McpServerRef, McpUserTokens

pytestmark = pytest.mark.asyncio


class _Row:
    """A loaded connection whose attributes, like an expired ORM row's, cannot be read once its session let go."""

    def __init__(self, *, expires_in: timedelta) -> None:
        now = datetime.now(UTC)
        self._fields = {
            "encrypted_access_token": encrypt_secret("still-good"),
            "expires_at": now + expires_in,
            "updated_at": now - timedelta(hours=1),
            "created_at": now - timedelta(hours=1),
        }
        self.released = False

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name == "released":
            raise AttributeError(name)
        if self.released:
            raise RuntimeError("read after the session was released")
        return self._fields[name]


class _Connections:
    def __init__(self, row: _Row) -> None:
        self.row = row

    async def find(self, server_id: uuid.UUID, user_id: str, *, lock: bool = False) -> _Row:
        return self.row

    async def find_shared(self, server_id: uuid.UUID, *, lock: bool = False) -> _Row:
        return self.row


@asynccontextmanager
async def _unavailable() -> AsyncIterator[UnitOfWork]:
    raise OperationalError("BEGIN", {}, Exception("connection refused"))
    yield  # pragma: no cover


@pytest.mark.parametrize("shared", [False, True])
async def test_a_failed_renewal_falls_back_to_a_token_still_inside_its_lifetime(
    monkeypatch: pytest.MonkeyPatch, shared: bool
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    row = _Row(expires_in=timedelta(minutes=2))

    async def release() -> None:
        row.released = True

    tokens = McpUserTokens(
        cast(McpUserConnectionRepository, _Connections(row)), unit_of_work=_unavailable, release=release
    )
    server = McpServerRef(uuid.uuid4(), "drive", None)

    token = await (tokens.shared_token_for(server) if shared else tokens.token_for(server, "u"))

    assert token == "still-good"
    assert row.released
