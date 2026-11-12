"""A tool refusal after admission gives back the container claim admission took."""

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import OperationalError

from gateway.api.routes import messages
from gateway.api.routes._pipeline import _refuse_after_admission, _ToolRefusal
from gateway.exceptions.api_keys_exceptions import KeyScopeMissingError
from gateway.services.code_execution.containers import ContainerLease

_REFUSAL = KeyScopeMissingError("code:execute", "running code")


def _admitted(*, held: bool, release: AsyncMock) -> Any:
    lease = (
        ContainerLease(
            container_id="cntr_held",
            provider="local",
            provider_session_id="session",
            expires_at=datetime.now(UTC),
            hard_expires_at=datetime.now(UTC),
        )
        if held
        else None
    )
    return SimpleNamespace(container_lease=lease, containers=SimpleNamespace(release=release))


@pytest.mark.asyncio
async def test_a_refusal_gives_back_the_claim_it_holds() -> None:
    release = AsyncMock()

    refusal = await _refuse_after_admission(messages._ADAPTER, _admitted(held=True, release=release), _REFUSAL)

    release.assert_awaited_once_with("cntr_held")
    assert isinstance(refusal, _ToolRefusal)
    assert refusal.error.status_code == 403


@pytest.mark.asyncio
async def test_a_refusal_with_no_claim_releases_nothing() -> None:
    release = AsyncMock()

    await _refuse_after_admission(messages._ADAPTER, _admitted(held=False, release=release), _REFUSAL)

    release.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_release_that_fails_still_refuses(caplog: pytest.LogCaptureFixture) -> None:
    release = AsyncMock(side_effect=OperationalError("UPDATE secret", {"token": "s3cr3t"}, Exception("down")))

    with caplog.at_level(logging.WARNING):
        refusal = await _refuse_after_admission(messages._ADAPTER, _admitted(held=True, release=release), _REFUSAL)

    assert refusal.error.status_code == 403
    assert "could not release container claim cntr_held" in caplog.text
    assert "s3cr3t" not in caplog.text
