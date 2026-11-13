"""A container claim taken at admission is given back only where no admitted candidate runs in it."""

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import OperationalError

from gateway.api.routes import messages
from gateway.api.routes._pipeline import (
    _DeclaredClaims,
    _refuse_after_admission,
    _release_unused_claims,
    _ToolRefusal,
)
from gateway.exceptions.tools_exceptions import WebToolDeclarationError
from gateway.services.code_execution.containers import ContainerLease

# What a candidate is refused with after its admission: the declaration this
# deployment cannot serve. A missing scope is refused before any admission.
_REFUSAL = WebToolDeclarationError("this deployment serves no web search")


def _admitted(release: AsyncMock, *, held: bool = True) -> Any:
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


def _refused(admitted: Any) -> _ToolRefusal:
    return _refuse_after_admission(messages._ADAPTER, admitted, _REFUSAL)


def test_a_refusal_keeps_its_claim_for_the_caller_to_settle() -> None:
    release = AsyncMock()
    admitted = _admitted(release)

    refusal = _refused(admitted)

    assert refusal.error.status_code == 400
    assert refusal.admitted is admitted
    release.assert_not_called()


@pytest.mark.asyncio
async def test_a_claim_only_refused_candidates_took_is_given_back_once() -> None:
    release = AsyncMock()
    admitted = _admitted(release)

    await _release_unused_claims([_refused(admitted), _refused(admitted)])

    release.assert_awaited_once_with("cntr_held")


@pytest.mark.asyncio
async def test_a_claim_an_admitted_candidate_runs_in_stays_held() -> None:
    """The head runs in the sandbox while a fallback is refused: the shared claim must stay."""
    release = AsyncMock()
    admitted = _admitted(release)
    head = _DeclaredClaims(code=admitted, web=SimpleNamespace())  # type: ignore[arg-type]

    await _release_unused_claims([head, _refused(admitted)])
    await _release_unused_claims([_refused(admitted), head])

    release.assert_not_called()


@pytest.mark.asyncio
async def test_a_refusal_with_no_claim_releases_nothing() -> None:
    release = AsyncMock()

    await _release_unused_claims([_refused(_admitted(release, held=False))])

    release.assert_not_called()


@pytest.mark.asyncio
async def test_a_release_that_fails_is_logged_without_its_statement(caplog: pytest.LogCaptureFixture) -> None:
    release = AsyncMock(side_effect=OperationalError("UPDATE secret", {"token": "s3cr3t"}, Exception("down")))

    with caplog.at_level(logging.WARNING):
        await _release_unused_claims([_refused(_admitted(release))])

    assert "could not release container claim cntr_held" in caplog.text
    assert "s3cr3t" not in caplog.text
