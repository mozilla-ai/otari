"""Whether this deployment checks for new releases, sending a check when one is due, and what the last one heard."""

import os
import random
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from gateway.core.config import GatewayConfig
from gateway.core.settings.update_check import UpdateCheckSettings
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.update_check_exceptions import UpdateCheckSetInConfigError
from gateway.log_config import logger
from gateway.models.update_check import UpdateCheckState
from gateway.repositories.update_check import UpdateCheckStateRepository
from gateway.schemas.update_check import UpdateCheckAdvisory, UpdateCheckAnswer, UpdateCheckReply, UpdateCheckStatus
from gateway.services.update_check._client import UpdateCheckFailedError, send_check
from gateway.services.update_check._schedule import (
    format_version,
    next_daily_check,
    next_retry,
    release_version,
)
from gateway.version import __version__

# How long a claimed check holds the row. Longer than a check takes, so a replica
# that dies mid-check leaves the row to be claimed again after it.
CLAIM_HOLD = timedelta(minutes=10)

SendCheck = Callable[[str, str, uuid.UUID], Awaitable[UpdateCheckReply]]


def preset_answer(config: UpdateCheckSettings) -> bool | None:
    """The answer the deployment's own configuration gives, or None when an operator is to be asked.

    ``DO_NOT_TRACK`` (https://consoledonottrack.com) and ``CI`` answer no, unless ``update_check`` says otherwise,
    so a CI run and an operator who opts out of every tool at once are neither asked nor counted.
    """
    if config.update_check is not None:
        return config.update_check
    if os.environ.get("DO_NOT_TRACK", "").strip().lower() in {"1", "true", "yes"} or os.environ.get("CI"):
        return False
    return None


async def _send(url: str, version: str, install_id: uuid.UUID) -> UpdateCheckReply:
    return await send_check(url, version=version, install_id=install_id)


class UpdateCheckService:
    """Reads and records the operator's answer, and sends the daily check when it is on."""

    def __init__(
        self,
        uow: UnitOfWork,
        state: UpdateCheckStateRepository,
        config: GatewayConfig,
        *,
        version: str = __version__,
        send: SendCheck = _send,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        rng: random.Random | None = None,
    ) -> None:
        self._uow = uow
        self._state = state
        self._config = config
        self._version = version
        self._release = release_version(version)
        self._send = send
        self._clock = clock
        self._rng = rng

    def _enabled(self, row: UpdateCheckState) -> bool | None:
        preset = preset_answer(self._config)
        return preset if preset is not None else row.enabled

    def _status(self, row: UpdateCheckState) -> UpdateCheckStatus:
        preset = preset_answer(self._config)
        enabled = self._enabled(row)
        heard = enabled is True and self._release is not None
        latest = release_version(row.latest_version or "") if heard else None
        # Advisories were matched against the version that sent the check, so an upgrade since makes them stale.
        current = heard and row.checked_version == format_version(self._release or (0, 0, 0))
        return UpdateCheckStatus(
            enabled=enabled,
            answered_by="config" if preset is not None else ("operator" if row.enabled is not None else None),
            current_version=self._version,
            checkable=self._release is not None,
            install_id=str(row.install_id),
            endpoint=self._config.update_check_url,
            last_checked_at=row.last_checked_at if heard else None,
            next_check_at=row.next_check_at if heard else None,
            latest_version=format_version(latest) if latest else None,
            release_url=row.release_url if latest else None,
            update_available=bool(latest and self._release and latest > self._release),
            advisories=[UpdateCheckAdvisory.model_validate(item) for item in row.advisories] if current else [],
        )

    async def get_status(self) -> UpdateCheckStatus:
        async with self._uow:
            return self._status(await self._state.get_or_create())

    async def answer(self, body: UpdateCheckAnswer) -> UpdateCheckStatus:
        """Record an operator's answer. A yes is checked within a poll. Refused when the configuration answers."""
        if preset_answer(self._config) is not None:
            raise UpdateCheckSetInConfigError
        async with self._uow:
            row = await self._state.get_or_create()
            row = await self._state.record_answer(row, enabled=body.enabled, now=self._clock())
            return self._status(row)

    async def check_if_due(self) -> bool:
        """Claim a due check and send it, returning whether one was sent.

        Nothing is sent unless the check is on, and never from a development build.
        The request runs outside any transaction.
        """
        if self._release is None:
            return False
        now = self._clock()
        async with self._uow:
            row = await self._state.get_or_create()
            if self._enabled(row) is not True:
                return False
            install_id = row.install_id
            if row.next_check_at is None:
                # Turned on by configuration rather than by an answer, so no check is due yet:
                # the first one waits for the install's time of day like every later one.
                await self._state.schedule_if_unset(next_daily_check(install_id, now))
                return False
            if await self._state.claim_due_check(now, now + CLAIM_HOLD) is None:
                return False

        version = format_version(self._release)
        try:
            reply = await self._send(self._config.update_check_url, version, install_id)
        except UpdateCheckFailedError as exc:
            async with self._uow:
                row = await self._state.get_or_create()
                retry = next_retry(install_id, now, row.failures + 1, retry_after=exc.retry_after, rng=self._rng)
                await self._state.record_failure(row, next_check_at=retry)
            logger.info("Update check got no answer (%s); retrying at %s", exc, retry.isoformat())
            return True

        async with self._uow:
            row = await self._state.get_or_create()
            await self._state.record_reply(
                row,
                now=now,
                version=version,
                latest_version=reply.latest_version,
                release_url=reply.release_url,
                advisories=[advisory.model_dump() for advisory in reply.advisories],
                next_check_at=next_daily_check(install_id, now),
            )
        self._log_news(reply)
        return True

    def _log_news(self, reply: UpdateCheckReply) -> None:
        if reply.advisories:
            logger.warning(
                "otari %s is affected by %s security advisor%s: %s. See the dashboard's Settings page.",
                self._version,
                len(reply.advisories),
                "y" if len(reply.advisories) == 1 else "ies",
                ", ".join(advisory.id for advisory in reply.advisories),
            )
        latest = release_version(reply.latest_version or "")
        if latest and self._release and latest > self._release:
            logger.info("otari %s is available (running %s)", format_version(latest), self._version)
