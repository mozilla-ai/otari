"""How much of its requests' content a workspace keeps, and who may change that.

Content is off in every workspace until one of its admins turns it on for that
workspace. The deployment's ``trace_content_capture_max`` only limits: it never
turns anything on, and a workspace cannot go past it.
"""

import uuid
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from time import monotonic

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.traces_exceptions import (
    ContentCaptureAboveCeilingError,
    ContentEncryptionNotConfiguredError,
)
from gateway.log_config import logger
from gateway.models.tenancy import User
from gateway.models.traces import CONTENT_CAPTURE_LEVELS
from gateway.ports.trace_storage_port import TraceStoragePort
from gateway.repositories.traces import TracesRepositories
from gateway.services.tenancy.authorization import WorkspaceAccess
from gateway.services.traces._content_keys import ContentKeys
from gateway.types.trace_views import ContentAccessView, TraceSettingsView

_POLICY_TTL_S = 30.0


def lower_of(first: str, second: str) -> str:
    """The lower of two capture levels."""
    return min(first, second, key=CONTENT_CAPTURE_LEVELS.index)


class ContentCapturePolicy:
    """A workspace's effective capture level, as the request path reads it.

    Cached briefly per workspace, so a request does not pay a read for it; a level
    an admin changes takes effect within the cache's lifetime on every worker.
    """

    def __init__(
        self,
        ceiling: str,
        open_unit_of_work: Callable[[], AbstractAsyncContextManager[UnitOfWork]],
        tables: Callable[[UnitOfWork], TracesRepositories],
    ) -> None:
        self._ceiling = ceiling
        self._open = open_unit_of_work
        self._tables = tables
        self._cache: dict[uuid.UUID, tuple[float, str]] = {}

    async def level(self, workspace_id: uuid.UUID) -> str:
        if self._ceiling == "off":
            return "off"
        cached = self._cache.get(workspace_id)
        if cached is not None and monotonic() - cached[0] < _POLICY_TTL_S:
            return cached[1]
        async with self._open() as uow:
            tables = self._tables(uow)
            async with uow:
                stored = await tables.settings.content_capture(workspace_id) or "off"
        level = lower_of(stored, self._ceiling)
        self._cache[workspace_id] = (monotonic(), level)
        return level

    def forget(self, workspace_id: uuid.UUID) -> None:
        self._cache.pop(workspace_id, None)


class TraceSettingsService:
    """Reads and changes a workspace's content capture, for an admin of that workspace."""

    def __init__(
        self,
        uow: UnitOfWork,
        tables: TracesRepositories,
        access: WorkspaceAccess,
        *,
        store: TraceStoragePort,
        keys: ContentKeys,
        policy: ContentCapturePolicy,
        ceiling: str,
    ) -> None:
        self._uow = uow
        self._tables = tables
        self._access = access
        self._store = store
        self._keys = keys
        self._policy = policy
        self._ceiling = ceiling

    async def _manageable(self, user: User, workspace_id: uuid.UUID) -> None:
        workspace = await self._access.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        await self._access.require_workspace_management_access(user=user, workspace=workspace)

    async def get(self, *, user: User, workspace_id: uuid.UUID) -> TraceSettingsView:
        await self._manageable(user, workspace_id)
        return await self._view(workspace_id)

    async def _view(self, workspace_id: uuid.UUID) -> TraceSettingsView:
        async with self._uow:
            row = await self._tables.settings.get(workspace_id)
        stored = row.content_capture if row is not None else "off"
        return TraceSettingsView(
            content_capture=stored,
            effective=lower_of(stored, self._ceiling),
            ceiling=self._ceiling,
            admin_content_access=row.admin_content_access if row is not None else False,
        )

    async def set_admin_content_access(
        self, *, user: User, workspace_id: uuid.UUID, allowed: bool
    ) -> TraceSettingsView:
        """Let the organization's owners and admins read this workspace's content, or stop them. Recorded."""
        await self._manageable(user, workspace_id)
        async with self._uow:
            await self._tables.settings.set_admin_content_access(workspace_id, allowed, updated_by=user.id)
        logger.info("Trace admin content access set workspace_id=%s allowed=%s by=%s", workspace_id, allowed, user.id)
        return await self._view(workspace_id)

    async def content_reads(
        self, *, user: User, workspace_id: uuid.UUID, limit: int, offset: int
    ) -> list[ContentAccessView]:
        """Every recorded read of this workspace's content, newest first, for its admins."""
        await self._manageable(user, workspace_id)
        async with self._uow:
            rows = await self._tables.access.for_workspace(workspace_id, limit=limit, offset=offset)
        return [
            ContentAccessView(
                accessed_at=row.accessed_at,
                trace_id=row.trace_id,
                span_id=row.span_id,
                reader_kind=row.reader_kind,
                reader=row.reader,
                reason=row.reason,
            )
            for row in rows
        ]

    async def set_content_capture(self, *, user: User, workspace_id: uuid.UUID, level: str) -> TraceSettingsView:
        """Change the workspace's level, recording who changed it.

        Raises:
            ContentCaptureAboveCeilingError: the level is above what the deployment permits.
            ContentEncryptionNotConfiguredError: a level other than off, where content cannot be sealed.
        """
        await self._manageable(user, workspace_id)
        if lower_of(level, self._ceiling) != level:
            raise ContentCaptureAboveCeilingError(self._ceiling)
        # Turning capture off never depends on the key backend.
        if level != "off" and not await self._keys.available():
            raise ContentEncryptionNotConfiguredError
        async with self._uow:
            await self._tables.settings.set_content_capture(workspace_id, level, updated_by=user.id)
        self._policy.forget(workspace_id)
        logger.info("Trace content capture set workspace_id=%s level=%s by=%s", workspace_id, level, user.id)
        return await self._view(workspace_id)

    async def purge_content(self, *, user: User, workspace_id: uuid.UUID) -> int:
        """Delete every span's stored content in the workspace and destroy its session keys. Spans stay."""
        await self._manageable(user, workspace_id)
        removed = await self._store.purge_content(workspace_id)
        await self._keys.destroy_workspace(workspace_id)
        logger.info("Trace content purged workspace_id=%s rows=%d by=%s", workspace_id, removed, user.id)
        return removed
