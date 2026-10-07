"""Who may read a session's captured content, and the record every read leaves.

Content is its session's own user's: they read it by default. An organization's
owners and admins read it only in a workspace that lets them, and a platform
operator only by breaking glass with a stated reason. Every read is recorded in
``trace_content_access`` before the content is returned, so a read that cannot
be recorded does not happen.
"""

import uuid

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.traces_exceptions import TraceContentNotFoundError, TraceContentNotYoursError
from gateway.log_config import logger
from gateway.models.traces import ContentReaderKind
from gateway.ports.trace_storage_port import StoredContent, TraceScope, TraceStoragePort
from gateway.repositories.traces import TracesRepositories
from gateway.services.traces._content_keys import ContentKeys, ContentUnreadableError


class ContentAccessService:
    """Opens one span's content for a reader the rules admit, and records the read."""

    def __init__(
        self, uow: UnitOfWork, tables: TracesRepositories, *, store: TraceStoragePort, keys: ContentKeys | None
    ) -> None:
        self._uow = uow
        self._tables = tables
        self._store = store
        self._keys = keys

    async def read_as_member(
        self,
        *,
        reader_id: uuid.UUID,
        visible: TraceScope,
        administered: TraceScope,
        trace_id: str,
        span_id: str,
    ) -> dict[str, str]:
        """Read content as a signed-in member: the session's own user, or an admin where the workspace allows it.

        Raises:
            TraceContentNotFoundError: no content the caller can see, or it no longer opens.
            TraceContentNotYoursError: the caller sees the session but neither owns it nor may read as an admin.
        """
        stored = await self._store.get_content(visible, trace_id, span_id)
        if stored is None:
            raise TraceContentNotFoundError(span_id)
        kind: ContentReaderKind
        if stored.owner_user_id is not None and stored.owner_user_id == str(reader_id):
            kind = "owner"
        elif stored.workspace_id in administered.workspace_ids and await self._admins_may_read(stored.workspace_id):
            kind = "admin"
        else:
            raise TraceContentNotYoursError
        return await self._open_and_record(
            stored, trace_id=trace_id, span_id=span_id, kind=kind, reader=f"user:{reader_id}", reason=None
        )

    async def read_break_glass(self, *, reader: str, reason: str, trace_id: str, span_id: str) -> dict[str, str]:
        """Read content as a platform operator, for a stated reason, in any workspace.

        Raises:
            TraceContentNotFoundError: no such content, or it no longer opens.
        """
        stored = await self._store.get_content(TraceScope.deployment(), trace_id, span_id)
        if stored is None:
            raise TraceContentNotFoundError(span_id)
        logger.warning(
            "Break-glass trace content read workspace_id=%s trace_id=%s span_id=%s reader=%s",
            stored.workspace_id,
            trace_id,
            span_id,
            reader,
        )
        return await self._open_and_record(
            stored, trace_id=trace_id, span_id=span_id, kind="break_glass", reader=reader, reason=reason
        )

    async def _admins_may_read(self, workspace_id: uuid.UUID) -> bool:
        async with self._uow:
            settings = await self._tables.settings.get(workspace_id)
        return settings is not None and settings.admin_content_access

    async def _open_and_record(
        self,
        stored: StoredContent,
        *,
        trace_id: str,
        span_id: str,
        kind: ContentReaderKind,
        reader: str,
        reason: str | None,
    ) -> dict[str, str]:
        if self._keys is None:
            raise TraceContentNotFoundError(span_id)
        try:
            payload = await self._keys.open(stored, trace_id=trace_id, span_id=span_id)
        except ContentUnreadableError as exc:
            raise TraceContentNotFoundError(span_id) from exc
        async with self._uow:
            await self._tables.access.record(
                workspace_id=stored.workspace_id,
                trace_id=trace_id,
                span_id=span_id,
                reader_kind=kind,
                reader=reader,
                reason=reason,
            )
        logger.info(
            "Trace content read workspace_id=%s trace_id=%s span_id=%s reader=%s kind=%s",
            stored.workspace_id,
            trace_id,
            span_id,
            reader,
            kind,
        )
        return {key: str(value) for key, value in payload.items()}
