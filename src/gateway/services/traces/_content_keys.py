"""Seal captured content before it is queued, and open it for a reader allowed to see it.

Each session's content is sealed with a data key of its own, from
``DataKeyPort``: AES-256-GCM, with the workspace, session and span bound in as
associated data, so ciphertext copied onto another row does not open. The key is
minted when the session's first content is sealed and kept only wrapped, in
``trace_content_keys``; a process holds recent sessions' plaintext keys in a
bounded cache to seal with, and asks the port to unwrap one to read. Deleting a
session's key makes its stored content unreadable through Otari, which is what
expiry and purge do. A database backup holds the wrapped key until it expires.
"""

import asyncio
import json
import os
import uuid
import weakref
from collections import OrderedDict
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field, replace
from datetime import datetime
from time import monotonic
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from gateway.core.database import DATABASE_ERRORS
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.traces_exceptions import TraceContentKeysUnavailableError
from gateway.log_config import logger
from gateway.ports.data_key_port import (
    DataKeyContext,
    DataKeyContextMismatchError,
    DataKeyPort,
    DataKeyUnavailableError,
)
from gateway.ports.trace_storage_port import SealedContent, StoredContent, TraceWrite
from gateway.repositories.traces import TracesRepositories

_NONCE_BYTES = 12
# How long a cached sealing key is trusted before its row is checked again. A purge
# on another worker deletes the row; this is how long that worker may go on sealing
# with it, the same window a capture level change takes to reach every worker.
_RECHECK_S = 30.0
# Longest a seal waits for its key, lock included; past it the content is dropped.
_KEY_DEADLINE_S = 5.0
# Sessions whose plaintext key one process keeps; the least recently used goes first.
_CACHED_SESSIONS = 4096
# Keys an orphan sweep checks per pass.
_SWEEP_BATCH = 500

Session = tuple[uuid.UUID, str]


class ContentUnreadableError(Exception):
    """Content whose key is gone, or that does not open under its own row."""


class ContentKeyStoreError(Exception):
    """A sealing key could not be issued or stored. Carries no key material and no bound parameters."""


def _associated_data(workspace_id: uuid.UUID, trace_id: str, span_id: str) -> bytes:
    return f"{workspace_id}\x00{trace_id}\x00{span_id}".encode()


@dataclass
class _SealingKey:
    plaintext: bytes = field(repr=False)
    wrapped: bytes = field(repr=False)
    checked_at: float


class ContentKeys:
    """Seals and opens span content with one data key per session."""

    def __init__(
        self,
        data_keys: DataKeyPort,
        open_unit_of_work: Callable[[], AbstractAsyncContextManager[UnitOfWork]],
        tables: Callable[[UnitOfWork], TracesRepositories],
        *,
        recheck_s: float = _RECHECK_S,
        deadline_s: float = _KEY_DEADLINE_S,
        cached_sessions: int = _CACHED_SESSIONS,
    ) -> None:
        self._data_keys = data_keys
        self._open = open_unit_of_work
        self._tables = tables
        self._recheck_s = recheck_s
        self._deadline_s = deadline_s
        self._cached_sessions = cached_sessions
        self._current: OrderedDict[Session, _SealingKey] = OrderedDict()
        # One lock per session, kept only while someone holds a reference to it.
        self._locks: weakref.WeakValueDictionary[Session, asyncio.Lock] = weakref.WeakValueDictionary()

    def _lock(self, session: Session) -> asyncio.Lock:
        lock = self._locks.get(session)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session] = lock
        return lock

    def _remember(self, session: Session, key: _SealingKey) -> None:
        self._current[session] = key
        self._current.move_to_end(session)
        while len(self._current) > self._cached_sessions:
            self._current.popitem(last=False)

    async def available(self) -> bool:
        """Whether the key backend can issue a key now."""
        return await self._data_keys.available()

    async def _key_for(self, session: Session) -> _SealingKey:
        cached = self._current.get(session)
        if cached is not None and monotonic() - cached.checked_at < self._recheck_s:
            self._current.move_to_end(session)
            return cached
        try:
            return await asyncio.wait_for(self._locked_key_for(session), self._deadline_s)
        except TimeoutError:
            msg = f"No sealing key within {self._deadline_s}s"
            raise ContentKeyStoreError(msg) from None

    async def _locked_key_for(self, session: Session) -> _SealingKey:
        # Under the session's lock, so concurrent first requests in this process mint one key.
        async with self._lock(session):
            cached = self._current.get(session)
            if cached is not None:
                if monotonic() - cached.checked_at < self._recheck_s:
                    return cached
                # Past its window, the cached key is confirmed against its row: a purge
                # elsewhere deleted it, and a fresh key is then minted. No KMS call.
                row = await self._find(session)
                if row is not None and row.wrapped == cached.wrapped:
                    cached.checked_at = monotonic()
                    return cached
                self._current.pop(session, None)
            key = await self._stored_or_minted(session)
            self._remember(session, key)
            return key

    async def _stored_or_minted(self, session: Session) -> _SealingKey:
        """The session's key: the stored one unwrapped, or a new one stored first by whichever worker wins."""
        workspace_id, trace_id = session
        context = DataKeyContext(workspace_id=workspace_id, session=trace_id)
        row = await self._find(session)
        if row is None:
            fresh = await self._data_keys.generate(context)
            try:
                async with self._open() as uow:
                    tables = self._tables(uow)
                    async with uow:
                        await tables.keys.add_if_absent(
                            workspace_id=workspace_id, trace_id=trace_id, key_ref=fresh.key_ref, wrapped=fresh.wrapped
                        )
                        row = await tables.keys.find(workspace_id, trace_id)
            except DATABASE_ERRORS as exc:
                # Raised bare: the database error's message carries the bound parameters,
                # the wrapped key among them, and the caller logs with a traceback.
                msg = f"The sealing key could not be stored ({type(exc).__name__})"
                raise ContentKeyStoreError(msg) from None
            if row is not None and row.wrapped == fresh.wrapped:
                return _SealingKey(plaintext=fresh.plaintext, wrapped=fresh.wrapped, checked_at=monotonic())
        if row is None:
            msg = "The sealing key was removed while it was being issued"
            raise ContentKeyStoreError(msg)
        plaintext = await self._data_keys.unwrap(row.wrapped, context)
        return _SealingKey(plaintext=plaintext, wrapped=row.wrapped, checked_at=monotonic())

    async def _find(self, session: Session) -> Any:
        async with self._open() as uow:
            tables = self._tables(uow)
            async with uow:
                return await tables.keys.find(*session)

    async def seal(self, write: TraceWrite, content: dict[str, dict[str, str]]) -> TraceWrite:
        """Return ``write`` with each span that has captured content carrying it sealed.

        Raises:
            ContentKeyStoreError: no sealing key was ready in time, or it could not be stored.
            DataKeyUnavailableError: the key backend cannot issue keys.
        """
        if not content:
            return write
        key = await self._key_for((write.workspace_id, write.trace_id))
        sealer = AESGCM(key.plaintext)
        spans = []
        for span in write.spans:
            payload = content.get(span.span_id)
            if payload:
                nonce = os.urandom(_NONCE_BYTES)
                ciphertext = sealer.encrypt(
                    nonce,
                    json.dumps(payload, ensure_ascii=False).encode(),
                    _associated_data(write.workspace_id, write.trace_id, span.span_id),
                )
                span = replace(span, sealed=SealedContent(nonce=nonce, ciphertext=ciphertext))
            spans.append(span)
        return replace(write, spans=tuple(spans))

    async def open(self, stored: StoredContent, *, trace_id: str, span_id: str) -> dict[str, Any]:
        """Decrypt one span's content.

        Raises:
            ContentUnreadableError: its session's key was purged or expired, or it does not open under this row.
            TraceContentKeysUnavailableError: the key backend could not be reached or used.
        """
        row = await self._find((stored.workspace_id, trace_id))
        if row is None:
            raise ContentUnreadableError(span_id)
        context = DataKeyContext(workspace_id=stored.workspace_id, session=trace_id)
        try:
            key = await self._data_keys.unwrap(row.wrapped, context)
        except DataKeyContextMismatchError as exc:
            raise ContentUnreadableError(span_id) from exc
        except DataKeyUnavailableError as exc:
            logger.warning("Trace content key backend unavailable: %s", exc)
            raise TraceContentKeysUnavailableError from None
        except Exception as exc:
            # A backend's own errors (botocore's, say) are types this module cannot name.
            logger.warning("Trace content key backend failed: %s", type(exc).__name__)
            raise TraceContentKeysUnavailableError from None
        try:
            plaintext = AESGCM(key).decrypt(
                stored.sealed.nonce,
                stored.sealed.ciphertext,
                _associated_data(stored.workspace_id, trace_id, span_id),
            )
            value = json.loads(plaintext)
        except (InvalidTag, ValueError) as exc:
            raise ContentUnreadableError(span_id) from exc
        if not isinstance(value, dict):
            raise ContentUnreadableError(span_id)
        return value

    async def destroy_orphaned(self, before: datetime) -> int:
        """Delete keys older than ``before`` whose session is no longer stored.

        Expiry and purge delete a session's key with it; this catches a key minted
        for a session the writer then dropped, which nothing else would delete.
        """
        destroyed = 0
        while True:
            async with self._open() as uow:
                tables = self._tables(uow)
                async with uow:
                    candidates = await tables.keys.created_before(before, limit=_SWEEP_BATCH)
                    gone = set(candidates) - await tables.traces.existing(candidates)
                    destroyed += await tables.keys.delete_sessions(gone)
            if len(candidates) < _SWEEP_BATCH or not gone:
                return destroyed

    async def destroy_workspace(self, workspace_id: uuid.UUID) -> int:
        """Delete every session key a workspace has, and forget them in memory.

        Content sealed before it and written after it is stored unreadable, and
        another worker seals with its cached key for up to ``_RECHECK_S``.
        """
        for session in [session for session in self._current if session[0] == workspace_id]:
            self._current.pop(session, None)
        async with self._open() as uow:
            tables = self._tables(uow)
            async with uow:
                return await tables.keys.delete_for_workspace(workspace_id)
