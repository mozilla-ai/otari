"""What a span was about, sealed at rest under a data key from :class:`ContentKeyPort`.

The one place that knows how content is sealed. A sealed value reads::

    otc1.<wrapper>.<wrapped data key>.<nonce + AES-256-GCM ciphertext>

each part url-safe base64. The data key is made for the session, which the
key source binds (KMS as encryption context), and the ciphertext is
authenticated with the workspace, the trace and the span as associated data.
So a value copied onto another span, another session or another workspace
opens as nothing.

A value without the ``otc1.`` prefix is a bare ``OTARI_SECRET_KEY`` Fernet token,
the form content was stored in before; it opens as it always did.

Each session has data keys of its own, made for its workspace and itself (KMS
binds both as encryption context), so a key that leaked would open one session.
Data keys are cached, so a busy session costs its key source one call every
few minutes rather than one a span: one write key per session, reused for
``_KEY_MAX_AGE_S`` or ``_WRITE_KEY_MAX_USES`` seals, and an LRU of opened
keys, each held for ``_KEY_MAX_AGE_S``. After the key source fails for lack of reach or permission, it is not
asked again for ``_PAUSE_AFTER_FAILURE_S``: content is dropped, or reads back
empty, rather than every batch waiting on a source that is down.
"""

import asyncio
import base64
import binascii
import json
import os
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from time import monotonic
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from gateway.log_config import logger
from gateway.metrics import REGISTRY, Counter
from gateway.ports.content_key_port import ContentKeyError, ContentKeyPort, DataKey
from gateway.services.secret_box import SecretBoxUnavailableError, SecretDecryptionError, decrypt_secret
from gateway.types.traces import SPAN_CONTENT_KEYS, SPAN_CONTENT_MAX_LENGTH

_FORMAT = "otc1"
_NONCE_BYTES = 12
_PURPOSE = "trace-content"

_KEY_MAX_AGE_S = 300.0
_WRITE_KEY_MAX_USES = 10_000
_KEY_CACHE_SIZE = 4_096
_PAUSE_AFTER_FAILURE_S = 30.0
# Well inside the trace writer's batch time limit, so a slow key source leaves
# the batch time to store its spans without their content.
_KEY_SOURCE_BUDGET_S = 1.5
# New data keys one workspace may ask the key source for: a burst, then a steady rate.
# One export can name a thousand sessions, each needing a key of its own, and the key
# source's quota is shared by every workspace; past this a workspace's new sessions are
# stored without content rather than throttling everyone else's capture.
_NEW_KEYS_BURST = 64.0
_NEW_KEYS_PER_S = 1.0
# The session the startup check makes a data key for, which no export can name.
_CHECK_WORKSPACE = uuid.UUID(int=0)
_CHECK_SESSION = "otari-startup-check"
_CHECK_TIMEOUT_S = 10.0

UNSEALED = Counter(
    "gateway_trace_content_unsealed",
    "Spans stored without their content because it could not be sealed, by reason",
    ["reason"],
    registry=REGISTRY,
)
UNOPENED = Counter(
    "gateway_trace_content_unopened",
    "Stored span content that read back empty because it could not be opened, by reason",
    ["reason"],
    registry=REGISTRY,
)


# A session: the workspace it is in and its trace id. Each has data keys of its own.
Session = tuple[uuid.UUID, str]


@dataclass(frozen=True)
class SealItem:
    """One span's content, and the row it is sealed for."""

    content: Mapping[str, str]
    workspace_id: uuid.UUID
    trace_id: str
    span_id: str


@dataclass
class _WriteKey:
    key: DataKey
    made_at: float
    uses: int = 0


class SpanContent:
    """Seals and opens span content under the data keys ``keys`` makes and opens.

    One per process, so its caches serve every request and the trace writer.
    A batch fetches the keys it lacks all at once and waits for them at most
    ``_KEY_SOURCE_BUDGET_S`` in all, so a slow source costs a batch its content
    and never the batch, however many workspaces it spans. Writing and reading
    pause apart: a read that fails does not stop capture, nor a write reads. Expired keys are
    dropped on every call and by a timer of their own, so an idle process holds
    none past its age; dropping a reference cannot make Python overwrite the
    bytes, which stay in memory until reused.
    """

    def __init__(self, keys: ContentKeyPort | None) -> None:
        self._keys = keys
        self._write_keys: OrderedDict[Session, _WriteKey] = OrderedDict()
        self._open_keys: OrderedDict[tuple[str, bytes, Session], tuple[bytes, float]] = OrderedDict()
        self._write_paused_until = 0.0
        self._read_paused_until = 0.0
        self._sweep: asyncio.TimerHandle | None = None
        # Per workspace: new keys it may still ask for, and when that was last topped up.
        self._new_key_allowance: dict[uuid.UUID, tuple[float, float]] = {}
        self._check_failure: ContentKeyError | None = None

    @classmethod
    def disabled(cls) -> "SpanContent":
        """One that seals nothing and opens only legacy values, for work that never reads content."""
        return cls(None)

    def clear(self) -> None:
        """Drop every cached key, as the process stops."""
        self._write_keys.clear()
        self._open_keys.clear()
        self._new_key_allowance.clear()
        if self._sweep is not None:
            self._sweep.cancel()
            self._sweep = None

    def close(self) -> None:
        """Drop every cached key and release the key source, as the process stops."""
        self.clear()
        if self._keys is not None:
            self._keys.close()

    @property
    def sealing(self) -> bool:
        """Whether a key source to seal new content with is configured, and has made a key since any failed check."""
        return self._keys is not None and self._keys.wrapper is not None and self._check_failure is None

    async def check(self) -> ContentKeyError | None:
        """Make a data key and open it again, for a session no export names; why that failed, or None.

        Run at startup, so a key the deployment cannot use is reported then rather
        than only as content that is quietly not kept. Until a key is next made,
        :attr:`sealing` answers no.
        """
        self._check_failure = await self._try_keys()
        return self._check_failure

    async def _try_keys(self) -> ContentKeyError | None:
        keys = self._keys
        if keys is None or keys.wrapper is None:
            return ContentKeyError("unavailable", "no content key is configured")
        context = _context(_CHECK_WORKSPACE, _CHECK_SESSION)
        try:
            key = await asyncio.wait_for(keys.new_data_key(context), timeout=_CHECK_TIMEOUT_S)
            await asyncio.wait_for(keys.open_data_key(key.wrapper, key.wrapped, context), timeout=_CHECK_TIMEOUT_S)
        except ContentKeyError as exc:
            return exc
        except TimeoutError:
            return ContentKeyError("unavailable", "the content key source did not answer in time")
        return None

    async def seal(
        self, content: Mapping[str, str], *, workspace_id: uuid.UUID, trace_id: str, span_id: str
    ) -> str | None:
        """The span's content sealed for its row; None when there is none, or it could not be sealed."""
        [sealed] = await self.seal_many([SealItem(content, workspace_id, trace_id, span_id)])
        return sealed

    async def seal_many(self, items: Sequence[SealItem]) -> list[str | None]:
        """Each item's content sealed for its row, in order; None where there is none or it could not be sealed.

        Each session has data keys of its own. The keys the batch lacks are fetched
        together, under one deadline for the batch; an item whose session's key is
        not in hand by then is not sealed.
        """
        keys = self._keys
        if keys is None or keys.wrapper is None:
            return [None] * len(items)
        self._prune()
        wanted = list(dict.fromkeys((item.workspace_id, item.trace_id) for item in items if item.content))
        held, reason, limited = await self._write_keys_for(keys, wanted) if wanted else ({}, "paused", set())
        sealed: list[str | None] = []
        for item in items:
            session = (item.workspace_id, item.trace_id)
            key = held.get(session) if item.content else None
            if key is not None and (cached := self._write_keys.get(session)) is not None:
                # Counted per seal, so a key is replaced after its seals, not its batches.
                cached.uses += 1
            if key is None:
                if item.content:
                    UNSEALED.labels(reason="limited" if session in limited else reason).inc()
                sealed.append(None)
                continue
            nonce = os.urandom(_NONCE_BYTES)
            ciphertext = AESGCM(key.plaintext).encrypt(
                nonce,
                json.dumps(dict(item.content)).encode(),
                _associated_data(item.workspace_id, item.trace_id, item.span_id),
            )
            sealed.append(".".join((_FORMAT, key.wrapper, _b64(key.wrapped), _b64(nonce + ciphertext))))
        return sealed

    async def _write_keys_for(
        self, keys: ContentKeyPort, sessions: Sequence[Session]
    ) -> tuple[dict[Session, DataKey], str, set[Session]]:
        """A write key for each session that has one cached or gets one in time.

        Also why the others went without, and which went without because their
        workspace had asked for its share of new keys.
        """
        held: dict[Session, DataKey] = {}
        missing: list[Session] = []
        for session in sessions:
            cached = self._write_keys.get(session)
            if cached is not None and cached.uses < _WRITE_KEY_MAX_USES:
                self._write_keys.move_to_end(session)
                held[session] = cached.key
            else:
                missing.append(session)
        if not missing:
            return held, "unavailable", set()
        if self._write_paused():
            return held, "paused", set()
        limited = {session for session in missing if not self._allow_new_key(session[0])}
        missing = [session for session in missing if session not in limited]
        if not missing:
            return held, "limited", limited
        tasks = {asyncio.ensure_future(keys.new_data_key(_context(*session))): session for session in missing}
        done, pending = await asyncio.wait(tasks, timeout=_KEY_SOURCE_BUDGET_S)
        for task in pending:
            # A call already in a thread runs on; collect what it ends with, so nothing is left unretrieved.
            task.add_done_callback(_collect)
            task.cancel()
        failure: ContentKeyError | None = None
        if pending:
            failure = ContentKeyError("unavailable", "the content key source did not answer in time")
        for task in done:
            exc = task.exception()
            if exc is None:
                session = tasks[task]
                held[session] = self._cache_write_key(session, task.result())
            elif isinstance(exc, ContentKeyError):
                failure = failure or exc
            else:
                # Anything else costs the batch its content, never the batch.
                failure = failure or ContentKeyError(
                    "unavailable", f"the content key source failed ({type(exc).__name__})"
                )
        if failure is not None:
            self._failed(failure, "sealed")
            return held, failure.reason, limited
        if held:
            self._check_failure = None
        return held, "unavailable", limited

    def _allow_new_key(self, workspace_id: uuid.UUID) -> bool:
        """Take one of the workspace's new keys, if it has one left."""
        now = monotonic()
        level, at = self._new_key_allowance.get(workspace_id, (_NEW_KEYS_BURST, now))
        level = min(_NEW_KEYS_BURST, level + (now - at) * _NEW_KEYS_PER_S)
        if level < 1.0:
            self._new_key_allowance[workspace_id] = (level, now)
            return False
        self._new_key_allowance[workspace_id] = (level - 1.0, now)
        return True

    def _cache_write_key(self, session: Session, key: DataKey) -> DataKey:
        self._write_keys[session] = _WriteKey(key, made_at=monotonic())
        while len(self._write_keys) > _KEY_CACHE_SIZE:
            self._write_keys.popitem(last=False)
        self._schedule_sweep()
        return key

    async def open(
        self,
        sealed: str | None,
        *,
        workspace_id: uuid.UUID,
        trace_id: str,
        span_id: str,
        keys_opened: dict[tuple[str, bytes, Session], bytes] | None = None,
    ) -> dict[str, str]:
        """The content sealed for this row; empty when none was, or it cannot be opened.

        ``keys_opened``, where given, stands in for the shared cache: the keys this
        one reading opened, so each is asked of the key source afresh, once, and is
        kept by nobody after. Break-glass reads this way, so every opening is a call
        the key source records.
        """
        if not sealed:
            return {}
        if not sealed.startswith(_FORMAT + "."):
            return _open_legacy(sealed, workspace_id=workspace_id, span_id=span_id)
        parts = sealed.split(".")
        try:
            if len(parts) != 4:
                raise ValueError
            _, wrapper, wrapped, body = parts[0], parts[1], _unb64(parts[2]), _unb64(parts[3])
        except (binascii.Error, ValueError):
            UNOPENED.labels(reason="rejected").inc()
            return {}
        self._prune()
        try:
            data_key = await self._open_key(wrapper, wrapped, (workspace_id, trace_id), keys_opened)
        except ContentKeyError as exc:
            self._failed(exc, "opened")
            UNOPENED.labels(reason=exc.reason).inc()
            return {}
        if data_key is None:
            # Not asked while paused, which is no new failure: the pause runs out as set.
            UNOPENED.labels(reason="paused").inc()
            return {}
        try:
            plaintext = AESGCM(data_key).decrypt(
                body[:_NONCE_BYTES], body[_NONCE_BYTES:], _associated_data(workspace_id, trace_id, span_id)
            )
            content = json.loads(plaintext)
        except (InvalidTag, ValueError):
            UNOPENED.labels(reason="rejected").inc()
            return {}
        return _allowed(content)

    async def _open_key(
        self,
        wrapper: str,
        wrapped: bytes,
        session: Session,
        keys_opened: dict[tuple[str, bytes, Session], bytes] | None = None,
    ) -> bytes | None:
        """The opened key; None when it is not cached and the source is paused."""
        if self._keys is None:
            raise ContentKeyError("unavailable", "no content key is configured")
        cache_key = (wrapper, wrapped, session)
        if keys_opened is not None:
            if cache_key in keys_opened:
                return keys_opened[cache_key]
            if self._read_paused():
                return None
            keys_opened[cache_key] = await _within_budget(
                self._keys.open_data_key(wrapper, wrapped, _context(*session))
            )
            return keys_opened[cache_key]
        cached = self._open_keys.get(cache_key)
        if cached is not None:
            self._open_keys.move_to_end(cache_key)
            return cached[0]
        if self._read_paused():
            return None
        data_key = await _within_budget(self._keys.open_data_key(wrapper, wrapped, _context(*session)))
        self._open_keys[cache_key] = (data_key, monotonic())
        while len(self._open_keys) > _KEY_CACHE_SIZE:
            self._open_keys.popitem(last=False)
        self._schedule_sweep()
        return data_key

    def _prune(self) -> None:
        """Drop every key past its age, written or opened."""
        now = monotonic()
        for session in [key for key, held in self._write_keys.items() if now - held.made_at >= _KEY_MAX_AGE_S]:
            del self._write_keys[session]
        for cache_key in [k for k, (_, at) in self._open_keys.items() if now - at >= _KEY_MAX_AGE_S]:
            del self._open_keys[cache_key]
        # A workspace whose allowance has refilled is as good as one never seen.
        refilled = [
            workspace_id
            for workspace_id, (level, at) in self._new_key_allowance.items()
            if level + (now - at) * _NEW_KEYS_PER_S >= _NEW_KEYS_BURST
        ]
        for workspace_id in refilled:
            del self._new_key_allowance[workspace_id]

    def _schedule_sweep(self) -> None:
        """Wake when the oldest held key reaches its age, so none outlives it by more than a sweep's delay."""
        if self._sweep is not None:
            return
        made = [held.made_at for held in self._write_keys.values()] + [at for _, at in self._open_keys.values()]
        if not made:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._sweep = loop.call_later(max(0.0, min(made) + _KEY_MAX_AGE_S - monotonic()), self._run_sweep)

    def _run_sweep(self) -> None:
        self._sweep = None
        self._prune()
        self._schedule_sweep()

    def _write_paused(self) -> bool:
        return monotonic() < self._write_paused_until

    def _read_paused(self) -> bool:
        return monotonic() < self._read_paused_until

    def _failed(self, exc: ContentKeyError, action: str) -> None:
        if exc.reason == "rejected":
            return
        # A key source that is down or refusing would fail every span the same way.
        # Writing and reading pause apart: one bad read must not stop capture.
        until = monotonic() + _PAUSE_AFTER_FAILURE_S
        if action == "sealed":
            self._write_paused_until = until
        else:
            self._read_paused_until = until
        logger.warning(
            "Trace content could not be %s (%s: %s); the key source is not asked again for %.0fs",
            action,
            exc.reason,
            exc,
            _PAUSE_AFTER_FAILURE_S,
        )


def _collect(task: "asyncio.Future[Any]") -> None:
    if not task.cancelled():
        task.exception()


async def _within_budget[T](call: Awaitable[T]) -> T:
    """A key source's answer, or ``unavailable`` once it has taken longer than a batch can spare."""
    try:
        return await asyncio.wait_for(call, timeout=_KEY_SOURCE_BUDGET_S)
    except TimeoutError:
        raise ContentKeyError("unavailable", "the content key source did not answer in time") from None


def _context(workspace_id: uuid.UUID, trace_id: str) -> dict[str, str]:
    """What a data key is bound to: KMS opens it for this session of this workspace and no other."""
    return {"otari:purpose": _PURPOSE, "otari:workspace": str(workspace_id), "otari:session": trace_id}


def _associated_data(workspace_id: uuid.UUID, trace_id: str, span_id: str) -> bytes:
    return json.dumps(
        {"v": 1, "workspace_id": str(workspace_id), "trace_id": trace_id, "span_id": span_id},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _allowed(content: object) -> dict[str, str]:
    if not isinstance(content, dict):
        return {}
    return {
        key: value
        for key, value in content.items()
        if key in SPAN_CONTENT_KEYS and isinstance(value, str) and len(value) <= SPAN_CONTENT_MAX_LENGTH
    }


def _open_legacy(sealed: str, *, workspace_id: uuid.UUID, span_id: str) -> dict[str, str]:
    """A bare Fernet token, as content was sealed before data keys, naming its row in the payload."""
    try:
        payload = json.loads(decrypt_secret(sealed))
    except (SecretBoxUnavailableError, SecretDecryptionError, ValueError):
        UNOPENED.labels(reason="rejected").inc()
        return {}
    if not isinstance(payload, dict) or (payload.get("workspace_id"), payload.get("span_id")) != (
        str(workspace_id),
        span_id,
    ):
        UNOPENED.labels(reason="rejected").inc()
        return {}
    return _allowed(payload.get("content"))
