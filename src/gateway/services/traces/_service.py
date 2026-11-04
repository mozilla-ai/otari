"""The traces use cases: storing a session's spans, reading sessions back, and erasing them."""

from __future__ import annotations

import dataclasses
import math
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.traces_exceptions import (
    GuardrailDigestMismatchError,
    GuardrailVersionLimitError,
    RequestTraceNotFoundError,
    TraceAmbiguousError,
    TraceContentShareForbiddenError,
    TraceContentShareNotFoundError,
    TraceNotFoundError,
)
from gateway.log_config import logger
from gateway.models.traces import (
    Trace,
    TraceBreakGlassRead,
    TraceContentRead,
    TraceContentShare,
    TraceGuardrailDefinition,
    TraceSpan,
)
from gateway.repositories.traces import TraceGrowth, TraceKey, TracesRepositories
from gateway.services.traces._collector import CALL_ID_DIGEST, call_id_digest
from gateway.services.traces._content import SealItem, SpanContent
from gateway.services.traces._reading import MAX_SPANS_PER_READ, ContextWindowOf, trace_view
from gateway.types.trace_views import TraceView
from gateway.types.traces import (
    TRACE_BUCKET_DURATION,
    USAGE_OPERATION,
    BreakGlass,
    BreakGlassKind,
    BreakGlassRead,
    ContentAccess,
    ContentRead,
    ContentReader,
    ContentShare,
    GuardrailDefinition,
    SessionOwners,
    ShareAudience,
    SpanRecord,
    TraceBucket,
    TraceBucketGrain,
    TraceDetail,
    TraceFilter,
    TracePage,
    TraceScope,
    TraceSort,
    TraceSummary,
    TraceWrite,
    WriteResult,
)
from otari_agent.domain.definitions import GateDefinition, definition_digest, gates_from_body

# How long after a user's erasure spans taken in for them are still refused.
ERASURE_MARGIN = timedelta(minutes=10)
# The oldest a write may be when it is stored, measured from when it was taken in;
# an older one is refused whole, however long its upload or request stalled.
MAX_WRITE_AGE = timedelta(hours=24)
# How long past the margin an erased user's row is kept: past the oldest write that
# can still be stored, by an hour that absorbs the clock of the replica deleting it.
ERASURE_KEPT_FOR = MAX_WRITE_AGE + timedelta(hours=1)

# How many spans' content one block clears when a workspace's content is deleted.
CLEAR_CONTENT_BATCH = 5_000

# How often one reader's reads through a share are recorded, and how many of a
# share's newest reads its listing carries.
READ_RECORDED_EVERY = timedelta(hours=1)
SHARE_LATEST_READS = 10

# The most buckets one series read returns, each joined against every session it
# overlaps. A dashboard preset asks for at most 60; its "All", by the day, keeps
# the newest two years.
MAX_SERIES_BUCKETS = 720
# Each request is bounded by its schema; this bounds how many a workspace keeps.
MAX_GUARDRAIL_VERSIONS_PER_WORKSPACE = 1_000


class TraceService:
    """Agent traces in this deployment's database, inside a tenant scope.

    A write is idempotent on ``(workspace_id, trace_id, span_id)``: a span repeated
    is counted as a duplicate, and a stored span is never changed. A span whose
    trace already belongs to another user is rejected, never merged. Each call
    settles in a block of its own on the Unit of Work it was built with.
    """

    def __init__(
        self,
        uow: UnitOfWork,
        repositories: TracesRepositories,
        content: SpanContent,
        owners: SessionOwners,
        context_window_of: ContextWindowOf,
    ) -> None:
        self._uow = uow
        self._content = content
        self._owners = owners
        self._context_window_of = context_window_of
        self._traces = repositories.traces
        self._spans = repositories.spans
        self._definitions = repositories.definitions
        self._shares = repositories.shares
        self._barriers = repositories.barriers
        self._erasures = repositories.erasures
        self._break_glass = repositories.break_glass

    async def write(self, traces: Sequence[TraceWrite]) -> WriteResult:
        """Store these traces' spans, creating each trace on its first span.

        A fixed number of statements however many traces the batch holds: lock its
        users' erasure rows (and, where it keeps content, its workspaces'
        generations), create the traces not stored yet, read their owners, insert
        the spans, grow the totals. Each runs in key order, so two writers sharing traces lock them in
        the same order. Where it keeps content, it also asks whether its users are
        still members of their workspaces, all at once.
        """
        written_at = datetime.now(UTC)
        # Sealed before the block, so a key source's round trip holds no connection.
        sealed = await self._seal(traces)
        async with self._uow:
            # First, so a user's erasure on any replica waits for this write to commit
            # and then deletes it, or this write sees the erasure and stores nothing of theirs.
            erased = await self._erasures.lock({trace.user_id for trace in traces if trace.user_id is not None})
            # Then, so a workspace turning content off or deleting it waits for this
            # write to commit, or this write sees the generation it moved on to.
            generations = await self._barriers.lock(
                {trace.workspace_id for trace, held in zip(traces, sealed, strict=True) if held}
            )
            # Asked under the user's lock, which removing a member takes too: content is
            # kept only for someone who is still a member as it is stored.
            members = await self._owners.members(
                {(trace.workspace_id, trace.user_id) for trace, held in zip(traces, sealed, strict=True) if held}
            )
            new_traces: dict[TraceKey, dict[str, Any]] = {}
            for trace in traces:
                # An erased user's write creates nothing, not even the session it names.
                if not _refused(trace, erased, written_at):
                    new_traces.setdefault(_key(trace), _new_trace(trace, written_at))
            await self._traces.create_absent(list(new_traces.values()))
            # Read after creating, so a trace a concurrent writer created first is
            # judged by the owner it actually has.
            holders = await self._traces.holders(new_traces.keys())

            rejected = 0
            kept: list[tuple[TraceWrite, list[SpanRecord]]] = []
            rows: dict[tuple[uuid.UUID, str, str], dict[str, Any]] = {}
            for trace, held in zip(traces, sealed, strict=True):
                spans = _unique_spans(trace.spans)
                holder = holders.get(_key(trace))
                if _refused(trace, erased, written_at) or holder is None or holder[0] != trace.user_id:
                    rejected += len(spans)
                    continue
                kept.append((trace, spans))
                for span in spans:
                    rows.setdefault(
                        (trace.workspace_id, trace.trace_id, span.span_id),
                        # The write whose row this is supplies its content, never another write of the batch.
                        _span_values(
                            trace,
                            span,
                            held
                            if _current(trace, generations) and (trace.workspace_id, trace.user_id) in members
                            else {},
                        ),
                    )
            inserted = await self._spans.insert_new(list(rows.values()))

            accepted = duplicate = 0
            grown: dict[TraceKey, list[SpanRecord]] = defaultdict(list)
            for trace, spans in kept:
                for span in spans:
                    key = (trace.workspace_id, trace.trace_id, span.span_id)
                    if key in inserted:
                        # Claimed once, so a span repeated across this batch's writes counts once.
                        inserted.discard(key)
                        grown[_key(trace)].append(span)
                        accepted += 1
                    else:
                        duplicate += 1
            await self._traces.add_totals([_growth(key, spans, written_at) for key, spans in grown.items()])
        return WriteResult(accepted=accepted, duplicate=duplicate, rejected=rejected)

    async def content_generation(self, workspace_id: uuid.UUID) -> int:
        """The workspace's content generation, to capture content on. Read before whether content is on at all."""
        async with self._uow:
            return await self._barriers.generation(workspace_id)

    async def turn_content_off[T](
        self, workspace_id: uuid.UUID, *, authorize: Callable[[], Awaitable[None]], switch: Callable[[], Awaitable[T]]
    ) -> T:
        """Turn a workspace's content off with ``switch``, so nothing taken in before it is stored after it.

        The order is the rule: ``authorize`` first, so only a manager ends the
        generation; the generation ends before the switch, so a failure there changes
        nothing; and again after it, for an export that read the switch on in between.
        """
        await authorize()
        await self.stop_content(workspace_id)
        result = await switch()
        await self.stop_content(workspace_id)
        return result

    async def stop_content(self, workspace_id: uuid.UUID) -> None:
        """Keep none of the content already captured in the workspace that is not stored yet, on any replica."""
        async with self._uow:
            await self._barriers.advance(workspace_id)

    async def forget_content(self, workspace_id: uuid.UUID) -> int:
        """Delete what every span in the workspace was about, keeping the spans; returns how many held content.

        Content captured before the deletion and not stored yet is never stored. The
        generation moves on first, in a block of its own, so every write holding content
        has committed by the time clearing starts, and none stores any after; clearing
        then goes a batch per block, so no write waits on the whole workspace's.
        """
        async with self._uow:
            await self._barriers.advance(workspace_id)
        cleared = 0
        while True:
            async with self._uow:
                batch = await self._spans.clear_content(workspace_id, limit=CLEAR_CONTENT_BATCH)
            cleared += batch
            if batch < CLEAR_CONTENT_BATCH:
                return cleared

    async def _seal(self, traces: Sequence[TraceWrite]) -> list[dict[str, str]]:
        """Each write's spans' content, sealed for that write's own rows, by span id.

        Per write rather than per row key, so a write that is rejected or stale cannot
        supply the content of a row another write of the batch stores. A write that
        carries no content generation keeps no content, so none is sealed for it.
        """
        items = [
            SealItem(span.content, trace.workspace_id, trace.trace_id, span.span_id)
            for trace in traces
            if trace.content_generation is not None
            for span in trace.spans
            if span.content
        ]
        values = iter(await self._content.seal_many(items))
        sealed: list[dict[str, str]] = []
        for trace in traces:
            held: dict[str, str] = {}
            if trace.content_generation is not None:
                for span in trace.spans:
                    if span.content:
                        value = next(values)
                        if value is not None:
                            held.setdefault(span.span_id, value)
            sealed.append(held)
        return sealed

    async def search(
        self, scope: TraceScope, filters: TraceFilter, *, limit: int, offset: int, sort: TraceSort = TraceSort()
    ) -> TracePage:
        """Return one page of the scope's traces in ``sort`` order, newest activity first by default."""
        async with self._uow:
            rows = await self._traces.page(_workspace_ids(scope), filters, limit=limit + 1, offset=offset, sort=sort)
            return TracePage(items=tuple(_summary(row) for row in rows[:limit]), has_more=len(rows) > limit)

    async def count(self, scope: TraceScope, filters: TraceFilter) -> int:
        """Return how many of the scope's traces match."""
        async with self._uow:
            return await self._traces.count_matching(_workspace_ids(scope), filters)

    async def get(
        self,
        scope: TraceScope,
        trace_id: str,
        *,
        span_limit: int,
        workspace_id: uuid.UUID | None = None,
        break_glass: BreakGlass | None = None,
    ) -> TraceDetail | None:
        """Return one trace with up to ``span_limit`` spans, or None when the scope holds no such trace.

        A trace in another tenant's workspace reads as absent, never as forbidden.

        Its content opens for the user it belongs to, or for a reader its owner
        shared it with, in which case the read is recorded for the owner to see.
        For anyone else a span that holds content says so, and shows none.

        With ``break_glass`` the content opens whoever owns the session, and the
        opening is recorded first, in the same block that read the session: if the
        record cannot be written, nothing opens.

        Raises:
            TraceAmbiguousError: no ``workspace_id`` was given and the id is in more
                than one of the scope's workspaces.
        """
        now = datetime.now(UTC)
        async with self._uow:
            rows = await self._traces.find(_workspace_ids(scope), trace_id, workspace_id=workspace_id)
            if not rows:
                return None
            if len(rows) > 1:
                raise TraceAmbiguousError(trace_id)
            row = rows[0]
            spans = list(await self._spans.for_trace(row.workspace_id, row.trace_id, limit=span_limit + 1))
            records = [_span_record(span) for span in spans[:span_limit]]
            digests = {
                digest for span in records if isinstance(digest := span.attributes.get("otari.policy.digest"), str)
            }
            definitions = await self._definitions.for_digests(row.workspace_id, digests)
            sealed = any(span.content_sealed for span in spans[:span_limit])
            access: ContentAccess = "owner" if scope.reads_content_of(row.user_id) else "none"
            share = None
            if break_glass is not None:
                await self._break_glass.record(
                    kind=break_glass.kind,
                    organization_id=break_glass.organization_id,
                    workspace_id=row.workspace_id,
                    trace_id=row.trace_id,
                    session_user_id=row.user_id,
                    reader_user_id=break_glass.reader_user_id,
                    reader_email=break_glass.reader_email,
                    reason=break_glass.reason,
                    reference=break_glass.reference,
                )
                # Ids only: the reason is the reader's own prose, and stays in the record.
                logger.info(
                    "Break-glass opening of a session's content: kind=%s workspace=%s trace=%s reader=%s",
                    break_glass.kind,
                    row.workspace_id,
                    row.trace_id,
                    break_glass.reader_user_id,
                )
                access = "break_glass"
            elif access == "none" and sealed and scope.reader is not None and scope.reader.audiences:
                share = await self._shares.active(row.workspace_id, row.trace_id, scope.reader.audiences)
                # A share lasts only while its owner belongs: one who left can no longer manage it,
                # so nobody reads through it, and it is ended for good rather than revived on a return.
                if share is not None and not await self._owners.is_member(row.workspace_id, row.user_id):
                    await self._shares.revoke(share, now)
                    share = None
                if share is not None:
                    access = "share"
                    # Once an hour per reader, so a page polling an active session is one read, not one a poll.
                    await self._shares.record_read(
                        share,
                        row.user_id,
                        scope.reader.user_id,
                        scope.reader.email,
                        unless_since=now - READ_RECORDED_EVERY,
                    )
        # Opened after the block, so a key source's round trip holds no connection. Break-glass
        # opens its keys afresh, bypassing the shared cache, so the key source records each opening.
        keys_opened: dict[Any, bytes] | None = {} if access == "break_glass" else None
        for position, span in enumerate(spans[:span_limit]):
            if not span.content_sealed:
                continue
            if access == "none":
                records[position] = dataclasses.replace(records[position], content_private=True)
                continue
            content = await self._content.open(
                span.content_sealed,
                workspace_id=span.workspace_id,
                trace_id=span.trace_id,
                span_id=span.span_id,
                keys_opened=keys_opened,
            )
            records[position] = dataclasses.replace(records[position], content=content)
        return TraceDetail(
            summary=_summary(row),
            spans=tuple(records),
            truncated=len(spans) > span_limit,
            definitions=tuple(_definition(definition) for definition in definitions),
            content_access=access,
            content_share=_share(share) if share is not None else None,
        )

    async def break_glass_detail(
        self, scope: TraceScope, trace_id: str, *, break_glass: BreakGlass, workspace_id: uuid.UUID | None = None
    ) -> TraceView:
        """One session with its turns and its content, opened without its owner's share and recorded first.

        Raises:
            TraceNotFoundError: the scope holds no such trace.
            TraceAmbiguousError: no ``workspace_id`` was given and the id is in more than one workspace.
        """
        stored = await self.get(
            scope, trace_id, span_limit=MAX_SPANS_PER_READ, workspace_id=workspace_id, break_glass=break_glass
        )
        if stored is None:
            raise TraceNotFoundError(trace_id)
        return trace_view(stored, now=datetime.now(UTC))

    async def break_glass_reads(
        self, *, organization_id: uuid.UUID | None, kind: BreakGlassKind | None, limit: int, offset: int = 0
    ) -> tuple[tuple[BreakGlassRead, ...], bool]:
        """A page of the recorded break-glass openings, newest first, and whether more follow."""
        async with self._uow:
            rows = list(
                await self._break_glass.page(organization_id=organization_id, kind=kind, limit=limit + 1, offset=offset)
            )
        return tuple(_break_glass_read(row) for row in rows[:limit]), len(rows) > limit

    async def expire_erasures(self, *, now: datetime) -> int:
        """Forget users erased longer ago than any of their spans could still be waiting, and those never erased."""
        async with self._uow:
            return await self._erasures.forget_erased_before(now - ERASURE_MARGIN - ERASURE_KEPT_FOR)

    async def share_content(
        self,
        scope: TraceScope,
        trace_id: str,
        *,
        audience: ShareAudience,
        note: str | None,
        workspace_id: uuid.UUID | None = None,
    ) -> ContentShare:
        """Let ``audience`` read what the session was about until the share ends. Its own user only.

        Raises:
            TraceNotFoundError: the scope holds no such trace.
            TraceAmbiguousError: no ``workspace_id`` was given and the id is in more than one workspace.
            TraceContentShareForbiddenError: the reader is not the session's own user.
        """
        reader = _owner(scope)
        async with self._uow:
            # Under the lock that ending the owner's membership takes too, and checked
            # again here, so a share is never made by someone whose membership has
            # just ended, however long ago this request resolved their scope. Taken
            # first, so on SQLite the block writes before it reads and waits its turn.
            await self._erasures.lock([str(reader.user_id)])
            row = await self._owned(scope, trace_id, workspace_id)
            if not await self._owners.is_member(row.workspace_id, row.user_id):
                raise TraceContentShareForbiddenError
            share = await self._shares.add(
                workspace_id=row.workspace_id,
                trace_id=row.trace_id,
                audience=audience,
                note=note,
                shared_by_user_id=reader.user_id,
            )
            return _share(share)

    async def end_shares_of(self, user_id: uuid.UUID) -> None:
        """End every share the person made, in every workspace, as their account is deactivated.

        Under the lock making a share takes, so none is made in between; a reactivated
        account gets none of them back.
        """
        async with self._uow:
            await self._erasures.lock([str(user_id)])
            await self._shares.revoke_all_by(user_id, datetime.now(UTC))

    async def content_shares(
        self,
        scope: TraceScope,
        trace_id: str,
        *,
        limit: int,
        offset: int = 0,
        workspace_id: uuid.UUID | None = None,
    ) -> tuple[tuple[ContentShare, ...], bool]:
        """A page of the shares the session has had, newest first, and whether more follow. Its own user only.

        Ended shares are listed too, so their reads stay visible. Each
        carries its newest ``SHARE_LATEST_READS`` reads and how many it has had;
        :meth:`content_share_reads` pages through the rest.

        Raises:
            TraceNotFoundError: the scope holds no such trace.
            TraceAmbiguousError: no ``workspace_id`` was given and the id is in more than one workspace.
            TraceContentShareForbiddenError: the reader is not the session's own user.
        """
        _owner(scope)
        async with self._uow:
            row = await self._owned(scope, trace_id, workspace_id)
            shares = list(await self._shares.for_trace(row.workspace_id, row.trace_id, limit=limit + 1, offset=offset))
            ids = [share.id for share in shares[:limit]]
            reads = await self._shares.latest_reads(ids, per_share=SHARE_LATEST_READS)
            counts = await self._shares.read_counts(ids)
        by_share: dict[uuid.UUID, list[ContentRead]] = defaultdict(list)
        for read in reads:
            if read.share_id is not None:
                by_share[read.share_id].append(_read(read))
        page = tuple(_share(share, tuple(by_share[share.id]), counts.get(share.id, 0)) for share in shares[:limit])
        return page, len(shares) > limit

    async def content_share_reads(
        self,
        scope: TraceScope,
        trace_id: str,
        share_id: uuid.UUID,
        *,
        limit: int,
        offset: int = 0,
        workspace_id: uuid.UUID | None = None,
    ) -> tuple[tuple[ContentRead, ...], bool]:
        """A page of the reads through one of the session's shares, newest first, and whether more follow.

        Raises:
            TraceNotFoundError: the scope holds no such trace.
            TraceContentShareNotFoundError: the session holds no such share.
            TraceContentShareForbiddenError: the reader is not the session's own user.
        """
        _owner(scope)
        async with self._uow:
            row = await self._owned(scope, trace_id, workspace_id)
            if await self._shares.find(row.workspace_id, row.trace_id, share_id) is None:
                raise TraceContentShareNotFoundError(str(share_id))
            reads = list(await self._shares.reads_page(share_id, limit=limit + 1, offset=offset))
        return tuple(_read(read) for read in reads[:limit]), len(reads) > limit

    async def end_content_share(
        self, scope: TraceScope, trace_id: str, share_id: uuid.UUID, *, workspace_id: uuid.UUID | None = None
    ) -> None:
        """Stop a share letting its audience read. Its own user only; ending one already ended does nothing.

        Raises:
            TraceNotFoundError: the scope holds no such trace.
            TraceContentShareNotFoundError: the session holds no such share.
            TraceContentShareForbiddenError: the reader is not the session's own user.
        """
        _owner(scope)
        async with self._uow:
            row = await self._owned(scope, trace_id, workspace_id)
            share = await self._shares.find(row.workspace_id, row.trace_id, share_id)
            if share is None:
                raise TraceContentShareNotFoundError(str(share_id))
            await self._shares.revoke(share, datetime.now(UTC))

    async def _owned(self, scope: TraceScope, trace_id: str, workspace_id: uuid.UUID | None) -> Trace:
        rows = await self._traces.find(_workspace_ids(scope), trace_id, workspace_id=workspace_id)
        if not rows:
            raise TraceNotFoundError(trace_id)
        if len(rows) > 1:
            raise TraceAmbiguousError(trace_id)
        if not scope.owns(rows[0].user_id):
            raise TraceContentShareForbiddenError
        return rows[0]

    async def record_definition(self, workspace_id: uuid.UUID, gates: Sequence[GateDefinition], digest: str) -> str:
        """Store this version of a workspace's agent guardrail, once, and return its fingerprint.

        The fingerprint is recomputed from the gates and must match the one sent,
        so a version is never stored under another's name. Raises
        ``GuardrailDigestMismatchError`` when it does not, and
        ``GuardrailVersionLimitError`` for a new version once the workspace keeps
        :data:`MAX_GUARDRAIL_VERSIONS_PER_WORKSPACE`.
        """
        if definition_digest(gates) != digest:
            raise GuardrailDigestMismatchError()
        async with self._uow:
            await self._definitions.lock_workspace(workspace_id)
            if (
                not await self._definitions.for_digests(workspace_id, (digest,))
                and await self._definitions.count_for_workspace(workspace_id) >= MAX_GUARDRAIL_VERSIONS_PER_WORKSPACE
            ):
                raise GuardrailVersionLimitError(MAX_GUARDRAIL_VERSIONS_PER_WORKSPACE)
            await self._definitions.insert_absent(workspace_id, digest, [dataclasses.asdict(gate) for gate in gates])
        return digest

    async def series(
        self, scope: TraceScope, filters: TraceFilter, *, bucket: TraceBucketGrain
    ) -> tuple[TraceBucket, ...]:
        """Count the scope's matching sessions active in each bucket, oldest first, populated buckets only.

        A session is active from its start to its last activity, so it counts in
        every bucket it overlaps. The window is the filters' (or the sessions'
        own, from the first start to the last activity), cut into buckets aligned
        to their size in UTC, the newest :data:`MAX_SERIES_BUCKETS` of them.
        """
        grain = TRACE_BUCKET_DURATION[bucket]
        async with self._uow:
            oldest, newest = await self._traces.activity_bounds(_workspace_ids(scope), filters)
            if oldest is None or newest is None:
                return ()
            lower = filters.start or oldest
            upper = filters.end or newest + timedelta(microseconds=1)
            starts = _bucket_starts(lower, upper, grain, MAX_SERIES_BUCKETS)
            rows = await self._traces.count_active(_workspace_ids(scope), filters, starts=starts, grain=grain)
            return tuple(
                TraceBucket(bucket=start.strftime("%Y-%m-%dT%H:%M:%SZ"), succeeded=ok, failed=bad)
                for start, ok, bad in rows
            )

    async def detail(self, scope: TraceScope, trace_id: str, *, workspace_id: uuid.UUID | None = None) -> TraceView:
        """One session with its turns.

        Raises:
            TraceNotFoundError: the scope holds no such trace, including one in a workspace outside it.
            TraceAmbiguousError: no ``workspace_id`` was given and the id is in more than one workspace.
        """
        stored = await self.get(scope, trace_id, span_limit=MAX_SPANS_PER_READ, workspace_id=workspace_id)
        if stored is None:
            raise TraceNotFoundError(trace_id)
        return trace_view(stored, now=datetime.now(UTC), context_window_of=self._context_window_of)

    async def for_request(self, scope: TraceScope, request_id: str) -> TraceSummary:
        """The session a gateway request was traced into, found by its ``Otari-Request-ID``.

        Raises:
            RequestTraceNotFoundError: the scope holds no trace of the request.
        """
        async with self._uow:
            workspace_ids = _workspace_ids(scope)
            key = await self._spans.trace_of_request(workspace_ids, request_id)
            rows = [] if key is None else await self._traces.find(workspace_ids, key[1], workspace_id=key[0])
            if not rows:
                raise RequestTraceNotFoundError(request_id)
            return _summary(rows[0])

    async def purge_user(self, user_id: str) -> int:
        """Delete every trace a user owns, in every workspace, for erasure.

        What any replica took in for them before now, and still holds, is never stored.
        """
        async with self._uow:
            await self._erasures.erase(user_id, datetime.now(UTC))
            await self._break_glass.forget_session_user(user_id)
            await self._shares.forget_session_user(user_id)
            return await self._traces.delete_for_user(user_id)

    async def expire(self, *, idle_before: datetime, started_before: datetime) -> int:
        """Delete every trace idle since ``idle_before``, and every session's trace started before ``started_before``.

        The second bound is what ends a session a client keeps alive: its id then
        starts a new trace. A request outside any session is kept for the first
        bound alone. Returns the traces deleted.
        """
        async with self._uow:
            return await self._traces.delete_expired(idle_before=idle_before, started_before=started_before)


def _bucket_starts(lower: datetime, upper: datetime, grain: timedelta, limit: int) -> list[datetime]:
    """The starts of the buckets covering ``[lower, upper)``, aligned to ``grain`` in UTC, the newest ``limit``."""
    lower, upper = _utc(lower), _utc(upper)
    if lower >= upper:
        return []
    step = grain.total_seconds()
    first = datetime.fromtimestamp((lower.timestamp() // step) * step, tz=UTC)
    total = math.ceil((upper - first) / grain)
    return [first + grain * index for index in range(max(0, total - limit), total)]


def _utc(moment: datetime) -> datetime:
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _workspace_ids(scope: TraceScope) -> frozenset[uuid.UUID] | None:
    return None if scope.deployment_wide else scope.workspace_ids


def _span_moment(span: SpanRecord, written_at: datetime) -> tuple[datetime, datetime]:
    """The earliest and latest instant a span vouches for, so a trace's window covers it."""
    earliest = span.start_time or span.end_time or written_at
    return earliest, span.end_time or earliest


def _unique_spans(spans: Sequence[SpanRecord]) -> list[SpanRecord]:
    """Drop a span repeated inside one write, so it counts once."""
    seen: dict[str, SpanRecord] = {}
    for span in spans:
        seen.setdefault(span.span_id, span)
    return list(seen.values())


def _new_trace(trace: TraceWrite, written_at: datetime) -> dict[str, Any]:
    moments = [_span_moment(span, written_at) for span in trace.spans] or [(written_at, written_at)]
    return {
        "workspace_id": trace.workspace_id,
        "trace_id": trace.trace_id,
        "user_id": trace.user_id,
        "api_key_id": trace.api_key_id,
        "session_source": trace.session_source,
        "harness": trace.harness,
        "name": trace.name,
        "started_at": min(earliest for earliest, _ in moments),
        "last_activity_at": max(latest for _, latest in moments),
    }


def _key(trace: TraceWrite) -> TraceKey:
    return (trace.workspace_id, trace.trace_id)


def _growth(key: TraceKey, spans: list[SpanRecord], written_at: datetime) -> TraceGrowth:
    moments = [_span_moment(span, written_at) for span in spans]
    llm = [span for span in spans if span.kind == "llm"]
    errors = [span for span in spans if span.outcome == "error" and not span.recovered]
    return TraceGrowth(
        workspace_id=key[0],
        trace_id=key[1],
        steps=sum(1 for span in spans if span.kind == "step"),
        # A span that only carries a request's cost is not one a person reads.
        spans=sum(1 for span in spans if span.operation != USAGE_OPERATION),
        errors=len(errors),
        # A tool's error goes back to the model, which usually carries on, so it does not fail the session.
        # An MCP tool's too, a transport failure included; a failed MCP connection is not a tool's and does.
        failing_errors=sum(1 for span in errors if span.kind != "tool"),
        input_tokens=sum(span.input_tokens or 0 for span in llm),
        output_tokens=sum(span.output_tokens or 0 for span in llm),
        cost=sum((span.cost_snapshot or Decimal(0) for span in llm), Decimal(0)),
        earliest=min(earliest for earliest, _ in moments),
        latest=max(latest for _, latest in moments),
    )


def _span_values(trace: TraceWrite, span: SpanRecord, sealed: dict[str, str]) -> dict[str, Any]:
    return {
        "workspace_id": trace.workspace_id,
        "trace_id": trace.trace_id,
        "span_id": span.span_id,
        "parent_span_id": span.parent_span_id,
        "kind": span.kind,
        "origin": span.origin,
        "name": span.name,
        "operation": span.operation,
        "start_time": span.start_time,
        "end_time": span.end_time,
        "duration_ms": span.duration_ms,
        "outcome": span.outcome,
        "recovered": span.recovered,
        "error_class": span.error_class,
        "opens_turn": span.opens_turn,
        "model": span.model,
        "provider": span.provider,
        "input_tokens": span.input_tokens,
        "output_tokens": span.output_tokens,
        "cache_read_tokens": span.cache_read_tokens,
        "cache_write_tokens": span.cache_write_tokens,
        "reasoning_tokens": span.reasoning_tokens,
        "cost_snapshot": span.cost_snapshot,
        "tool_name": span.tool_name,
        "tool_type": span.tool_type,
        "tool_call_id": span.tool_call_id,
        "request_id": span.request_id,
        "otel_trace_id": span.otel_trace_id,
        "otel_span_id": span.otel_span_id,
        "attributes": dict(span.attributes) or None,
        "content_sealed": sealed.get(span.span_id),
    }


def _stored_call_id(row: TraceSpan) -> str | None:
    """A row's tool call id, digested on read where an older version stored the id as the client sent it."""
    if row.tool_call_id is None or CALL_ID_DIGEST.fullmatch(row.tool_call_id):
        return row.tool_call_id
    return call_id_digest(row.workspace_id, row.tool_call_id)


def _span_record(row: TraceSpan) -> SpanRecord:
    """A stored span without its content, which is opened apart, only for a reader it belongs to."""
    return SpanRecord(
        span_id=row.span_id,
        kind=row.kind,
        origin=row.origin,
        name=row.name,
        outcome=row.outcome,
        parent_span_id=row.parent_span_id,
        operation=row.operation,
        start_time=row.start_time,
        end_time=row.end_time,
        duration_ms=row.duration_ms,
        recovered=row.recovered,
        error_class=row.error_class,
        opens_turn=row.opens_turn,
        model=row.model,
        provider=row.provider,
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        cache_read_tokens=row.cache_read_tokens,
        cache_write_tokens=row.cache_write_tokens,
        reasoning_tokens=row.reasoning_tokens,
        cost_snapshot=row.cost_snapshot,
        tool_name=row.tool_name,
        tool_type=row.tool_type,
        tool_call_id=_stored_call_id(row),
        request_id=row.request_id,
        otel_trace_id=row.otel_trace_id,
        otel_span_id=row.otel_span_id,
        attributes=row.attributes or {},
    )


def _refused(trace: TraceWrite, erased: dict[str, datetime | None], written_at: datetime) -> bool:
    """Whether the write is refused whole: taken in too long ago, or for a user erased since.

    The age bound is what lets an erased user's row be deleted at all: a write
    older than any row still kept would otherwise find no erasure to refuse it.
    """
    return written_at - trace.captured_at > MAX_WRITE_AGE or _erased_since(trace, erased)


def _erased_since(trace: TraceWrite, erased: dict[str, datetime | None]) -> bool:
    """Whether the write's user was erased after, or too near, the moment it was taken in.

    ``ERASURE_MARGIN`` absorbs the difference between two replicas' clocks, at the
    cost of refusing a recreated user's first spans for that long.
    """
    at = erased.get(trace.user_id) if trace.user_id is not None else None
    return at is not None and trace.captured_at < at + ERASURE_MARGIN


def _current(trace: TraceWrite, generations: dict[uuid.UUID, int]) -> bool:
    """Whether the write's content was captured on its workspace's current generation."""
    return trace.content_generation is not None and generations.get(trace.workspace_id) == trace.content_generation


def _owner(scope: TraceScope) -> ContentReader:
    if scope.reader is None:
        raise TraceContentShareForbiddenError
    return scope.reader


def _break_glass_read(row: TraceBreakGlassRead) -> BreakGlassRead:
    return BreakGlassRead(
        id=row.id,
        kind=row.kind,  # type: ignore[arg-type]
        organization_id=row.organization_id,
        workspace_id=row.workspace_id,
        trace_id=row.trace_id,
        session_user_id=row.session_user_id,
        reader_user_id=row.reader_user_id,
        reader_email=row.reader_email,
        reason=row.reason,
        reference=row.reference,
        read_at=row.read_at,
    )


def _read(row: TraceContentRead) -> ContentRead:
    return ContentRead(reader_user_id=row.reader_user_id, reader_email=row.reader_email, read_at=row.read_at)


def _share(row: TraceContentShare, reads: tuple[ContentRead, ...] = (), read_count: int = 0) -> ContentShare:
    return ContentShare(
        id=row.id,
        audience=row.audience,  # type: ignore[arg-type]
        note=row.note,
        created_at=row.created_at,
        revoked_at=row.revoked_at,
        reads=reads,
        read_count=read_count,
    )


def _definition(row: TraceGuardrailDefinition) -> GuardrailDefinition:
    return GuardrailDefinition(digest=row.digest, gates=gates_from_body(row.gates))


def _summary(row: Trace) -> TraceSummary:
    return TraceSummary(
        workspace_id=row.workspace_id,
        trace_id=row.trace_id,
        user_id=row.user_id,
        api_key_id=row.api_key_id,
        session_source=row.session_source,
        harness=row.harness,
        name=row.name,
        started_at=row.started_at,
        last_activity_at=row.last_activity_at,
        step_count=row.step_count,
        span_count=row.span_count,
        error_count=row.error_count,
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        cost_snapshot=row.cost_snapshot,
    )
