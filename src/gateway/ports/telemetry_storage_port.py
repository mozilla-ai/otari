"""Where captured coding-agent telemetry is stored and read back.

The seam between the OTLP receiver and whichever build keeps what it captures.
Otari's own storage is the ``agent_telemetry`` table in its database, which is
enough for a single deployment reading its own window; a deployment collecting
telemetry across many tenants wants a scale-out store instead, and that is the
second implementation this port exists for (``ARCHITECTURE.md``, rule 7).

What travels through here is deliberately narrow. The receiver captures two
kinds of content-free row, behavioral events and outcome-metric points, and
this port owns both. It does not own ``usage_logs``: a span that carries GenAI
usage is projected into a usage row before it reaches any of this, and those
rows are the money path (cost, budget attribution, billing meters), so they
stay in the database in every build rather than behind a swappable backend.

Reads are aggregates, not row dumps. Every question the read endpoints ask is
"how much happened, grouped how", so the port answers in those terms and each
adapter is free to compute them where they are cheapest: a SQL ``GROUP BY``, a
search engine's date histogram. The one exception is
:meth:`TelemetryStoragePort.metric_points`, which hands back raw readings,
because turning a cumulative counter into an increment is subtle domain logic
(generations split at a counter reset, a baseline for a generation that starts
in the window) that must not be reimplemented once per adapter. Time buckets
cross the seam as canonical UTC strings so two adapters cannot disagree about
which bucket a reading falls in.

Durability is the adapter's own. :meth:`TelemetryStoragePort.record` and the
purge methods settle their work before returning, so a caller cannot roll one
back and must not treat one as part of its own unit of work. That is what
today's storage already does (it commits per batch), and the only contract an
out-of-process store could honor. The corollary for a caller: do not hold
uncommitted changes across one of these calls, because an adapter sharing your
session will settle those too.

Stability: this interface is not frozen while Otari is pre-1.0. Overlay authors
should pin a released tag and expect the shape to move.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

# Repeated rather than imported from the read endpoints: a port may not depend
# on the API layer, and the two are the same type to the type checker, so a
# bucket name crosses the seam without either side naming the other.
TelemetryBucket = Literal["hour", "day"]

# The dimensions a grouped read may split on. Both are stored on every row, so
# an adapter groups without a join.
TelemetryGroupBy = Literal["user_id", "api_key_id"]


class TelemetryScanTooLargeError(Exception):
    """Raised when a read would have to materialize more points than it may.

    The port owns this error so a caller can refuse a read without naming the
    store that refused it. Carries the ceiling that was exceeded, so the caller
    can say what to narrow. Only :meth:`TelemetryStoragePort.metric_points`
    raises it: the aggregates beside it collapse in the store and are bounded
    by the window alone.
    """

    def __init__(self, limit: int, message: str | None = None) -> None:
        super().__init__(message or f"read would return more than {limit} metric data points")
        self.limit = limit


@dataclass(frozen=True)
class TelemetryRecord:
    """One captured row, before it is stored.

    Either a behavioral event (``kind`` is ``None``) or an outcome-metric point
    (``kind`` is ``"metric"``), never both: each shape leaves the other's
    fields unset. ``dedup_key`` is the natural idempotency key its mapper
    derived, and ``(source, dedup_key)`` is what makes a re-exported row a
    duplicate rather than a second row.

    ``tool_use_id`` and ``event_sequence`` feed the dedup key and are not
    stored in their own right, so an adapter persists everything else.
    """

    name: str
    timestamp: datetime
    source: str
    dedup_key: str
    tool_name: str | None = None
    decision: str | None = None
    success: bool | None = None
    duration_ms: int | None = None
    status_code: int | None = None
    prompt_length: int | None = None
    session_label: str | None = None
    tool_use_id: str | None = None
    event_sequence: int | None = None
    # Metric-point fields; all None on a behavioral event.
    kind: str | None = None
    value: float | None = None
    temporality: str | None = None
    series_start: datetime | None = None
    series_key: str | None = None


@dataclass(frozen=True)
class IngestResult:
    """How one export's records were accounted for.

    ``duplicate`` is a success, not a failure: an exporter that resends a batch
    must not be told anything went wrong. ``rejected`` is the caller's own
    refusal (an export attributed to a user who is no longer active), so an
    adapter never sets it.
    """

    accepted: int = 0
    duplicate: int = 0
    rejected: int = 0


@dataclass(frozen=True)
class TelemetryFilter:
    """The scope a read or a purge applies to.

    Every field narrows; an unset field does not filter. ``start`` is
    inclusive and ``end`` exclusive, matching the half-open windows the usage
    endpoints already report. Several values in ``user_ids`` or ``api_key_ids``
    match any of them.

    A naive bound means UTC. The date query parameters advertise ISO 8601 and
    parse to a naive value when the caller omits the offset, so an adapter that
    resolved one against its own local time would select a different set of
    rows per deployment, and two adapters would disagree about the same filter.
    """

    start: datetime | None = None
    end: datetime | None = None
    user_ids: tuple[str, ...] = ()
    api_key_ids: tuple[str, ...] = ()
    name: str | None = None
    session_label: str | None = None


@dataclass(frozen=True)
class MetricPoint:
    """One stored metric reading, exactly as the agent reported it.

    ``series_key`` and ``series_start`` identify the generation the reading
    belongs to: a counter reset arrives as a new ``series_start``, and diffing
    across one would subtract the pre-reset total. ``temporality`` is
    ``"cumulative"`` or ``"delta"``.
    """

    name: str
    series_key: str | None
    series_start: datetime | None
    temporality: str | None
    timestamp: datetime
    value: float


@dataclass(frozen=True)
class BehaviorGroup:
    """How many behavioral events share one ``(name, tool_name, decision)``."""

    name: str
    tool_name: str | None
    decision: str | None
    count: int


@dataclass(frozen=True)
class BehaviorCounts:
    """Behavioral-event volume for a scope, plus how many sessions produced it.

    ``sessions`` counts distinct non-null session labels, which is a separate
    question from the groups and cannot be derived by summing them.
    """

    groups: tuple[BehaviorGroup, ...] = ()
    sessions: int = 0


@dataclass(frozen=True)
class BucketedEventCount:
    """How many behavioral events of one name landed in one time bucket.

    ``bucket_start`` is the canonical UTC bucket key
    (``2026-08-25T00:00:00Z`` for a day, ``...T13:00:00Z`` for an hour), so
    the caller can line these up against the spend it buckets on the same grid.
    """

    bucket_start: str
    name: str
    count: int


@dataclass(frozen=True)
class GroupTotal:
    """One group's total row count. ``key`` is ``None`` for a group whose
    column is NULL, for instance a since-deleted user."""

    key: str | None
    rows: int


@dataclass(frozen=True)
class GroupedBucketCount:
    """One (bucket, group) cell of a grouped series.

    ``is_other`` marks the fold holding every group past the top N, which is
    why ``key`` being ``None`` does not identify it: a real group can have a
    NULL key too.
    """

    bucket_start: str
    key: str | None
    is_other: bool
    rows: int


@dataclass(frozen=True)
class GroupedRowCounts:
    """Row volume over time, split by one dimension.

    ``groups`` holds the top N by volume, ``total`` is every matching row
    including the groups that did not rank, and ``points`` is the grid. The
    caller reconciles the fold from ``total`` minus the ranked groups, so an
    adapter reports the two consistently or the chart will not add up.
    """

    groups: tuple[GroupTotal, ...] = ()
    total: int = 0
    points: tuple[GroupedBucketCount, ...] = ()


class TelemetryStoragePort(Protocol):
    """What a build must answer to keep the receiver's captured telemetry."""

    async def record(
        self,
        *,
        api_key_id: str,
        user_id: str,
        records: tuple[TelemetryRecord, ...],
    ) -> IngestResult:
        """Store ``records``, attributed to the key and user that exported them.

        Idempotent on ``(source, dedup_key)``: a record already stored counts
        as ``duplicate`` and is not stored twice, so an exporter that resends a
        batch after a timeout does not double count. Repeats *within* this
        batch are the store's to collapse as well, and count the same way: the
        stored projection is lossy by design, so one export can carry two
        records that land on one key.

        Settles before returning: a caller cannot roll this back, and a
        successful return means the accepted records are durable. It is not
        all-or-nothing. An adapter may settle in pieces, so a call that raises
        can leave some records stored, and this port asks for idempotency
        rather than atomicity precisely so that resolves itself: the honest
        recovery is to send the batch again, and whatever landed comes back as
        ``duplicate``.
        """
        ...

    async def metric_points(
        self,
        *,
        filters: TelemetryFilter,
        limit: int,
    ) -> tuple[MetricPoint, ...]:
        """Return the metric readings in scope, for the caller to diff.

        Readings are returned as stored rather than aggregated, because the
        cumulative-to-increment arithmetic is shared domain logic and must have
        one implementation across adapters.

        Raises:
            TelemetryScanTooLargeError: If more than ``limit`` readings match.
                Fails closed rather than truncating, since a silently short
                answer would read as less work having happened.

        """
        ...

    async def behavior_counts(self, *, filters: TelemetryFilter) -> BehaviorCounts:
        """Count the behavioral events in scope, grouped, with distinct sessions.

        Metric points are excluded: they are counted by their own readings, and
        folding them in here would inflate every behavioral total.
        """
        ...

    async def behavior_counts_by_bucket(
        self,
        *,
        filters: TelemetryFilter,
        bucket: TelemetryBucket,
    ) -> tuple[BucketedEventCount, ...]:
        """The same counts as :meth:`behavior_counts`, split by time bucket.

        Only populated buckets are returned; the caller densifies the grid,
        so an empty bucket costs nothing to report.
        """
        ...

    async def count(self, *, filters: TelemetryFilter) -> int:
        """Count every row in scope, behavioral and metric alike.

        Sizes exactly what :meth:`purge` with the same filters would remove,
        which is what lets a caller show "delete all N matching" honestly.
        """
        ...

    async def grouped_row_counts(
        self,
        *,
        filters: TelemetryFilter,
        group_by: TelemetryGroupBy,
        bucket: TelemetryBucket,
        top_n: int,
    ) -> GroupedRowCounts:
        """Row volume over time, split by ``group_by``, top ``top_n`` groups.

        Groups past ``top_n`` collapse into the ``is_other`` fold inside the
        store, so the grid stays bounded by buckets times ``top_n`` however
        high the dimension's cardinality is.
        """
        ...

    async def purge(self, *, ids: tuple[str, ...], filters: TelemetryFilter) -> int:
        """Remove rows by explicit id or by filter, and report how many went.

        Exactly one mode: a non-empty ``ids`` targets those rows and ignores
        ``filters``; an empty ``ids`` targets everything ``filters`` matches.
        Matching nothing removes nothing and is not an error.

        Settles before returning, like :meth:`record`.
        """
        ...

    async def purge_user(self, *, user_id: str) -> int:
        """Remove every row attributed to one user, and report how many went.

        Separate from :meth:`purge` because it answers an erasure request
        rather than an operator's selection: it takes no window and no other
        filter, so a caller cannot narrow it by accident and leave data behind.
        """
        ...
