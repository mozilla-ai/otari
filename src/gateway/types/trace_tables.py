"""What the database trace store needs from the traces repositories.

A leaf type, so the store in ``adapters/`` can be typed against the repositories
without importing them: a domain's repositories are imported by its own packages
and the builders in ``api/deps.py`` alone. The traces repositories satisfy these
protocols structurally, and ``api/deps.py`` hands the store a builder for them
(``adapters.trace_storage_adapter.TraceTablesBuilder``).
"""

import uuid
from collections.abc import Collection, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, Protocol

from gateway.models.traces import Trace, TraceSpan
from gateway.ports.trace_storage_port import TraceFilter


class TraceRows(Protocol):
    """The trace queries the local adapter runs."""

    async def owners(self, workspace_id: uuid.UUID, trace_ids: Collection[str]) -> dict[str, str | None]: ...

    async def create_if_absent(self, values: dict[str, Any]) -> None: ...

    async def add_totals(
        self,
        workspace_id: uuid.UUID,
        trace_id: str,
        *,
        steps: int,
        spans: int,
        errors: int,
        input_tokens: int,
        output_tokens: int,
        cost: Decimal,
        earliest: datetime,
        latest: datetime,
    ) -> None: ...

    async def page(
        self, workspace_ids: Collection[uuid.UUID] | None, query: TraceFilter, *, limit: int, offset: int
    ) -> Sequence[Trace]: ...

    async def count_matching(self, workspace_ids: Collection[uuid.UUID] | None, query: TraceFilter) -> int: ...

    async def bucket_counts(
        self, workspace_ids: Collection[uuid.UUID] | None, query: TraceFilter, *, bucket: Literal["hour", "day"]
    ) -> list[tuple[str, int, int]]: ...

    async def find(self, workspace_ids: Collection[uuid.UUID] | None, trace_id: str) -> Trace | None: ...

    async def delete_matching(self, workspace_ids: Collection[uuid.UUID] | None, query: TraceFilter) -> int: ...

    async def delete_for_user(self, user_id: str) -> int: ...

    async def delete_inactive_before(self, before: datetime) -> int: ...

    async def inactive_before(self, before: datetime, *, limit: int) -> list[tuple[uuid.UUID, str]]: ...

    async def matching_sessions(
        self, workspace_ids: Collection[uuid.UUID] | None, query: TraceFilter, *, limit: int
    ) -> list[tuple[uuid.UUID, str]]: ...

    async def sessions_for_user(self, user_id: str, *, limit: int) -> list[tuple[uuid.UUID, str]]: ...

    async def delete_sessions(self, keys: Collection[tuple[uuid.UUID, str]]) -> int: ...


class SpanRows(Protocol):
    """The span queries the local adapter runs."""

    async def insert_new(self, rows: Sequence[dict[str, Any]]) -> set[str]: ...

    async def for_trace(self, workspace_id: uuid.UUID, trace_id: str, *, limit: int) -> Sequence[TraceSpan]: ...


class TraceTables(Protocol):
    """The traces repositories, built on one Unit of Work."""

    @property
    def traces(self) -> TraceRows: ...

    @property
    def spans(self) -> SpanRows: ...
