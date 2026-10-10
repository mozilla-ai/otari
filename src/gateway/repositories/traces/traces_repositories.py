from dataclasses import dataclass
from typing import Self

from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.traces.span_repository import SpanRepository
from gateway.repositories.traces.trace_repository import TraceRepository


@dataclass(frozen=True)
class TracesRepositories:
    """The traces domain's repositories, all on one Unit of Work."""

    traces: TraceRepository
    spans: SpanRepository

    @classmethod
    def on(cls, uow: UnitOfWork) -> Self:
        """Build every repository on this Unit of Work."""
        return cls(traces=TraceRepository(uow), spans=SpanRepository(uow))
