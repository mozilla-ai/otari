from dataclasses import dataclass
from typing import Self

from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.traces.barrier_repository import ContentBarrierRepository, UserBarrierRepository
from gateway.repositories.traces.break_glass_repository import BreakGlassRepository
from gateway.repositories.traces.definition_repository import GuardrailDefinitionRepository
from gateway.repositories.traces.share_repository import ContentShareRepository
from gateway.repositories.traces.span_repository import SpanRepository
from gateway.repositories.traces.trace_repository import TraceRepository


@dataclass(frozen=True)
class TracesRepositories:
    """The traces domain's repositories, all on one Unit of Work."""

    traces: TraceRepository
    spans: SpanRepository
    definitions: GuardrailDefinitionRepository
    shares: ContentShareRepository
    barriers: ContentBarrierRepository
    erasures: UserBarrierRepository
    break_glass: BreakGlassRepository

    @classmethod
    def on(cls, uow: UnitOfWork) -> Self:
        """Build every repository on this Unit of Work."""
        return cls(
            traces=TraceRepository(uow),
            spans=SpanRepository(uow),
            definitions=GuardrailDefinitionRepository(uow),
            shares=ContentShareRepository(uow),
            barriers=ContentBarrierRepository(uow),
            erasures=UserBarrierRepository(uow),
            break_glass=BreakGlassRepository(uow),
        )
