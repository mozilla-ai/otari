from dataclasses import dataclass
from typing import Self

from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.traces.content_access_repository import ContentAccessRepository
from gateway.repositories.traces.content_key_repository import ContentKeyRepository
from gateway.repositories.traces.content_repository import ContentRepository
from gateway.repositories.traces.span_repository import SpanRepository
from gateway.repositories.traces.trace_repository import TraceRepository
from gateway.repositories.traces.trace_settings_repository import TraceSettingsRepository


@dataclass(frozen=True)
class TracesRepositories:
    """The traces domain's repositories, all on one Unit of Work."""

    traces: TraceRepository
    spans: SpanRepository
    content: ContentRepository
    keys: ContentKeyRepository
    access: ContentAccessRepository
    settings: TraceSettingsRepository

    @classmethod
    def on(cls, uow: UnitOfWork) -> Self:
        """Build every repository on this Unit of Work."""
        return cls(
            traces=TraceRepository(uow),
            spans=SpanRepository(uow),
            content=ContentRepository(uow),
            keys=ContentKeyRepository(uow),
            access=ContentAccessRepository(uow),
            settings=TraceSettingsRepository(uow),
        )
