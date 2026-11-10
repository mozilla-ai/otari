from dataclasses import dataclass
from typing import Self

from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.memory.entry_repository import EntryRepository
from gateway.repositories.memory.key_repository import KeyRepository
from gateway.repositories.memory.organization_settings_repository import OrganizationSettingsRepository
from gateway.repositories.memory.proposal_repository import ProposalRepository
from gateway.repositories.memory.read_repository import ReadRepository
from gateway.repositories.memory.settings_repository import SettingsRepository
from gateway.repositories.memory.signal_repository import SignalRepository


@dataclass(frozen=True)
class MemoryRepositories:
    """The memory domain's repositories, all on one Unit of Work."""

    keys: KeyRepository
    entries: EntryRepository
    proposals: ProposalRepository
    signals: SignalRepository
    reads: ReadRepository
    settings: SettingsRepository
    organizations: OrganizationSettingsRepository

    @classmethod
    def on(cls, uow: UnitOfWork) -> Self:
        """Build every repository on this Unit of Work."""
        return cls(
            keys=KeyRepository(uow),
            entries=EntryRepository(uow),
            proposals=ProposalRepository(uow),
            signals=SignalRepository(uow),
            reads=ReadRepository(uow),
            settings=SettingsRepository(uow),
            organizations=OrganizationSettingsRepository(uow),
        )
