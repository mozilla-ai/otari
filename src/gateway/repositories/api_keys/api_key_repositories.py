from dataclasses import dataclass
from typing import Self

from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.api_keys.api_key_repository import ApiKeyRepository
from gateway.repositories.api_keys.federation_rule_repository import FederationRuleRepository


@dataclass(frozen=True)
class ApiKeyRepositories:
    """The api-keys domain's repositories, all on one Unit of Work."""

    keys: ApiKeyRepository
    federation_rules: FederationRuleRepository

    @classmethod
    def on(cls, uow: UnitOfWork) -> Self:
        """Build every repository on this Unit of Work."""
        return cls(keys=ApiKeyRepository(uow), federation_rules=FederationRuleRepository(uow))
