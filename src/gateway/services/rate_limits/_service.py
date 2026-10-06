"""The rate-limit rules an operator manages from the dashboard."""

from pydantic import ValidationError

from gateway.core.config import GatewayConfig, RateLimitRule
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.rate_limits_exceptions import (
    RateLimitRuleAlreadyExistsError,
    RateLimitRuleInConfigError,
    RateLimitRuleInvalidError,
    RateLimitRuleNotFoundError,
)
from gateway.models.rate_limits import StoredRateLimitRule
from gateway.repositories.rate_limits import RateLimitRuleNameTaken, RateLimitRuleRepository
from gateway.schemas.rate_limits import (
    RateLimitRuleCreate,
    RateLimitRulePublic,
    RateLimitRulesPublic,
    RateLimitRuleUpdate,
)
from gateway.services.rate_limits._overlay import config_file_rules, refresh_rate_limit_rules, rule_of


def _public(row: StoredRateLimitRule) -> RateLimitRulePublic:
    return RateLimitRulePublic(**rule_of(row).model_dump(), source="dashboard", updated_at=row.updated_at)


class RateLimitService:
    """Lists every rule in effect, and adds, changes and removes the stored ones.

    A write is applied to this worker's rule set as soon as it commits; the
    refresher brings the other workers and replicas level.
    """

    def __init__(self, uow: UnitOfWork, rules: RateLimitRuleRepository, config: GatewayConfig) -> None:
        self._uow = uow
        self._rules = rules
        self._config = config

    async def list_rules(self) -> RateLimitRulesPublic:
        """Every rule in effect: config-file rules first, then stored ones."""
        from_config = config_file_rules(self._config)
        taken = {rule.name for rule in from_config}
        async with self._uow:
            rows = await self._rules.list_all()
            stored = [_public(row) for row in rows if row.name not in taken]
        return RateLimitRulesPublic(
            rules=[
                *(
                    RateLimitRulePublic(**rule.model_dump(), source="config")
                    for rule in sorted(from_config, key=lambda rule: rule.name)
                ),
                *stored,
            ]
        )

    async def create_rule(self, request: RateLimitRuleCreate) -> RateLimitRulePublic:
        """Store a new rule and start enforcing it."""
        self._refuse_config_rule(request.name)
        async with self._uow:
            try:
                row = await self._rules.add(StoredRateLimitRule(**request.model_dump()))
            except RateLimitRuleNameTaken as exc:
                raise RateLimitRuleAlreadyExistsError(exc.name) from exc
            public = _public(row)
        await self._apply()
        return public

    async def update_rule(self, name: str, request: RateLimitRuleUpdate) -> RateLimitRulePublic:
        """Change a stored rule. Its counts carry over, because they are kept under its name."""
        self._refuse_config_rule(name)
        async with self._uow:
            row = await self._rules.get(name)
            if row is None:
                raise RateLimitRuleNotFoundError(name)
            changes = request.model_dump(exclude_unset=True)
            if changes.get("per") not in (None, "model") and "models" not in changes:
                # Only a per-model rule names models, so a rule moved off per: model drops them.
                changes["models"] = None
            try:
                merged = RateLimitRule(**{**rule_of(row).model_dump(), **changes})
            except ValidationError as exc:
                raise RateLimitRuleInvalidError(exc.errors()[0]["msg"]) from exc
            row = await self._rules.update(row, merged.model_dump(exclude={"name"}))
            public = _public(row)
        await self._apply()
        return public

    async def delete_rule(self, name: str) -> None:
        """Remove a stored rule and stop enforcing it."""
        async with self._uow:
            row = await self._rules.get(name)
            if row is None:
                self._refuse_config_rule(name)
                raise RateLimitRuleNotFoundError(name)
            await self._rules.delete(row)
        await self._apply()

    def _refuse_config_rule(self, name: str) -> None:
        if any(rule.name == name for rule in config_file_rules(self._config)):
            raise RateLimitRuleInConfigError(name)

    async def _apply(self) -> None:
        await refresh_rate_limit_rules(self._uow, self._config)
