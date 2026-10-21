"""The subagent recommenders an operator adds, changes and removes.

Each write commits, then hands the committed rows to the snapshot, so this
process answers with them at once; other replicas pick them up on their next
reload.
"""

import uuid
from collections.abc import Awaitable, Sequence
from typing import Any

from gateway.core.config import DECISION_PROVIDERS, DECISION_PROVIDERS_SELF_HOSTED
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.routing_exceptions import (
    AgentRecommenderNotFoundError,
    AgentRecommenderSwitchConflictError,
    InvalidAgentRecommenderError,
)
from gateway.exceptions.shared_exceptions import SecretBoxUnavailableTenancyError
from gateway.models.money import to_usd
from gateway.models.routing import AgentRecommender
from gateway.repositories.routing import AgentRecommenderRepository, AgentRecommenderSwitchConflict
from gateway.schemas.routing import (
    AgentRecommenderCreate,
    AgentRecommenderPublic,
    AgentRecommendersPublic,
    AgentRecommenderUpdate,
    DecisionProviderOption,
)
from gateway.services.routing._recommender_pricing import RecommenderPrice
from gateway.services.routing._recommender_rules import check_recommender
from gateway.services.routing._recommender_snapshot import RecommenderSnapshot
from gateway.services.secret_box import SecretBoxUnavailableError, encrypt_secret
from gateway.services.url_safety import UnsafeURLError, validate_provider_api_base

_DECISION_PROVIDER_OPTIONS = [
    DecisionProviderOption(provider=provider, self_hosted=provider in DECISION_PROVIDERS_SELF_HOSTED)
    for provider in DECISION_PROVIDERS
]


def _api_base(value: str | None) -> str | None:
    """Fold a blank address to None, which means the provider's own."""
    return (value or "").strip() or None


def _key_fields(api_key: str | None) -> dict[str, Any]:
    try:
        encrypted = encrypt_secret(api_key) if api_key is not None else None
    except SecretBoxUnavailableError:
        raise SecretBoxUnavailableTenancyError("subagent recommender keys") from None
    # A key of four characters or fewer would be shown whole.
    last4 = api_key[-4:] if api_key is not None and len(api_key) > 4 else None
    return {"encrypted_api_key": encrypted, "api_key_last4": last4}


async def _check(*, provider: str, has_api_key: bool, api_base: str | None, price: RecommenderPrice) -> None:
    try:
        check_recommender(provider=provider, has_api_key=has_api_key, api_base=api_base, price=price)
    except ValueError as exc:
        raise InvalidAgentRecommenderError(str(exc)) from None
    if api_base is not None:
        try:
            await validate_provider_api_base(api_base)
        except UnsafeURLError as exc:
            raise InvalidAgentRecommenderError(str(exc)) from None


class AgentRecommenders:
    """The operator's use cases over the recommender rows, for ``RoutingService`` to offer."""

    def __init__(self, uow: UnitOfWork, rows: AgentRecommenderRepository, snapshot: RecommenderSnapshot) -> None:
        self._uow = uow
        self._rows = rows
        self._snapshot = snapshot

    async def list(self) -> AgentRecommendersPublic:
        async with self._uow:
            rows = await self._rows.list_all()
            data = [AgentRecommenderPublic.from_row(row) for row in rows]
        return AgentRecommendersPublic(data=data, decision_providers=_DECISION_PROVIDER_OPTIONS)

    async def add(self, body: AgentRecommenderCreate) -> AgentRecommenderPublic:
        api_base = _api_base(body.api_base)
        price = RecommenderPrice(to_usd(body.input_price_per_million), to_usd(body.output_price_per_million))
        await _check(provider=body.provider, has_api_key=body.api_key is not None, api_base=api_base, price=price)
        fields: dict[str, Any] = {
            "id": uuid.uuid4(),
            "provider": body.provider,
            "model": body.model,
            "api_base": api_base,
            "input_price_per_million": price.input_price_per_million,
            "output_price_per_million": price.output_price_per_million,
            "enabled": body.enabled,
            **_key_fields(body.api_key),
        }
        async with self._uow:
            if body.enabled:
                await self._rows.switch_off_all_but(fields["id"])
            row = await self._save(self._rows.add(fields))
            public, rows = await self._written(row)
        self._snapshot.replace(rows)
        return public

    async def update(self, recommender_id: uuid.UUID, patch: AgentRecommenderUpdate) -> AgentRecommenderPublic:
        async with self._uow:
            row = await self._get(recommender_id)
            provider = patch.provider or row.provider
            api_base = _api_base(patch.api_base) if "api_base" in patch.model_fields_set else row.api_base
            # The key goes to the API base, so a new address or provider needs it typed again.
            keeps_key = patch.api_key is None and provider == row.provider and api_base == row.api_base
            price = RecommenderPrice(
                row.input_price_per_million
                if patch.input_price_per_million is None
                else to_usd(patch.input_price_per_million),
                row.output_price_per_million
                if patch.output_price_per_million is None
                else to_usd(patch.output_price_per_million),
            )
            has_api_key = patch.api_key is not None or (keeps_key and row.encrypted_api_key is not None)
            await _check(provider=provider, has_api_key=has_api_key, api_base=api_base, price=price)
            fields: dict[str, Any] = {
                "provider": provider,
                "model": patch.model or row.model,
                "api_base": api_base,
                "input_price_per_million": price.input_price_per_million,
                "output_price_per_million": price.output_price_per_million,
            }
            if not keeps_key:
                fields.update(_key_fields(patch.api_key))
            if patch.enabled is not None:
                fields["enabled"] = patch.enabled
            if patch.enabled:
                await self._rows.switch_off_all_but(row.id)
            row = await self._save(self._rows.change(row, fields))
            public, rows = await self._written(row)
        self._snapshot.replace(rows)
        return public

    async def remove(self, recommender_id: uuid.UUID) -> None:
        async with self._uow:
            await self._rows.delete(await self._get(recommender_id))
            rows = await self._rows.list_all()
        self._snapshot.replace(rows)

    async def _get(self, recommender_id: uuid.UUID) -> AgentRecommender:
        row = await self._rows.get(recommender_id)
        if row is None:
            raise AgentRecommenderNotFoundError
        return row

    @staticmethod
    async def _save(write: Awaitable[AgentRecommender]) -> AgentRecommender:
        try:
            return await write
        except AgentRecommenderSwitchConflict:
            raise AgentRecommenderSwitchConflictError from None

    async def _written(self, row: AgentRecommender) -> tuple[AgentRecommenderPublic, Sequence[AgentRecommender]]:
        """The row as the API shows it, and every row, for the snapshot once the block commits."""
        return AgentRecommenderPublic.from_row(row), await self._rows.list_all()
