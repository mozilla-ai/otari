"""The Subagent router's admin surface: list, add, change and remove the stored recommenders.

Each write is one Unit of Work block, and the rows it committed are handed to
this worker's in-memory copy at once, so the request after it is answered with
them. Other workers pick them up on their next refresh.
"""

import uuid
from decimal import Decimal

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.routing_exceptions import (
    AgentRecommenderNotFoundError,
    AgentRecommenderUnsafeApiBaseError,
    InvalidAgentRecommenderError,
)
from gateway.exceptions.shared_exceptions import SecretBoxUnavailableTenancyError
from gateway.models.money import to_usd
from gateway.models.routing import AgentRecommender
from gateway.repositories.routing import AgentRecommenderRepository
from gateway.schemas.routing import (
    AgentRecommenderCreate,
    AgentRecommenderPublic,
    AgentRecommendersPublic,
    AgentRecommenderUpdate,
)
from gateway.services.providers import last4, normalize_api_base, public_api_base
from gateway.services.routing._agent_recommender_price import RecommenderPrice
from gateway.services.routing._agent_recommender_rules import check_recommender
from gateway.services.routing._agent_recommenders import apply_agent_recommenders
from gateway.services.secret_box import SecretBoxUnavailableError, encrypt_secret
from gateway.services.url_safety import UnsafeURLError, validate_provider_api_base

_STORED = "the subagent router's key"


def _public(row: AgentRecommender) -> AgentRecommenderPublic:
    return AgentRecommenderPublic(
        id=row.id,
        provider=row.provider,
        model=row.model,
        api_base=public_api_base(row.api_base),
        api_key_last4=row.api_key_last4,
        has_api_key=row.encrypted_api_key is not None,
        input_price_per_million=float(row.input_price_per_million),
        output_price_per_million=float(row.output_price_per_million),
        enabled=row.enabled,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class AgentRecommenderService:
    """Configure the deployment's subagent routers."""

    def __init__(self, uow: UnitOfWork, recommenders: AgentRecommenderRepository) -> None:
        self._uow = uow
        self._rows = recommenders

    async def list(self) -> AgentRecommendersPublic:
        """Every stored recommender, oldest first."""
        async with self._uow:
            rows = await self._rows.list_all()
        return AgentRecommendersPublic(data=[_public(row) for row in rows])

    async def create(self, body: AgentRecommenderCreate) -> AgentRecommenderPublic:
        """Add a recommender, switched on when it is the first stored.

        Raises:
            InvalidAgentRecommenderError: the configuration could not work.
            AgentRecommenderUnsafeApiBaseError: the SSRF gate refuses the API base.
            SecretBoxUnavailableTenancyError: no usable ``OTARI_SECRET_KEY``.
            AgentRecommenderSwitchConflictError: another one was switched on at the same moment.

        """
        api_base = normalize_api_base(body.api_base)
        price = RecommenderPrice(to_usd(body.input_price_per_million), to_usd(body.output_price_per_million))
        await self._check(provider=body.provider, has_api_key=body.api_key is not None, api_base=api_base, price=price)
        encrypted = self._encrypt(body.api_key)
        async with self._uow:
            rows = await self._rows.list_all()
            row = await self._rows.insert(
                provider=body.provider,
                model=body.model,
                api_base=api_base,
                encrypted_api_key=encrypted,
                api_key_last4=last4(body.api_key) if body.api_key is not None else None,
                input_price_per_million=price.input_price_per_million,
                output_price_per_million=price.output_price_per_million,
                enabled=not rows,
            )
            rows = await self._rows.list_all()
        apply_agent_recommenders(rows)
        return _public(row)

    async def update(self, recommender_id: uuid.UUID, patch: AgentRecommenderUpdate) -> AgentRecommenderPublic:
        """Change a recommender; switching it on switches every other one off.

        The stored key is kept when none is sent and neither the provider nor
        the API base changed: the key goes to that address.

        Raises:
            AgentRecommenderNotFoundError: no recommender has this id.
            InvalidAgentRecommenderError: the configuration could not work.
            AgentRecommenderUnsafeApiBaseError: the SSRF gate refuses the API base.
            SecretBoxUnavailableTenancyError: no usable ``OTARI_SECRET_KEY``.
            AgentRecommenderSwitchConflictError: another one was switched on at the same moment.

        """
        async with self._uow:
            row = await self._get(recommender_id)
        # Readable after the block: the session does not expire on commit.
        api_base = normalize_api_base(patch.api_base) if "api_base" in patch.model_fields_set else row.api_base
        provider = patch.provider or row.provider
        keeps_key = patch.api_key is None and provider == row.provider and api_base == row.api_base
        has_key = patch.api_key is not None or (keeps_key and row.encrypted_api_key is not None)
        price = RecommenderPrice(
            _or(patch.input_price_per_million, row.input_price_per_million),
            _or(patch.output_price_per_million, row.output_price_per_million),
        )
        await self._check(provider=provider, has_api_key=has_key, api_base=api_base, price=price)
        encrypted = row.encrypted_api_key if keeps_key else self._encrypt(patch.api_key)
        async with self._uow:
            row.provider = provider
            row.model = patch.model or row.model
            row.api_base = api_base
            if not keeps_key:
                row.encrypted_api_key = encrypted
                row.api_key_last4 = last4(patch.api_key) if patch.api_key is not None else None
            row.input_price_per_million = price.input_price_per_million
            row.output_price_per_million = price.output_price_per_million
            if patch.enabled:
                await self._rows.switch_off_all_but(row.id)
            if patch.enabled is not None:
                row.enabled = patch.enabled
            row = await self._rows.save(row)
            rows = await self._rows.list_all()
        apply_agent_recommenders(rows)
        return _public(row)

    async def delete(self, recommender_id: uuid.UUID) -> None:
        """Remove a recommender and its key. With none left, nothing answers.

        Raises:
            AgentRecommenderNotFoundError: no recommender has this id.

        """
        async with self._uow:
            await self._rows.delete_row(await self._get(recommender_id))
            rows = await self._rows.list_all()
        apply_agent_recommenders(rows)

    async def _get(self, recommender_id: uuid.UUID) -> AgentRecommender:
        row = await self._rows.get(recommender_id)
        if row is None:
            raise AgentRecommenderNotFoundError
        return row

    async def _check(self, *, provider: str, has_api_key: bool, api_base: str | None, price: RecommenderPrice) -> None:
        """The pure rules, then the SSRF gate, which resolves the address."""
        try:
            check_recommender(provider=provider, has_api_key=has_api_key, api_base=api_base, price=price)
        except ValueError as exc:
            raise InvalidAgentRecommenderError(str(exc)) from None
        if api_base is not None:
            try:
                await validate_provider_api_base(api_base)
            except UnsafeURLError as exc:
                raise AgentRecommenderUnsafeApiBaseError(str(exc)) from None

    @staticmethod
    def _encrypt(api_key: str | None) -> str | None:
        if api_key is None:
            return None
        try:
            return encrypt_secret(api_key)
        except SecretBoxUnavailableError:
            raise SecretBoxUnavailableTenancyError(_STORED) from None


def _or(value: float | None, stored: Decimal) -> Decimal:
    return stored if value is None else to_usd(value)
