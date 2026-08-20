"""The caller's organization's model rate overrides (standalone mode only).

Thin composition over `gateway.services.organization_pricing_service`: resolve
the caller's identity, call the service, return its typed result. The overlap
rule and the role gate live there, and the domain errors it raises carry their
own statuses (see `gateway.services.tenancy.errors`), so nothing here catches
them.

Scoped to ``/me`` for the same reason `routes/organizations.py` is: a request
cannot name an organization at all, because a standalone deployment has exactly
one and the caller's identity already points at it. Multi-organization
administration is the overlay's to contribute.

These rates sit *above* ``/v1/pricing``, which stays the deployment price list.
A model with no override here prices exactly as it did before, and the resolution
order (override, deployment row, genai-prices dataset) is
`services.pricing_service.find_model_pricing`.
"""

import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import CurrentIdentity, get_db, verify_master_key

# The tier shape comes from the deployment pricing route rather than a second
# copy here. An override resolves into a transient ``ModelPricing`` and is read by
# the same cost-math core, so a tier that meant something different on this
# surface would be a silent mispricing.
from gateway.api.routes.pricing import PricingTier
from gateway.models.entities import OrganizationModelPricing
from gateway.services.organization_pricing_service import (
    OrganizationPricingService,
    PricingOverrideInput,
)

# Master key on the router, as every standalone management router declares it.
# The role gate is a separate question answered in the service: the credential
# says a request is the operator's, the membership says whether that identity may
# change what the organization is billed.
router = APIRouter(
    prefix="/v1/organizations/me/pricing",
    tags=["organization-pricing"],
    dependencies=[Depends(verify_master_key)],
)

_MODEL_KEY_DESCRIPTION = (
    "Model identifier in 'provider:model' form, matching the key the deployment price list uses. "
    "A provider instance name is valid here ('home_lab:llama-3'), because pricing keys on the "
    "instance a request resolves to."
)
_EFFECTIVE_FROM_DESCRIPTION = "ISO 8601 datetime from which this rate applies, inclusive. Defaults to now."
_EFFECTIVE_TO_DESCRIPTION = (
    "ISO 8601 datetime at which this rate stops applying, exclusive. Null leaves it open ended. "
    "Because the end is exclusive, the next period may begin at exactly this instant without overlapping."
)


class OrganizationModelPricingRates(BaseModel):
    """The rates and period shared by the create and update bodies."""

    input_price_per_million: float = Field(ge=0, description="Price per 1M input tokens")
    output_price_per_million: float = Field(ge=0, description="Price per 1M output tokens")
    cache_read_price_per_million: float | None = Field(
        default=None, ge=0, description="Price per 1M cached-input tokens"
    )
    cache_write_price_per_million: float | None = Field(
        default=None, ge=0, description="Price per 1M cache-write (creation) tokens"
    )
    cache_write_1h_price_per_million: float | None = Field(
        default=None, ge=0, description="Price per 1M Anthropic 1-hour cache-write tokens"
    )
    pricing_tiers: list[PricingTier] | None = Field(
        default=None,
        description="Whole-request context thresholds. Fields omitted by a tier inherit the base rate.",
    )
    effective_from: datetime | None = Field(default=None, description=_EFFECTIVE_FROM_DESCRIPTION)
    effective_to: datetime | None = Field(default=None, description=_EFFECTIVE_TO_DESCRIPTION)


class OrganizationModelPricingCreate(OrganizationModelPricingRates):
    """Create one rate override for a model, for a period."""

    model_key: str = Field(min_length=1, max_length=255, description=_MODEL_KEY_DESCRIPTION)


class OrganizationModelPricingUpdate(OrganizationModelPricingRates):
    """Replace an override's rates and period.

    A full replacement rather than a patch: every rate field is present in the
    body and an omitted optional rate is cleared, so the stored row is exactly
    what was sent. That is the opposite of ``POST /v1/pricing``, which inherits an
    omitted cache rate from the model's previous version, and deliberately so:
    that surface versions a catalog where each write adds a row, while this one
    edits a single row in place and an inheriting patch would make the result
    depend on what happened to be stored before.

    ``model_key`` is absent because it is immutable. Repointing an override at
    another model is retiring one and creating another, which is two requests.
    """


class OrganizationModelPricingPublic(BaseModel):
    """One stored rate override."""

    id: uuid.UUID
    organization_id: uuid.UUID
    model_key: str
    input_price_per_million: float
    output_price_per_million: float
    cache_read_price_per_million: float | None
    cache_write_price_per_million: float | None
    cache_write_1h_price_per_million: float | None
    pricing_tiers: list[PricingTier]
    effective_from: datetime
    effective_to: datetime | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_model(cls, override: OrganizationModelPricing) -> "OrganizationModelPricingPublic":
        """Build the response from a stored row."""
        return cls(
            id=override.id,
            organization_id=override.organization_id,
            model_key=override.model_key,
            input_price_per_million=override.input_price_per_million,
            output_price_per_million=override.output_price_per_million,
            cache_read_price_per_million=override.cache_read_price_per_million,
            cache_write_price_per_million=override.cache_write_price_per_million,
            cache_write_1h_price_per_million=override.cache_write_1h_price_per_million,
            pricing_tiers=[PricingTier.model_validate(tier) for tier in override.pricing_tiers or []],
            effective_from=override.effective_from,
            effective_to=override.effective_to,
            created_at=override.created_at,
            updated_at=override.updated_at,
        )


class OrganizationModelPricingsPublic(BaseModel):
    """Every override in the organization, with a count.

    The envelope shape the platform's equivalent endpoint returns, kept so the
    generated dashboard client stays recognizable across both trees.
    """

    data: list[OrganizationModelPricingPublic]
    count: int


def get_organization_pricing_service(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrganizationPricingService:
    """Build the pricing service on the request's session."""
    return OrganizationPricingService(db)


ServiceDep = Annotated[OrganizationPricingService, Depends(get_organization_pricing_service)]


def _to_input(body: OrganizationModelPricingRates) -> PricingOverrideInput:
    """Translate a validated request body into the service's input.

    ``effective_from`` defaults to now here rather than in the service, so the
    instant a period starts is the one the request arrived at and is echoed back
    in the response.
    """
    return PricingOverrideInput(
        input_price_per_million=body.input_price_per_million,
        output_price_per_million=body.output_price_per_million,
        cache_read_price_per_million=body.cache_read_price_per_million,
        cache_write_price_per_million=body.cache_write_price_per_million,
        cache_write_1h_price_per_million=body.cache_write_1h_price_per_million,
        pricing_tiers=[tier.model_dump(exclude_none=True) for tier in body.pricing_tiers or []],
        effective_from=body.effective_from or datetime.now(tz=UTC),
        effective_to=body.effective_to,
    )


async def _commit(db: AsyncSession) -> None:
    """Commit the request's work, mapping a database failure to a 500.

    The services flush rather than commit (the house contract), so the route owns
    the transaction boundary.
    """
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None


@router.get("")
async def list_organization_pricing(
    identity: CurrentIdentity,
    service: ServiceDep,
) -> OrganizationModelPricingsPublic:
    """List the organization's rate overrides.

    Readable by any member: these rates decide what the caller's own requests
    cost, so they are not withheld from the people billed at them. Writing needs
    an owner or admin.
    """
    overrides = await service.list_for_caller(identity)
    return OrganizationModelPricingsPublic(
        data=[OrganizationModelPricingPublic.from_model(override) for override in overrides],
        count=len(overrides),
    )


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_organization_pricing(
    body: OrganizationModelPricingCreate,
    identity: CurrentIdentity,
    service: ServiceDep,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrganizationModelPricingPublic:
    """Set the organization's rate for a model over a period.

    Refused with a 409 when the period overlaps one already stored for that model,
    naming the period it collides with, rather than shadowing it.
    """
    override = await service.create_for_caller(identity, body.model_key, _to_input(body))
    response = OrganizationModelPricingPublic.from_model(override)
    await _commit(db)
    return response


@router.put("/{pricing_id}")
async def replace_organization_pricing(
    pricing_id: uuid.UUID,
    body: OrganizationModelPricingUpdate,
    identity: CurrentIdentity,
    service: ServiceDep,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrganizationModelPricingPublic:
    """Replace an override's rates and period.

    Future requests in the period price at the new rate; usage already settled
    keeps the cost it was billed, because a settled cost is stored on the usage
    row rather than recomputed.
    """
    override = await service.replace_for_caller(identity, pricing_id, _to_input(body))
    response = OrganizationModelPricingPublic.from_model(override)
    await _commit(db)
    return response


@router.delete("/{pricing_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_organization_pricing(
    pricing_id: uuid.UUID,
    identity: CurrentIdentity,
    service: ServiceDep,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Remove an override, returning the model to the deployment price list."""
    await service.delete_for_caller(identity, pricing_id)
    await _commit(db)
