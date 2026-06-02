"""Pricing initialization from configuration."""

from any_llm import AnyLLM
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import ModelPricing
from gateway.services.pricing_service import normalize_effective_at


async def warn_if_require_pricing_without_pricing(config: GatewayConfig, db: AsyncSession) -> None:
    """Warn at startup when require_pricing is on but no pricing is configured.

    With ``require_pricing=True`` (the default), any model lacking a pricing
    entry is rejected with HTTP 402 — so a deployment with zero pricing rows
    would reject every billable request. Surface that loudly rather than letting
    operators discover it via failed traffic.
    """
    if not config.require_pricing:
        return
    count = (await db.execute(select(func.count()).select_from(ModelPricing))).scalar_one()
    if count == 0:
        logger.warning(
            "require_pricing is enabled but no model pricing is configured: ALL billable requests "
            "will be rejected with HTTP 402. Add pricing (config `pricing` section or POST /v1/pricing), "
            "set require_pricing=false, or add explicit $0 pricing for free/self-hosted models."
        )


async def initialize_pricing_from_config(config: GatewayConfig, db: AsyncSession) -> None:
    """Initialize model pricing from configuration file."""

    if not config.pricing:
        logger.debug("No pricing configuration found in config file")
        return

    logger.info("Loading pricing configuration for %s model(s)", len(config.pricing))

    for raw_model_key, pricing_config in config.pricing.items():
        provider, model_name = AnyLLM.split_model_provider(raw_model_key)
        model_key = f"{provider.value}:{model_name}"

        if provider.value not in config.providers:
            msg = (
                f"Cannot set pricing for model '{model_key}': "
                f"provider '{provider}' is not configured in the providers section"
            )
            raise ValueError(msg)

        input_price = pricing_config.input_price_per_million
        output_price = pricing_config.output_price_per_million
        effective_at = normalize_effective_at(pricing_config.effective_at)

        existing_pricing = (
            await db.execute(
                select(ModelPricing).where(
                    ModelPricing.model_key == model_key,
                    ModelPricing.effective_at == effective_at,
                )
            )
        ).scalar_one_or_none()

        if existing_pricing:
            logger.warning(
                f"Pricing for model '{model_key}' effective {effective_at.isoformat()} already exists in database. "
                f"Keeping database value (input: ${existing_pricing.input_price_per_million}/M, "
                f"output: ${existing_pricing.output_price_per_million}/M). "
                f"To update, use the pricing API or delete the existing entry."
            )
            continue

        new_pricing = ModelPricing(
            model_key=model_key,
            effective_at=effective_at,
            input_price_per_million=input_price,
            output_price_per_million=output_price,
        )
        db.add(new_pricing)
        logger.info("Added pricing for '%s': input=$%s/M, output=$%s/M", model_key, input_price, output_price)

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise
    logger.info("Pricing initialization complete")
