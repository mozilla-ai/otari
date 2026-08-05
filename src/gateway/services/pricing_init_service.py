"""Pricing initialization from configuration."""

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.core.env import otari_env
from gateway.log_config import logger
from gateway.models.entities import ModelPricing
from gateway.services.pricing_service import (
    GATEWAY_TOOL_PRICING_PROVIDER,
    find_model_pricing,
    gateway_tool_pricing_key,
    normalize_effective_at,
)
from gateway.services.provider_kwargs import normalize_pricing_key
from gateway.services.sandbox_backend import CODE_EXECUTION_TOOL_NAME
from gateway.services.web_search_backend import WEB_SEARCH_TOOL_NAME


async def warn_if_require_pricing_without_pricing(config: GatewayConfig, db: AsyncSession) -> None:
    """Warn at startup when require_pricing is on but no pricing is configured.

    With ``require_pricing=True`` (the default), any model lacking a pricing
    entry is rejected with HTTP 402 — so a deployment with zero pricing rows
    would reject every billable request. Surface that loudly rather than letting
    operators discover it via failed traffic.

    When community-maintained default pricing is enabled (opt-in), the dire "all
    requests rejected" warning no longer applies; a softer note is logged instead,
    since models outside genai-prices coverage are still rejected.
    """
    if not config.require_pricing:
        return
    # Tool rows are excluded: pricing only ``otari:web_search`` leaves every *model*
    # unpriced, which is precisely the state this warning exists to announce.
    count = (
        await db.execute(
            select(func.count())
            .select_from(ModelPricing)
            .where(ModelPricing.model_key.notlike(f"{GATEWAY_TOOL_PRICING_PROVIDER}:%"))
        )
    ).scalar_one()
    if count > 0:
        return
    if not config.default_pricing:
        logger.warning(
            "require_pricing is enabled but no model pricing is configured: ALL billable requests "
            "will be rejected with HTTP 402. Add pricing (config `pricing` section or POST /v1/pricing), "
            "set require_pricing=false, or add explicit $0 pricing for free/self-hosted models."
        )
    else:
        logger.warning(
            "require_pricing is enabled with no configured pricing; relying on default_pricing "
            "(genai-prices) for billing. Models outside its coverage are still rejected with HTTP 402."
        )


async def warn_if_search_tools_lack_flat_pricing(config: GatewayConfig, db: AsyncSession) -> None:
    """Warn at startup for a search tool with no flat per-request rate.

    ``POST /v1/search`` reserves the rate configured for ``<provider>:<tool>``
    before calling the provider, so an unpriced tool reserves nothing: a user
    already over their cap is still refused, but one just under it can overshoot
    by a search, and concurrent searches cannot see each other's holds. The
    provider's own reported charge is still billed afterwards, so spend stays
    truthful; it is the pre-flight hold that is missing. An explicit rate of 0 is
    a deliberate "this tool is free" and is left alone.
    """
    unpriced: list[str] = []
    for name, entry in config.search_tools.items():
        provider = str(entry.get("provider") or name) if isinstance(entry, dict) else name
        # use_defaults=False for the same reason the route sets it: a search tool
        # is not a model, so the community dataset can only produce a false match.
        if await find_model_pricing(db, provider, name, use_defaults=False) is None:
            unpriced.append(f"{provider}:{name}")
    if not unpriced:
        return
    logger.warning(
        "No flat per-request rate is configured for search tool(s): %s. Nothing is reserved against a "
        "caller's budget before the search runs, so a user just under their cap can overshoot by one "
        "search. Add a rate for the model key (config `pricing` section or POST /v1/pricing); the "
        "convention is USD per million requests, so 5000.0 charges $0.005 per search.",
        ", ".join(sorted(unpriced)),
    )


async def warn_if_gateway_tools_lack_pricing(config: GatewayConfig, db: AsyncSession) -> None:
    """Warn at startup when a configured gateway-run tool has no per-request rate.

    A gateway-run tool (``otari_web_search`` / ``otari_code_execution``) is priced
    under ``otari:<tool>``. With ``require_pricing`` on, an unpriced tool is refused
    at admission, so this warning is what tells the operator *before* the first 402
    rather than after. With ``require_pricing`` off the tool runs and is recorded at
    zero cost, which is worth flagging too: the work is real and the invoice is not.
    """
    configured: list[str] = []
    if config.web_search_url or otari_env("WEB_SEARCH_URL"):
        configured.append(WEB_SEARCH_TOOL_NAME)
    if config.sandbox_url or otari_env("SANDBOX_URL"):
        configured.append(CODE_EXECUTION_TOOL_NAME)

    unpriced = [
        tool
        for tool in configured
        # use_defaults=False for the reason the settlement path sets it: a tool is
        # not a model, so the community dataset can only produce a false match.
        if await find_model_pricing(db, GATEWAY_TOOL_PRICING_PROVIDER, tool, use_defaults=False) is None
    ]
    if not unpriced:
        return

    keys = ", ".join(gateway_tool_pricing_key(tool) for tool in unpriced)
    if config.require_pricing:
        logger.warning(
            "Gateway tool(s) %s are configured but have no pricing (%s), and require_pricing is on, "
            "so requests using them will be rejected with HTTP 402. Add a per-request rate "
            "(config `pricing` section or POST /v1/pricing). The convention is USD per million "
            "requests, so 10000.0 charges $0.01 per call.",
            ", ".join(unpriced),
            keys,
        )
    else:
        logger.warning(
            "Gateway tool(s) %s are configured but have no pricing (%s). Their calls will be "
            "recorded at zero cost while still costing the operator money upstream. Add a "
            "per-request rate; the convention is USD per million requests, so 10000.0 charges "
            "$0.01 per call.",
            ", ".join(unpriced),
            keys,
        )


async def initialize_pricing_from_config(config: GatewayConfig, db: AsyncSession) -> None:
    """Initialize model pricing from configuration file."""

    if not config.pricing:
        logger.debug("No pricing configuration found in config file")
        return

    logger.info("Loading pricing configuration for %s model(s)", len(config.pricing))

    for raw_model_key, pricing_config in config.pricing.items():
        model_key = normalize_pricing_key(config, raw_model_key)
        instance = model_key.split(":", 1)[0] if ":" in model_key else model_key

        if instance not in config.providers and instance != GATEWAY_TOOL_PRICING_PROVIDER:
            logger.warning(
                "Skipping pricing for '%s': provider '%s' is not listed in the providers section. "
                "The provider may still work if its credentials come from the environment, but its "
                "pricing is ignored. Add '%s' to the providers section, or remove this pricing entry.",
                model_key,
                instance,
                instance,
            )
            continue

        input_price = pricing_config.input_price_per_million
        output_price = pricing_config.output_price_per_million
        cache_read_price = pricing_config.cache_read_price_per_million
        cache_write_price = pricing_config.cache_write_price_per_million
        cache_write_1h_price = pricing_config.cache_write_1h_price_per_million
        pricing_tiers = [tier.model_dump(exclude_none=True) for tier in pricing_config.pricing_tiers]
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
            cache_read_price_per_million=cache_read_price,
            cache_write_price_per_million=cache_write_price,
            cache_write_1h_price_per_million=cache_write_1h_price,
            pricing_tiers=pricing_tiers,
        )
        db.add(new_pricing)
        logger.info("Added pricing for '%s': input=$%s/M, output=$%s/M", model_key, input_price, output_price)

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise
    logger.info("Pricing initialization complete")
