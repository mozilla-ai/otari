"""Pricing initialization from configuration."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import distinct, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.log_config import logger
from gateway.models.money import to_usd
from gateway.models.pricing import MODEL_KIND, ModelPricing, PricingKind
from gateway.repositories.pricing import OrganizationModelPricingRepository
from gateway.services.pricing_service import (
    GATEWAY_TOOL_PRICING_PROVIDER,
    gateway_tool_pricing_key,
    legacy_tool_price_keys,
    normalize_effective_at,
    resolve_tool_rate,
)
from gateway.services.provider_kwargs import normalize_pricing_key
from gateway.services.tools import configured_tools_to_price, deployment_tool_rate_keys, enriching_fetch_tool
from gateway.types.tool_charges import InstanceRateKey


async def warn_if_require_pricing_without_pricing(config: GatewayConfig, db: AsyncSession) -> None:
    """Warn at startup when require_pricing is on but no pricing is configured.

    With ``require_pricing=True`` (the default), any model lacking a pricing
    entry is rejected with HTTP 402 — so a deployment with zero pricing rows
    would reject every billable request. Surface that loudly rather than letting
    operators discover it via failed traffic.

    When community-maintained default pricing is enabled (opt-in), the dire "all
    requests rejected" warning no longer applies; a softer note is logged instead,
    since models outside models.dev coverage are still rejected.
    """
    if not config.require_pricing:
        return
    # Tool rows are excluded: pricing only ``otari:web_search`` or a search tool
    # leaves every *model* unpriced, which is precisely the state this warning
    # exists to announce. A tool's row written under its old model-kind key counts
    # as a tool's too.
    legacy = sorted(legacy_tool_price_keys(config))
    count = (
        await db.execute(
            select(func.count())
            .select_from(ModelPricing)
            .where(
                ModelPricing.kind == MODEL_KIND,
                ModelPricing.model_key.notlike(f"{GATEWAY_TOOL_PRICING_PROVIDER}:%"),
                ModelPricing.model_key.notin_(legacy),
            )
        )
    ).scalar_one()
    if count > 0:
        return
    if not config.default_pricing:
        logger.warning(
            "require_pricing is enabled but no model pricing is configured: ALL billable requests "
            f"will be rejected with HTTP 402. Add pricing (config `pricing` section or POST {API_ROOT}/pricing), "
            "set require_pricing=false, or add explicit $0 pricing for free/self-hosted models."
        )
    else:
        logger.warning(
            "require_pricing is enabled with no configured pricing; relying on default_pricing "
            "(models.dev) for billing. Models outside its coverage are still rejected with HTTP 402."
        )


async def warn_about_unpriced_web_tools(config: GatewayConfig, db: AsyncSession) -> None:
    """Warn at startup for each search and fetch tool, ``builtin_fetch`` included, with no price of its own.

    That row prices the tool's calls, in the loop and on the direct endpoint.
    With ``require_pricing`` on, a request that would run the tool is refused
    with HTTP 402, except on a key exempt from budgets. With it off, a call
    reserves nothing before the provider is called and settles at zero unless
    its provider reports a cost. A rate of 0 is a price. A fetch tool is checked
    where the fetch tool is enabled, and where it reads the pages of the in-loop
    search's hits, which take its price.
    """
    enriching = enriching_fetch_tool(config)
    keys = [
        key
        for key in deployment_tool_rate_keys(config)
        if key.kind == "search" or config.web_fetch_enabled or key.name == enriching
    ]
    unpriced = [key.name for key in keys if await resolve_tool_rate(db, (key,)) is None]
    if not unpriced:
        return
    if config.require_pricing:
        consequence = (
            "require_pricing is on, so a request that would run one is refused with HTTP 402, except on keys "
            "exempt from budgets. Set a rate, 0 included"
        )
    else:
        consequence = (
            "Their calls are recorded at zero cost unless the provider reports one, and nothing is reserved "
            "against a caller's budget before they run. Set a rate"
        )
    logger.warning(
        "No price is set for search or fetch tool(s): %s. %s: on the dashboard's Web search page, with POST "
        "%s/pricing and kind 'search' or 'fetch' with the tool's name as model_key, or with price_per_call "
        "(USD per call) on its entry if it is declared in config.yml.",
        ", ".join(sorted(unpriced)),
        consequence,
        API_ROOT,
    )


async def warn_if_gateway_tools_lack_pricing(config: GatewayConfig, db: AsyncSession) -> None:
    """Warn at startup when a configured gateway-run tool priced under ``otari:<tool>`` has no per-request rate.

    Those are code execution, and web search on the legacy settings or on the
    old backend; :func:`warn_about_unpriced_web_tools` names the search and
    fetch tools priced by their own row. With ``require_pricing`` on, an
    unpriced tool is refused at admission, so this warning is what tells the
    operator *before* the first 402 rather than after. With ``require_pricing``
    off the tool runs, and a call its provider reports no cost for is recorded at
    zero cost, which is worth flagging too: the work is real and the invoice is not.
    """
    unpriced = [
        tool.name
        for tool in configured_tools_to_price(config)
        if not isinstance(tool.rate_keys[0], InstanceRateKey)
        and await resolve_tool_rate(db, tool.rate_keys, label=tool.label, deployment_paid=tool.deployment_paid) is None
    ]
    if not unpriced:
        return

    keys = ", ".join(gateway_tool_pricing_key(tool) for tool in unpriced)
    if config.require_pricing:
        logger.warning(
            "Gateway tool(s) %s are configured but have no pricing (%s), and require_pricing is on, "
            "so requests using them will be rejected with HTTP 402. Add a per-request rate "
            f"(config `pricing` section or POST {API_ROOT}/pricing). The convention is USD per million "
            "requests, so 10000.0 charges $0.01 per call.",
            ", ".join(unpriced),
            keys,
        )
    else:
        logger.warning(
            "Gateway tool(s) %s are configured but have no pricing (%s). Their calls will be "
            "recorded at zero cost, unless their provider reports one, while still consuming operator "
            "resources. Add a per-request rate; the convention is USD per million requests, so 10000.0 "
            "charges $0.01 per call.",
            ", ".join(unpriced),
            keys,
        )


async def initialize_pricing_from_config(config: GatewayConfig, db: AsyncSession) -> None:
    """Initialize model pricing from configuration file."""

    if not config.pricing:
        logger.debug("No pricing configuration found in config file")
        return

    logger.info("Loading pricing configuration for %s model(s)", len(config.pricing))

    # A search or fetch instance's rate written here under its old
    # ``<provider>:<instance>`` key keeps loading, matched whole (the deprecation
    # warning names it); its provider is never under ``providers``.
    legacy_tool_keys = legacy_tool_price_keys(config)

    for raw_model_key, pricing_config in config.pricing.items():
        model_key = normalize_pricing_key(config, raw_model_key)
        instance = model_key.split(":", 1)[0] if ":" in model_key else model_key

        if (
            instance not in config.providers
            and instance not in config.decision_providers
            and model_key not in legacy_tool_keys
            and instance != GATEWAY_TOOL_PRICING_PROVIDER
        ):
            tools_under = sorted({key.split(":", 1)[1] for key in legacy_tool_keys if key.split(":", 1)[0] == instance})
            if tools_under:
                logger.warning(
                    "Skipping pricing for '%s': it names none of %s's search or fetch tools (%s), and '%s' is not "
                    "listed in the providers section. A tool is priced by price_per_call on its entry, or by "
                    "POST %s/pricing with its kind and its name as model_key.",
                    model_key,
                    instance,
                    ", ".join(tools_under),
                    instance,
                    API_ROOT,
                )
                continue
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
        pricing_tiers = [tier.model_dump(exclude_none=True) for tier in pricing_config.pricing_tiers]
        effective_at = normalize_effective_at(pricing_config.effective_at)

        existing_pricing = (
            await db.execute(
                select(ModelPricing).where(
                    ModelPricing.kind == MODEL_KIND,
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
            pricing_tiers=pricing_tiers,
            unit=pricing_config.unit,
            rates=None if pricing_config.rates is None else pricing_config.rates.document(),
            origin="config",
        )
        db.add(new_pricing)
        logger.info("Added pricing for '%s': input=$%s/M, output=$%s/M", model_key, input_price, output_price)

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise
    logger.info("Pricing initialization complete")


async def initialize_tool_prices_from_config(config: GatewayConfig, db: AsyncSession) -> None:
    """Store each configured search and fetch instance's ``price_per_call`` as its own price row.

    A new version only when the rate in force differs from the one configured,
    so an unchanged file adds nothing on a restart and a changed one takes effect.
    """
    sections: tuple[tuple[PricingKind, dict[str, Any]], ...] = (
        ("search", config.search_tools),
        ("fetch", config.fetch_tools),
    )
    configured = [
        (kind, name, to_usd(entry["price_per_call"]))
        for kind, entries in sections
        for name, entry in entries.items()
        if isinstance(entry, dict) and entry.get("price_per_call") is not None
    ]
    now = datetime.now(UTC)
    for kind, name, price in configured:
        per_million = price * 1_000_000
        latest = (
            await db.execute(
                select(ModelPricing)
                .where(ModelPricing.kind == kind, ModelPricing.model_key == name, ModelPricing.effective_at <= now)
                .order_by(ModelPricing.effective_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if latest is not None and latest.input_price_per_million == per_million and latest.unit == "request":
            continue
        db.add(
            ModelPricing(
                kind=kind,
                model_key=name,
                effective_at=now,
                input_price_per_million=per_million,
                output_price_per_million=Decimal(0),
                pricing_tiers=[],
                unit="request",
                origin="config",
            )
        )
        logger.info("Priced %s tool '%s' at $%s per call", kind, name, price)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise


async def warn_about_legacy_tool_prices(config: GatewayConfig, db: AsyncSession) -> None:
    """Warn at startup for a tool's price still stored under its old ``<provider>:<instance>`` key.

    Such a row still prices its tool, as a second spelling of the tool's own key,
    until a later release stops reading it. Names only, as stored.
    """
    legacy = sorted(legacy_tool_price_keys(config))
    if not legacy:
        return
    deployment = set(
        (
            await db.scalars(
                select(distinct(ModelPricing.model_key)).where(
                    ModelPricing.kind == MODEL_KIND, ModelPricing.model_key.in_(legacy)
                )
            )
        ).all()
    )
    overrides = {
        key for _id, _organization_id, key in await OrganizationModelPricingRepository(db).ids_for_keys(legacy)
    }
    if not deployment and not overrides:
        return
    logger.warning(
        "Search or fetch tool price(s) stored under the old <provider>:<instance> key: %s. They still price "
        "their tool until a later release stops reading them. Give each tool its own price, then delete the "
        f"old row: POST {API_ROOT}/pricing with kind 'search' or 'fetch' and the tool's name as model_key "
        "(builtin_fetch for the built-in fetcher), or price_per_call on its entry if it is declared in "
        f"config.yml. A deployment operator recreates an organization override the same way, on "
        f"{API_ROOT}/organizations/me/pricing in that organization.",
        ", ".join(
            [f"{key} (deployment)" for key in sorted(deployment)]
            + [f"{key} (organization override)" for key in sorted(overrides)]
        ),
    )
