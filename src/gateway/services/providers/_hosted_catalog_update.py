"""The catalog update: models.dev prices first, then what every hosted provider lists.

One operation behind the scheduled update, the Update catalog button and an
approval of reviewed prices, so the two halves never run out of order: the model
step prices new rows at the prices the first step just applied, and a row held
for want of a price is served once it has one. Removing the prices nothing
serves is opt-in and never scheduled, because it deletes price history.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol

from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.exceptions.providers_exceptions import HostedCatalogPricesChangedError, HostedCatalogPricesUnsavedError
from gateway.log_config import logger
from gateway.schemas.providers import (
    HostedCatalogPriceChangePublic,
    HostedCatalogPriceUpdatePublic,
    HostedCatalogRefreshPublic,
    HostedCatalogUpdatePreviewPublic,
    HostedCatalogUpdatePublic,
)
from gateway.services.pricing_refresh_service import PendingSnapshotChanged, PricingRefreshError, PricingRefreshPreview
from gateway.services.providers._hosted_provider_service import HostedProviderService

PRICES_UNREACHABLE = "Unable to fetch the latest models.dev prices, so only the model lists were checked."

# A policy and the interval it runs on, to what one scheduled run did, or None
# when another worker owns the run.
ScheduledRun = Callable[[str, float], Awaitable[str | None]]


class PriceUpdates(Protocol):
    """The models.dev price steps a catalog update runs, each committing on its own."""

    async def claim(self, interval_seconds: float) -> bool:
        """Whether this worker owns the next scheduled update."""
        ...

    async def fetch(self) -> PricingRefreshPreview | None:
        """Hold the latest prices as pending, or ``None`` when they change nothing."""
        ...

    async def poll(self, policy: str) -> str:
        """One scheduled price check: ``unchanged``, ``pending`` or ``applied``."""
        ...

    async def approve(self, digest: str) -> bool:
        """Apply the pending prices reviewed under ``digest``; false when none is pending."""
        ...


def _price_public(preview: PricingRefreshPreview) -> HostedCatalogPriceUpdatePublic:
    return HostedCatalogPriceUpdatePublic(
        fetched_at=preview.fetched_at,
        added_count=preview.added_count,
        changed_count=preview.changed_count,
        removed_count=preview.removed_count,
        changes=[HostedCatalogPriceChangePublic(model_key=c.model_key, change=c.change) for c in preview.changes],
        changes_truncated=preview.changes_truncated,
        digest=preview.digest,
        needs_review=preview.needs_review,
        review_reason=preview.review_reason,
    )


def _summary(prices: str, catalog: HostedCatalogRefreshPublic) -> str:
    added = sum(len(provider.added) for provider in catalog.providers)
    unreached = sum(1 for provider in catalog.providers if provider.error is not None)
    return f"prices {prices}; {added} models added; {unreached} providers not reached"


class HostedCatalogUpdate:
    """Preview, apply or schedule the catalog update."""

    def __init__(self, hosted: HostedProviderService, prices: PriceUpdates) -> None:
        self.hosted = hosted
        self.prices = prices

    async def preview(self, *, cleanup: bool) -> HostedCatalogUpdatePreviewPublic:
        """Fetch the latest prices and hold them, and dial every provider, writing nothing else.

        A failed fetch is reported in the answer rather than raised, so the
        model lists are still previewed.

        Raises:
            HostedCatalogEmptyError: ``cleanup`` was asked for while no provider offers a model.
        """
        price_error = None
        try:
            fetched = await self.prices.fetch()
        except PricingRefreshError:
            fetched, price_error = None, PRICES_UNREACHABLE
        catalog = await self.hosted.refresh_catalog(apply=False, remove=cleanup)
        return HostedCatalogUpdatePreviewPublic(
            prices=None if fetched is None else _price_public(fetched), price_error=price_error, catalog=catalog
        )

    async def apply(self, *, digest: str | None, cleanup: bool) -> HostedCatalogUpdatePublic:
        """Apply the prices reviewed under ``digest``, if any, then refresh every provider's models.

        Raises:
            HostedCatalogPricesChangedError: the pending prices are not the reviewed ones; nothing changed.
            HostedCatalogPricesUnsavedError: the prices could not be applied; nothing changed.
            HostedCatalogEmptyError: ``cleanup`` was asked for while no provider offers a model.
        """
        applied = False
        if digest is not None:
            try:
                applied = await self.prices.approve(digest)
            except PendingSnapshotChanged as exc:
                raise HostedCatalogPricesChangedError from exc
            except PricingRefreshError as exc:
                raise HostedCatalogPricesUnsavedError from exc
            if not applied:
                raise HostedCatalogPricesChangedError
        catalog = await self.hosted.refresh_catalog(apply=True, remove=cleanup)
        return HostedCatalogUpdatePublic(prices_applied=applied, catalog=catalog)

    async def run_scheduled(self, policy: str, interval_seconds: float) -> str | None:
        """One scheduled update under ``policy``, or ``None`` when another worker owns it.

        ``review`` leaves the prices pending and refreshes the model lists at
        the prices already in force; ``auto`` applies them first, unless they
        need review. A failed price fetch still refreshes the model lists.
        """
        if not await self.prices.claim(interval_seconds):
            return None
        try:
            prices = await self.prices.poll(policy)
        except PricingRefreshError:
            logger.warning("Scheduled catalog update could not fetch models.dev prices; refreshing models only")
            prices = "unreachable"
        catalog = await self.hosted.refresh_catalog(apply=True, remove=False)
        return _summary(prices, catalog)


async def run_catalog_updater(config: GatewayConfig, run_once: ScheduledRun) -> None:
    """Run the catalog update on ``pricing_refresh``'s schedule, forever.

    Every worker and replica runs this loop, and ``run_once`` claims each run
    so exactly one of them updates. The policy is read on every tick, because
    it is a runtime setting. Every error is logged and retried on the next tick.
    """
    while True:
        await asyncio.sleep(config.pricing_refresh_interval_seconds)
        if config.pricing_refresh == "manual":
            continue
        try:
            outcome = await run_once(config.pricing_refresh, config.pricing_refresh_interval_seconds)
        except asyncio.CancelledError:
            raise
        except DATABASE_ERRORS as exc:
            logger.warning("Scheduled catalog update failed (%s); retrying next interval", describe_database_error(exc))
        except Exception:
            logger.warning("Scheduled catalog update failed; retrying next interval", exc_info=True)
        else:
            if outcome is not None:
                logger.info("Scheduled catalog update: %s", outcome)
