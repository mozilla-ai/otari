"""Stored rules added to ``config.rate_limits``, which the request path reads on every request.

Loaded at startup, applied at once on the worker that served a write, and
reloaded on a timer so sibling workers and other replicas converge. A stored
rule whose name a config-file rule uses is skipped: config.yml is the whole
truth about the rules it declares.
"""

import asyncio
from collections.abc import Sequence

from gateway.core.config import GatewayConfig, RateLimitRule
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger
from gateway.models.rate_limits import StoredRateLimitRule
from gateway.repositories.rate_limits import RateLimitRuleRepository

# How long another worker may enforce a stale rule set, the TTL the other dashboard overlays use.
RATE_LIMIT_RULES_TTL_SECONDS = 30.0


def config_file_rules(config: GatewayConfig) -> list[RateLimitRule]:
    """The rules config.yml declares, with no stored rule added."""
    baseline = config._rate_limit_baseline
    return list(baseline if baseline is not None else config.rate_limits)


def rule_of(row: StoredRateLimitRule) -> RateLimitRule:
    """The ``RateLimitRule`` a stored row stands for."""
    return RateLimitRule(
        name=row.name,
        per=row.per,  # type: ignore[arg-type]  # the check constraint holds it to the Literal
        rpm=row.rpm,
        tpm=row.tpm,
        max_concurrent=row.max_concurrent,
        lease_sec=row.lease_sec,
    )


def apply_stored_rules(config: GatewayConfig, stored: Sequence[RateLimitRule]) -> list[str]:
    """Set ``config.rate_limits`` to the config-file rules plus ``stored``, returning the names skipped.

    The config-file rules are captured on the first call, so applying again
    after a stored rule is deleted takes it back out.
    """
    if config._rate_limit_baseline is None:
        config._rate_limit_baseline = list(config.rate_limits)
    baseline = config._rate_limit_baseline
    taken = {rule.name for rule in baseline}
    skipped = [rule.name for rule in stored if rule.name in taken]
    config.rate_limits = [*baseline, *(rule for rule in stored if rule.name not in taken)]
    return skipped


async def refresh_rate_limit_rules(uow: UnitOfWork, config: GatewayConfig) -> list[str]:
    """Reload the stored rules and apply them, returning the names skipped."""
    async with uow:
        rows = await RateLimitRuleRepository(uow).list_all()
        stored = [rule_of(row) for row in rows]
    return apply_stored_rules(config, stored)


async def load_rate_limit_rules_at_startup(config: GatewayConfig) -> None:
    """Apply the stored rules before the first request.

    A failure is logged rather than raised: the config-file rules still hold, and
    a gateway that enforces those is better than one that will not start.
    """
    try:
        async with create_unit_of_work() as uow:
            skipped = await refresh_rate_limit_rules(uow, config)
    except Exception:
        logger.exception("Failed to load stored rate limit rules; enforcing the config.yml rules only")
        return
    for name in skipped:
        logger.warning(
            "Stored rate limit rule '%s' is skipped: config.yml defines a rule of the same name.",
            name,
        )


async def run_rate_limit_refresher(config: GatewayConfig, interval: float = RATE_LIMIT_RULES_TTL_SECONDS) -> None:
    """Reload the stored rules forever, so another worker's writes arrive here within ``interval``.

    Every error is logged and retried on the next tick, so a database blip
    cannot stop the refresher and freeze the rule set. Cancelled at shutdown.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            async with create_unit_of_work() as uow:
                await refresh_rate_limit_rules(uow, config)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Stored rate limit rule refresh failed; retrying in %ss", interval, exc_info=True)
