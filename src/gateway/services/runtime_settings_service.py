"""Runtime-toggleable settings, persisted so a dashboard change survives restart.

A tiny layer over the ``runtime_settings`` table for the handful of config flags
the dashboard can flip: model discovery and the default-pricing fallback. A
stored override wins over the config-file/env value; it is loaded and applied on
startup and updated in place when the dashboard toggles it.

Scope note: applying an override mutates this worker's in-memory config and the
process-wide pricing flag, so it takes effect immediately here. Other workers or
replicas pick up the change on their next startup (the stored value is the source
of truth), not live; that is acceptable for these operator toggles.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import RuntimeSetting
from gateway.services.pricing_service import configure_default_pricing

MODEL_DISCOVERY = "model_discovery"
DEFAULT_PRICING = "default_pricing"

# The flags the dashboard may override, and how each is applied to the running
# gateway. Everything else stays config-file/env only.
SETTABLE_KEYS = (MODEL_DISCOVERY, DEFAULT_PRICING)


def _to_str(value: bool) -> str:
    return "true" if value else "false"


def _to_bool(value: str) -> bool:
    return value.strip().lower() == "true"


async def load_overrides(session: AsyncSession) -> dict[str, bool]:
    """Return every stored override as a ``{key: bool}`` map."""
    rows = (await session.execute(select(RuntimeSetting))).scalars().all()
    return {row.key: _to_bool(row.value) for row in rows if row.key in SETTABLE_KEYS}


def apply_override(config: GatewayConfig, key: str, value: bool) -> None:
    """Apply one override to the running gateway (mutating config + globals)."""
    if key == MODEL_DISCOVERY:
        config.model_discovery = value
    elif key == DEFAULT_PRICING:
        config.default_pricing = value
        configure_default_pricing(value)


async def apply_overrides_from_db(config: GatewayConfig, session: AsyncSession) -> None:
    """Load persisted overrides and apply them to ``config`` at startup."""
    overrides = await load_overrides(session)
    for key, value in overrides.items():
        apply_override(config, key, value)
        logger.info("Applied runtime setting override %s=%s", key, value)


async def set_override(session: AsyncSession, config: GatewayConfig, key: str, value: bool) -> None:
    """Persist an override and apply it to the running gateway.

    Caller is responsible for committing the session.
    """
    if key not in SETTABLE_KEYS:
        msg = f"Unknown runtime setting: {key!r}"
        raise ValueError(msg)
    row = await session.get(RuntimeSetting, key)
    if row is None:
        session.add(RuntimeSetting(key=key, value=_to_str(value)))
    else:
        row.value = _to_str(value)
    apply_override(config, key, value)
