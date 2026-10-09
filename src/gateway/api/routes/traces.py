"""Agent traces as a core feature: recorded for every completion, and kept for their retention."""

from gateway.api.deps import build_trace_service
from gateway.core.config import GatewayConfig
from gateway.core.feature import CoreFeature
from gateway.services.traces import run_trace_retention

# Traces expire on the scale of days, so an hourly pass is soon enough.
_RETENTION_INTERVAL_S = 3600.0


async def _retention(config: GatewayConfig) -> None:
    await run_trace_retention(
        build_trace_service, retention_days=config.trace_retention_days, interval=_RETENTION_INTERVAL_S
    )


FEATURE = CoreFeature(
    name="traces",
    surface=None,
    enabled=lambda config: config.trace_capture_enabled,
    routers=lambda config: (),
    worker=_retention,
)
