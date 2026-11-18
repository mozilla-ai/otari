"""When a stored file stops being served."""

from datetime import UTC, datetime, timedelta

from gateway.core.config import GatewayConfig


def expiry_for(config: GatewayConfig, now: datetime | None = None) -> datetime | None:
    """When a file stored now stops being served, or ``None`` when files are kept indefinitely."""
    if config.files_retention_hours is None:
        return None
    return (now or datetime.now(UTC)) + timedelta(hours=config.files_retention_hours)
