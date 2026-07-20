"""Operator settings for the admin dashboard.

Exposes a small, non-secret slice of gateway configuration (mode, version, and
the runtime-toggleable policy flags) so the dashboard can explain and control
runtime behavior, such as whether model discovery runs and whether the
genai-prices default fallback is active. The two toggleable flags are persisted
(see runtime_settings_service) so a change survives a restart.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, verify_master_key
from gateway.core.config import GatewayConfig
from gateway.services.runtime_settings_service import (
    DEFAULT_PRICING,
    MODEL_DISCOVERY,
    apply_override,
    stage_override,
)
from gateway.version import __version__

router = APIRouter(prefix="/v1/settings", tags=["settings"])


class GatewaySettings(BaseModel):
    """Operator-facing runtime settings surfaced to the dashboard."""

    mode: str
    version: str
    model_discovery: bool
    default_pricing: bool
    require_pricing: bool


class UpdateSettingsRequest(BaseModel):
    """Toggle one or more runtime settings. Omitted fields are left unchanged."""

    model_discovery: bool | None = None
    default_pricing: bool | None = None


def _current_settings(config: GatewayConfig) -> GatewaySettings:
    return GatewaySettings(
        mode=config.effective_mode,
        version=__version__,
        model_discovery=config.model_discovery,
        default_pricing=config.default_pricing,
        require_pricing=config.require_pricing,
    )


@router.get("", dependencies=[Depends(verify_master_key)])
async def get_settings(
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> GatewaySettings:
    """Return non-secret runtime settings for the admin dashboard."""
    return _current_settings(config)


@router.patch("", dependencies=[Depends(verify_master_key)])
async def update_settings(
    request: UpdateSettingsRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> GatewaySettings:
    """Persist and apply runtime setting toggles.

    Each provided flag is stored as an override (winning over config/env) and
    applied to the running gateway immediately. Master-key gated: these change
    how the gateway meters and lists models.
    """
    updates: dict[str, bool] = {}
    if request.model_discovery is not None:
        updates[MODEL_DISCOVERY] = request.model_discovery
    if request.default_pricing is not None:
        updates[DEFAULT_PRICING] = request.default_pricing

    if updates:
        try:
            for key, value in updates.items():
                await stage_override(db, key, value)
            await db.commit()
        except SQLAlchemyError:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error",
            ) from None
        # Apply only after the write has committed, so a failed commit never
        # leaves this worker metering or listing against an unpersisted value.
        for key, value in updates.items():
            apply_override(config, key, value)

    return _current_settings(config)
