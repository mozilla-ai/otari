"""Read-only operator settings for the admin dashboard.

Exposes a small, non-secret slice of gateway configuration (mode, version, and
the pricing-policy flags) so the dashboard can explain runtime behavior, such as
whether the genai-prices default fallback is active.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from gateway.api.deps import get_config, verify_master_key
from gateway.core.config import GatewayConfig
from gateway.version import __version__

router = APIRouter(prefix="/v1/settings", tags=["settings"])


class GatewaySettings(BaseModel):
    """Operator-facing runtime settings surfaced to the dashboard."""

    mode: str
    version: str
    default_pricing: bool
    require_pricing: bool


@router.get("", dependencies=[Depends(verify_master_key)])
async def get_settings(
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> GatewaySettings:
    """Return non-secret runtime settings for the admin dashboard."""
    return GatewaySettings(
        mode=config.effective_mode,
        version=__version__,
        default_pricing=config.default_pricing,
        require_pricing=config.require_pricing,
    )
