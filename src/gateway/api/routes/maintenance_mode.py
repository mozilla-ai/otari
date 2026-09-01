"""The maintenance-mode switch: freeze and unfreeze dashboard sign-ins.

Two endpoints under ``/v1/settings``, operator-gated and standalone-only like
the rest of the management API. The state itself, and why it is stored the way
it is, belongs to ``services/maintenance_mode_service.py``.

It sits here rather than in ``settings.py`` because that module's payload is a
projection of ``GatewayConfig`` fields, and this is not one: it is deployment
state with no config field behind it, read per sign-in attempt rather than from
the running worker's config. ``mail.py`` already sets the precedent for a
``/v1/settings/*`` surface owning its own module.

``GET /v1/bootstrap`` publishes the same flag unauthenticated, so the sign-in
screen can say a deployment is down for maintenance rather than presenting a
form whose only outcome is a refusal. This pair is what an operator toggles it
with, and it is separate from the bootstrap because the settings page needs a
value that changes when the switch is flipped, while the bootstrap is read once
per page load.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db, require_deployment_operator
from gateway.log_config import logger
from gateway.services.maintenance_mode_service import is_maintenance_mode, stage_maintenance_mode

router = APIRouter(
    prefix="/v1/settings/maintenance-mode",
    tags=["settings"],
    dependencies=[Depends(require_deployment_operator)],
)


class MaintenanceMode(BaseModel):
    """Whether this deployment is currently refusing new dashboard sign-ins."""

    enabled: bool = Field(
        description=(
            "When true, POST /v1/auth/session refuses every credential with 503 so nobody starts "
            "a new dashboard session during a redeploy. Sessions already issued keep working, and "
            "the management API and the data plane are unaffected: a caller presenting the master "
            "key or an API key through the header is never frozen out."
        )
    )


class UpdateMaintenanceModeRequest(BaseModel):
    """Turn the sign-in freeze on or off."""

    enabled: bool = Field(description="True to freeze new dashboard sign-ins, false to allow them again.")


@router.get("")
async def get_maintenance_mode(db: Annotated[AsyncSession, Depends(get_db)]) -> MaintenanceMode:
    """Report whether new dashboard sign-ins are frozen."""
    return MaintenanceMode(enabled=await is_maintenance_mode(db))


@router.patch("")
async def update_maintenance_mode(
    request: UpdateMaintenanceModeRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MaintenanceMode:
    """Freeze or unfreeze dashboard sign-ins, for this and every other replica.

    The new state is persisted and nothing is applied to the running worker,
    because every reader goes back to the stored row. That is what makes one
    call enough for a deployment running more than one of them.
    """
    try:
        await stage_maintenance_mode(db, enabled=request.enabled)
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        logger.warning("Failed to persist the maintenance-mode flag", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    logger.info("Dashboard sign-in freeze set to %s", request.enabled)
    return MaintenanceMode(enabled=request.enabled)
