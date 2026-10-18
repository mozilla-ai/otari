"""The scheduled health checks, from the operator's Health page (``/api/v1/health-checks``).

Every offering the deployment's own instances serve is sent one small
completion on a schedule, and what each answered is kept for 48 hours. Not
``/health``, which answers whether this process is up. Operator-gated and
standalone only.
"""

from fastapi import APIRouter, Depends, status

from gateway.api.deps import HealthCheckServiceDep, require_deployment_operator
from gateway.core.surface import Surface
from gateway.schemas.health_checks import (
    HealthCheckSettingsPublic,
    HealthCheckSettingsUpdate,
    HealthCheckStatus,
)

router = APIRouter(
    prefix="/health-checks",
    tags=["health-checks"],
    dependencies=[Depends(require_deployment_operator)],
)

SURFACE = Surface("health_checks", hosted=False)


@router.get("")
async def get_health_check_status(service: HealthCheckServiceDep) -> HealthCheckStatus:
    """The settings, the run in progress, and the latest finished run by provider and by offering."""
    return await service.get_status()


@router.post("/run", status_code=status.HTTP_202_ACCEPTED)
async def run_health_checks(service: HealthCheckServiceDep) -> HealthCheckSettingsPublic:
    """Start a run within a minute, whether or not the schedule is on."""
    return await service.request_run()


@router.get("/settings")
async def get_health_check_settings(service: HealthCheckServiceDep) -> HealthCheckSettingsPublic:
    """Whether the checks run on a schedule, how often, and what they send."""
    return await service.get_settings()


@router.patch("/settings")
async def update_health_check_settings(
    service: HealthCheckServiceDep, body: HealthCheckSettingsUpdate
) -> HealthCheckSettingsPublic:
    """Change the settings. Turning the schedule on starts a run within a minute."""
    return await service.update_settings(body)
