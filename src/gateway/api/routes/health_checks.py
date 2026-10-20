"""The scheduled health checks, from the operator's Health page (``/health-checks``).

Every model the deployment's hosted providers serve is sent one small
completion on a schedule, and what each answered is kept for 48 hours. Not
``/health``, which answers whether this process is up. Operator-gated, and
listed in ``CORE_FEATURES`` as ``FEATURE`` below.

A failed check's error is the provider's message cut to its first line, with
anything shaped like a key or a bearer token replaced, and at most 300
characters (``services/health_checks/_probe.py``), before it is stored, shown
or emailed.
"""

from fastapi import APIRouter, Depends, status

from gateway.api.deps import (
    HealthCheckServiceDep,
    build_health_check_service,
    build_hosted_provider_service,
    require_deployment_operator,
)
from gateway.core.config import GatewayConfig
from gateway.core.feature import CoreFeature
from gateway.core.surface import Surface
from gateway.schemas.health_checks import (
    HealthCheckDisabled,
    HealthCheckSettingsPublic,
    HealthCheckSettingsUpdate,
    HealthCheckStatus,
)
from gateway.services.health_checks import run_health_check_worker

router = APIRouter(
    prefix="/health-checks",
    tags=["health-checks"],
    dependencies=[Depends(require_deployment_operator)],
)


@router.get("")
async def get_health_check_status(service: HealthCheckServiceDep) -> HealthCheckStatus:
    """The settings, the run in progress, and the latest finished run by provider and by offering."""
    return await service.get_status()


@router.post("/run", status_code=status.HTTP_202_ACCEPTED)
async def run_health_checks(service: HealthCheckServiceDep) -> HealthCheckSettingsPublic:
    """Start a run within a minute, whether or not the schedule is on."""
    return await service.request_run()


@router.post("/disable-failing")
async def disable_failing_models(service: HealthCheckServiceDep) -> HealthCheckDisabled:
    """Stop serving every model the latest run found failing, on its hosted provider.

    Each stays listed on the Hosted providers page, priced, to be switched back on there.
    """
    return HealthCheckDisabled(disabled=await service.disable_failing())


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


async def _poll(config: GatewayConfig) -> None:
    await run_health_check_worker(
        lambda uow: build_health_check_service(uow, config),
        lambda uow: build_hosted_provider_service(uow, config=config),
        config,
    )


FEATURE = CoreFeature(
    name="health_checks",
    surface=Surface("health_checks"),
    enabled=lambda config: config.health_checks_enabled,
    routers=lambda config: (router,),
    worker=_poll,
)
