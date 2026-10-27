"""Whether this deployment checks otari.ai for new releases and security fixes (``/update-check``).

Nothing is sent until an operator answers here, or ``update_check`` answers in
config. A check sends this deployment's version and a random install id, once a
day. Operator-gated, and listed in ``CORE_FEATURES`` as ``FEATURE`` below.
"""

from fastapi import APIRouter, Depends

from gateway.api.deps import UpdateCheckServiceDep, build_update_check_service, require_deployment_operator
from gateway.core.config import GatewayConfig
from gateway.core.feature import CoreFeature
from gateway.core.surface import Surface
from gateway.core.unit_of_work import UnitOfWork
from gateway.schemas.update_check import UpdateCheckAnswer, UpdateCheckStatus
from gateway.services.update_check import UpdateCheckService, run_update_check_worker

router = APIRouter(
    prefix="/update-check",
    tags=["update-check"],
    dependencies=[Depends(require_deployment_operator)],
)


@router.get("")
async def get_update_check(service: UpdateCheckServiceDep) -> UpdateCheckStatus:
    """Whether checks are on and who decided, what a check sends, and what the last one heard."""
    return await service.get_status()


@router.put("")
async def answer_update_check(service: UpdateCheckServiceDep, body: UpdateCheckAnswer) -> UpdateCheckStatus:
    """Turn the checks on or off. A yes is checked within a minute.

    Refused with 409 when the deployment's configuration answers instead.
    """
    return await service.answer(body)


async def _poll(config: GatewayConfig) -> None:
    def build(uow: UnitOfWork) -> UpdateCheckService:
        return build_update_check_service(uow, config)

    await run_update_check_worker(build)


FEATURE = CoreFeature(
    name="update_check",
    surface=Surface("update_check"),
    enabled=lambda config: True,
    routers=lambda config: (router,),
    worker=_poll,
)
