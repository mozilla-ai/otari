"""What the health checks do about a hosted model its provider refuses to serve."""

from collections.abc import Callable

from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.services.health_checks._service import HealthCheckService
from gateway.services.inference import UnavailableModel


class UnavailableModelListener:
    """Switches off a hosted model when its provider refuses to serve it.

    A provider can stop serving a model it still lists, and every request for it then fails.
    Only a refusal on the deployment's own credential counts.
    What an organization's account may reach says nothing about the deployment's.
    """

    def __init__(self, build_service: Callable[[UnitOfWork], HealthCheckService]) -> None:
        self._build_service = build_service

    async def model_unavailable(self, unavailable_model: UnavailableModel) -> None:
        """Switch the model off and email the alert list, when the refused call used the deployment's credential."""
        if not unavailable_model.uses_deployment_credential:
            return
        async with create_unit_of_work() as uow:
            await self._build_service(uow).switch_off_unavailable(
                unavailable_model.instance, unavailable_model.model, unavailable_model.refusal
            )
