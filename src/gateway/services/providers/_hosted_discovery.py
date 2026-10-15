"""The hosted providers' discovery dials, for a worker that has no request."""

from collections.abc import Callable

from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.services.model_discovery_service import DiscoveryDial, HostedDiscovery
from gateway.services.providers._hosted_provider_service import HostedProviderService


def hosted_discovery(build_service: Callable[[UnitOfWork], HostedProviderService]) -> HostedDiscovery:
    """Read the hosted providers' dials in a unit of work of each call's own.

    ``build_service`` is the builder in ``api/deps.py``, which the code that
    starts the worker passes in. The unit of work closes before any dial runs,
    so no session is held across an upstream call.
    """

    async def dials() -> dict[str, DiscoveryDial]:
        async with create_unit_of_work() as uow:
            return await build_service(uow).discovery_dials()

    return dials
