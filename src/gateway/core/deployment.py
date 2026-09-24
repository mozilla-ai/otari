"""What a deployment serves, and what a router needs it to serve."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntFlag, auto

from fastapi import APIRouter

from gateway.core.config import GatewayConfig


class Plane(IntFlag):
    """A plane a deployment may serve.

    ``CONTROL`` owns the tenancy, the policy and the database they live in.
    ``DATA`` runs customer traffic.
    One process may serve both.
    """

    CONTROL = auto()
    DATA = auto()


@dataclass(frozen=True)
class Deployment:
    """The planes this process serves."""

    planes: Plane

    def supports(self, required: Plane) -> bool:
        """Whether this deployment serves every plane in ``required``."""
        return required in self.planes


@dataclass(frozen=True)
class RouterMount:
    """One core router, and what a deployment must be for it to answer.

    ``requires`` names the planes without which the router has nothing to answer with.
    It defaults to none, which mounts the router on every deployment.
    ``when`` narrows that further on configuration, where a deployment may lack
    what the router needs to answer.
    """

    router: APIRouter
    requires: Plane = Plane(0)
    when: Callable[[GatewayConfig], bool] | None = None

    def applies_to(self, deployment: Deployment, config: GatewayConfig) -> bool:
        """Whether this router answers on ``deployment``."""
        if not deployment.supports(self.requires):
            return False
        return self.when is None or self.when(config)


def deployment_for(config: GatewayConfig) -> Deployment:
    """The planes ``config`` describes.

    A hosted control plane serves no data plane, because it owns many tenants'
    wallets and credentials while running none of their traffic: a completion
    served here would skip the usage report that debits the wallet.
    A hybrid gateway owns no control plane, because a peer answers for it.
    A standalone deployment serves both from the one process.
    """
    planes = Plane(0)
    if not config.is_hosted_mode:
        planes |= Plane.DATA
    if not config.is_hybrid_mode:
        planes |= Plane.CONTROL
    return Deployment(planes)
