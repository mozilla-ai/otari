"""What the request or the app already has, that the rest of this package is built from.

Config, the enabled features, the container, the request's session and its Unit
of Work, and the shared resources the lifespan built. A builder of a domain's own
objects belongs in that domain's module, not here.
"""

from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.container import Container
from gateway.core.config import GatewayConfig
from gateway.core.database import get_db
from gateway.core.feature import CoreFeature
from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.api_keys import ApiKeyRepositories
from gateway.repositories.budgets import BudgetRepositories
from gateway.repositories.traces import TracesRepositories
from gateway.services.api_keys import ApiKeyMembershipListener
from gateway.services.budgets import BudgetMembershipListener
from gateway.services.log_writer import LogWriter
from gateway.services.routing import clear_router_backend_cache
from gateway.services.tenancy import MembershipListener, MembershipListeners
from gateway.services.tools import McpGrantMembershipListener
from gateway.services.traces import TraceMembershipListener

# Legacy module-level fallback. Config now lives on ``app.state.config`` (set in
# ``create_app``); ``get_config`` reads from the request's app state and only
# falls back to this shim for callers that set it directly (see ``set_config``).
_config: GatewayConfig | None = None


def set_config(config: GatewayConfig) -> None:
    """Set the legacy module-level config fallback.

    Compatibility shim. Config is stored on ``app.state.config`` by
    ``create_app``; this only updates the module-level fallback that
    ``get_config`` consults when no request-scoped app state is available.
    """
    global _config  # noqa: PLW0603
    _config = config


async def get_config(request: Request) -> GatewayConfig:
    """Return the config for the current app from ``request.app.state``.

    Every shared resource (rate limiter, log writer, file store) lives on
    ``app.state``; config does too, so two apps in one process no longer share
    a single instance. Falls back to the legacy module-level shim only when the
    app state has no config attached.
    """
    config: GatewayConfig | None = getattr(request.app.state, "config", None)
    if config is None:
        config = _config
    if config is None:
        msg = "Config not initialized"
        raise RuntimeError(msg)
    return config


def reset_config() -> None:
    """Reset the legacy module-level config fallback. Intended for testing only."""
    global _config  # noqa: PLW0603
    _config = None
    # Router backends are cached per config signature and hold a per-process
    # decision cache, so a test that swaps config must not inherit the previous
    # one's trace stickiness.
    clear_router_backend_cache()


def get_enabled_features(request: Request) -> tuple[CoreFeature, ...]:
    """Return the core features this app enabled when it was built."""
    enabled: tuple[CoreFeature, ...] | None = getattr(request.app.state, "enabled_features", None)
    if enabled is None:
        msg = "Enabled features not initialized"
        raise RuntimeError(msg)
    return enabled


def get_container(request: Request) -> Container:
    """Return the composition-root container this app was built with."""
    container: Container | None = getattr(request.app.state, "container", None)
    if container is None:
        msg = "Composition root not initialized"
        raise RuntimeError(msg)
    return container


ContainerDep = Annotated[Container, Depends(get_container)]
# The request's own session: FastAPI caches a dependency per callable, so a port
# resolved on this shares the session of the route that resolved it, and its
# writes land in the commit that settles the request.
PortSessionDep = Annotated[AsyncSession, Depends(get_db)]


def get_unit_of_work(db: Annotated[AsyncSession, Depends(get_db)]) -> UnitOfWork:
    """Return the request's Unit of Work over its session.

    NOTE: the rest of the request writes to this same session. A block's commit also stores
    what that code staged outside a block, and its rollback discards it.
    """
    return UnitOfWork(db)


UnitOfWorkDep = Annotated[UnitOfWork, Depends(get_unit_of_work)]


def get_membership_listener(uow: UnitOfWorkDep) -> MembershipListener:
    """Return the listener that keeps members' budget ceilings, own keys, session shares and MCP access in step.

    It writes through the request's Unit of Work, so a service that changes membership must be built on the same one.
    """
    return MembershipListeners(
        BudgetMembershipListener(BudgetRepositories.on(uow)),
        ApiKeyMembershipListener(ApiKeyRepositories.on(uow)),
        TraceMembershipListener(TracesRepositories.on(uow)),
        McpGrantMembershipListener(uow),
    )


MembershipListenerDep = Annotated[MembershipListener, Depends(get_membership_listener)]


async def get_log_writer(request: Request) -> LogWriter:
    """Return the usage-log writer this app's lifespan built."""
    writer: LogWriter = request.app.state.log_writer
    return writer
