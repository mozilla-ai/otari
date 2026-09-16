"""The shape of a core feature, as the registry lists it."""

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter

from gateway.core.config import GatewayConfig

Worker = Callable[[GatewayConfig], Coroutine[Any, Any, None]]


@dataclass(frozen=True)
class CoreFeature:
    """One feature the build ships.

    ``enabled`` is asked once per app, before the dashboard's stored settings are applied,
    and its answer holds for the life of the process.
    ``surface`` is the dashboard surface the feature hosts when enabled, or ``None`` for a feature with no page.
    ``worker``, when set, runs in the background while a standalone or hosted app serves, and is cancelled at shutdown.
    It must let that cancellation through and leave no write half done:
    shutdown waits only a few seconds before it abandons a worker and closes the database.
    A worker that raises is logged once and not restarted, and the feature's routes and page stay up.

    NOTE: ``enabled`` must read only settings the dashboard cannot change.
    A switch the dashboard can change never takes effect, not even after a restart:
    a feature switched off there keeps its routes, its page and its worker, and one switched on gets none of them.
    """

    name: str
    surface: str | None
    enabled: Callable[[GatewayConfig], bool]
    routers: Callable[[GatewayConfig], tuple[APIRouter, ...]]
    worker: Worker | None = None
