"""The shape of a core feature package, as the registry lists it."""

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter

from gateway.core.config import GatewayConfig

Worker = Callable[[GatewayConfig], Coroutine[Any, Any, None]]


@dataclass(frozen=True)
class CorePackage:
    """One feature package the build ships.

    ``enabled`` is read at startup, when routers are mounted and workers are
    created, and again whenever the deployment bootstrap publishes the
    surfaces it hosts. It must therefore answer from startup-only settings;
    a runtime-settable field would let the published surface set drift from
    what is mounted and running.
    ``surface`` is the dashboard surface the package hosts when enabled, or
    ``None`` for a package with no page.
    """

    name: str
    surface: str | None
    enabled: Callable[[GatewayConfig], bool]
    routers: Callable[[GatewayConfig], tuple[APIRouter, ...]]
    worker: Worker | None = None
