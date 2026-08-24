"""Composition root: the one place that names a concrete adapter.

The container is a process-level registry of ``Port -> factory`` bindings,
built once per app in ``create_app`` and read per request through
``gateway.api.deps``. It is a plain mapping, deliberately not a
dependency-injection framework and not entry-point auto-discovery: only a
handful of ports ever need swapping, and only at startup, so the whole wiring
is readable in this one file and there is no install-time magic to trace
(``ARCHITECTURE.md``, "How a port is resolved").

Every port is bound here to a working core adapter, real or Null Object, so
Otari runs with no overlay present and behaves as it does today. Only real
ports belong here. A plain, single-implementation service stays wired directly
as an ordinary FastAPI dependency; routing one through the container would
claim a swap point that does not exist.

An overlay rebinds ports without editing any Otari source file, by pointing
``OTARI_BOOTSTRAP`` at a ``module:callable`` selector. The callable receives
this container after the core defaults are bound, and may rebind any port and
contribute routers of its own. Unset, the defaults stand.
"""

import importlib
import inspect
from collections.abc import Callable, ItemsView
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from fastapi import APIRouter
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.adapters.billing_adapter import NullBillingAdapter
from gateway.adapters.entitlement_adapter import BaseEntitlementAdapter
from gateway.adapters.growth_signal_adapter import NullGrowthSignalAdapter
from gateway.adapters.model_provider_adapter import SelfHostedModelProviderAdapter
from gateway.log_config import logger
from gateway.ports.billing_port import BillingPort
from gateway.ports.entitlement_port import EntitlementPort
from gateway.ports.growth_signal_port import GrowthSignalPort
from gateway.ports.model_provider_port import ModelProviderPort

T = TypeVar("T")

# The registry key: the port's ``Protocol`` class object itself, so a caller
# names the port and nothing else. Spelled ``Callable[..., T]`` rather than the
# more obvious ``type[T]`` because a Protocol class is abstract, and mypy
# refuses one where ``type[T]`` is expected on the assumption that the argument
# is about to be instantiated. A key is never instantiated; the callable form
# keys on the same class object and still carries ``T`` through to
# :meth:`Container.resolve`.
PortKey = Callable[..., T]
# A port is resolved per request against that request's database session; an
# adapter that needs no session ignores it. The session is ``None`` in hybrid
# mode, where the gateway has no local database at all (``init_db`` is skipped,
# see ``gateway.main``) and its control plane lives at the other end of the
# resolve protocol, so an adapter that needs one must say what it does without.
PortFactory = Callable[[AsyncSession | None], T]
Register = Callable[["Container"], None]


@dataclass(frozen=True)
class RouterContribution:
    """One router an overlay mounts on top of Otari's own.

    The additive half of the seam: nothing is swapped here, a surface is added
    and made conditional. Every route the router exposes is served only when
    the deployment is entitled to ``capability``, resolved through
    ``EntitlementPort``, because hiding a link in a dashboard is not
    authorization and a route mounted into this process has to refuse for
    itself.

    **Entitlement is not authentication, and the mount point adds none.**
    ``capability`` answers "is this build licensed for this surface", a
    deployment-wide question that names no caller, so on an entitled deployment
    a contributed route is reachable by anyone unless the router says otherwise.
    Declare the credential each route needs on the route, the way Otari's own
    routers do (``verify_master_key`` or ``verify_api_key_or_master_key`` per
    route in ``gateway.api.routes``); Otari mounts no router-level default here
    because there is no single right answer to mount. The choice differs per
    route, a contributed route may be deliberately public, and
    ``verify_api_key_or_master_key`` resolves ``get_db``, which has no session
    to open in hybrid mode.
    """

    capability: str
    router: APIRouter


class ContainerError(Exception):
    """Base error for composition-root wiring failures."""


class PortNotBoundError(ContainerError):
    """Raised when a port is resolved but no adapter has been bound to it."""

    def __init__(self, port: PortKey[Any]) -> None:
        name = getattr(port, "__name__", repr(port))
        super().__init__(f"No adapter is bound for port {name}")
        self.port = port


class BootstrapError(ContainerError):
    """Raised when the bootstrap module ``OTARI_BOOTSTRAP`` names cannot be loaded."""


class Container:
    """A registry mapping each port to the adapter that satisfies it.

    Built once per app with the core bindings plus whatever the configured
    bootstrap rebinds, then read per request. Attached to ``app.state`` rather
    than kept module-global, so two apps in one process (as the test suite
    builds) never share one.
    """

    def __init__(self) -> None:
        self._factories: dict[Any, PortFactory[Any]] = {}
        self._router_contributions: list[RouterContribution] = []
        # One line naming what this container was built from, logged by
        # build_container and asserted on by tests.
        self.summary = "unbuilt"

    def bind(self, port: PortKey[T], factory: PortFactory[T]) -> None:
        """Bind ``port`` to ``factory``, so a bootstrap can replace a core default.

        A later bind for the same port replaces an earlier one.
        """
        self._factories[port] = factory

    def bindings(self) -> ItemsView[Any, PortFactory[Any]]:
        """Return a snapshot of the (port, factory) pairs bound so far.

        A snapshot rather than a live view, so iterating it stays safe while a
        bootstrap binds. For the composition root itself, which compares a
        bootstrap's bindings against the defaults to report what it rebound.
        Not a resolution path; callers use :meth:`resolve`.
        """
        return dict(self._factories).items()

    def resolve(self, port: PortKey[T], session: AsyncSession | None) -> T:
        """Return the adapter bound to ``port``, built for this request's session.

        Raises:
            PortNotBoundError: If no adapter has been bound to ``port``.

        """
        factory = self._factories.get(port)
        if factory is None:
            raise PortNotBoundError(port)
        return cast(T, factory(session))

    def contribute_router(self, contribution: RouterContribution) -> None:
        """Record a router this build mounts on top of Otari's own."""
        self._router_contributions.append(contribution)

    def router_contributions(self) -> tuple[RouterContribution, ...]:
        """Return the recorded router contributions, in contribution order."""
        return tuple(self._router_contributions)


def _billing_adapter(session: AsyncSession | None) -> BillingPort:
    """Build the core ``BillingPort`` adapter for one request."""
    return NullBillingAdapter(session)


def _entitlement_adapter(session: AsyncSession | None) -> EntitlementPort:
    """Build the core ``EntitlementPort`` adapter for one request."""
    return BaseEntitlementAdapter(session)


def _model_provider_adapter(session: AsyncSession | None) -> ModelProviderPort:
    """Build the core ``ModelProviderPort`` adapter for one request."""
    return SelfHostedModelProviderAdapter(session)


def _growth_signal_adapter(session: AsyncSession | None) -> GrowthSignalPort:
    """Build the core ``GrowthSignalPort`` adapter for one request."""
    return NullGrowthSignalAdapter(session)


def _load_register(selector: str) -> Register:
    """Load the register callable a ``module:callable`` selector names.

    Surrounding whitespace is ignored. A bootstrap module that exists but fails
    to import is reported as its own failure, distinct from a selector naming a
    module that is not there.

    Raises:
        BootstrapError: If the selector is malformed, its module or attribute
            cannot be imported, or the attribute is not callable or is a
            coroutine function.

    """
    module_path, separator, attribute = selector.strip().partition(":")
    if not separator or not module_path or not attribute:
        msg = f"OTARI_BOOTSTRAP must be 'module:callable', got {selector!r}"
        raise BootstrapError(msg)

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as error:
        # The named module (or a package on its path) being absent is a wrong
        # selector; anything else missing is a broken bootstrap.
        missing = error.name or ""
        if module_path == missing or module_path.startswith(missing + "."):
            msg = f"Bootstrap module {module_path!r} was not found"
        else:
            msg = f"Bootstrap module {module_path!r} failed to import"
        raise BootstrapError(msg) from error
    except ImportError as error:
        msg = f"Bootstrap module {module_path!r} failed to import"
        raise BootstrapError(msg) from error

    register = getattr(module, attribute, None)
    if register is None:
        msg = f"Bootstrap module {module_path!r} has no attribute {attribute!r}"
        raise BootstrapError(msg)
    if not callable(register):
        msg = f"Bootstrap {selector!r} is not callable"
        raise BootstrapError(msg)
    if inspect.iscoroutinefunction(register):
        # An ``async def register`` is callable, so it passes the check above,
        # and calling it only builds a coroutine nobody awaits: every bind and
        # every contribution in it is silently dropped and the gateway serves
        # the plain build. That is the one outcome this whole path exists to
        # prevent, and it is an easy mistake to make when every port method is
        # async, so it is refused by name rather than left to a stray
        # "coroutine was never awaited" warning in the startup log.
        msg = f"Bootstrap {selector!r} is async; the container is built synchronously, so register must be a plain def"
        raise BootstrapError(msg)

    return cast(Register, register)


def build_container(bootstrap_selector: str | None = None) -> Container:
    """Build the composition-root container for this deployment.

    Binds the core adapters, then, if a selector is given, lets the bootstrap it
    names rebind ports and contribute routers. With no selector the core
    defaults stand and Otari boots standalone.

    Raises:
        BootstrapError: If the selector is present but blank, or names a
            bootstrap that cannot be loaded.

    """
    container = Container()
    # Core port bindings. A bootstrap may rebind any of these below; unset,
    # these stand and the gateway behaves exactly as it does with no overlay.
    #
    # Billing has no core implementation, so the default is the Null Object:
    # this deployment runs billing-free, holding and charging nothing.
    container.bind(BillingPort, _billing_adapter)
    # Entitlement: the base grants the capability set it ships, which is
    # currently empty, and reports every overlay-only capability as absent.
    container.bind(EntitlementPort, _entitlement_adapter)
    # Model inference: the base has no hosted-inference fleet, so every
    # candidate with no BYO credential is unavailable. Self-hosting is served
    # upstream of this port, not behind it.
    container.bind(ModelProviderPort, _model_provider_adapter)
    # Growth and support-messenger notifications: the base has no vendor of its
    # own, so every lifecycle event is a no-op.
    container.bind(GrowthSignalPort, _growth_signal_adapter)

    if bootstrap_selector is None:
        # No selector is a legitimate deployment (the plain open-source one), so
        # it is recorded rather than refused. Worth stating even so: which build
        # a process is running is otherwise invisible until traffic exposes it.
        container.summary = f"no bootstrap, core defaults for {_port_names(container)}"
        logger.info("Composition root: %s", container.summary)
        return container
    if not bootstrap_selector.strip():
        # A blank-but-present selector is a broken deployment rather than a
        # request for the plain build: something set OTARI_BOOTSTRAP and lost
        # its value. Refusing to boot beats silently running a build nobody
        # chose. (An empty string never reaches here: the config layer reads
        # OTARI_BOOTSTRAP="" as unset, like every other scalar. Whitespace does.)
        msg = "OTARI_BOOTSTRAP is set but blank; unset it to run without a bootstrap"
        raise BootstrapError(msg)

    defaults = dict(container.bindings())
    outcome = _load_register(bootstrap_selector)(container)
    if inspect.isawaitable(outcome):
        # The same silent drop the ``iscoroutinefunction`` guard refuses, by the
        # route that guard cannot see: a callable *object* whose ``__call__`` is
        # ``async def`` is not a coroutine function, so it passes every check
        # above and its body never runs until awaited. Closing the coroutine
        # keeps the refusal from also emitting "was never awaited" at whatever
        # point the garbage collector gets to it.
        outcome.close()
        msg = (
            f"Bootstrap {bootstrap_selector!r} returned an awaitable; the container is built "
            "synchronously, so register must run to completion when called"
        )
        raise BootstrapError(msg)
    rebound = sorted(_port_name(port) for port, factory in container.bindings() if defaults.get(port) is not factory)
    container.summary = f"{bootstrap_selector} rebound {', '.join(rebound) or 'no ports'}"
    contributed = ", ".join(contribution.capability for contribution in container.router_contributions())
    if contributed:
        container.summary += f", contributed routers for {contributed}"
    logger.info("Composition root: %s", container.summary)
    return container


def _port_name(port: PortKey[Any]) -> str:
    """Return a port's name for a log line."""
    name: str = getattr(port, "__name__", repr(port))
    return name


def _port_names(container: Container) -> str:
    """Return the bound ports' names, for the startup log."""
    return ", ".join(sorted(_port_name(port) for port, _ in container.bindings()))
