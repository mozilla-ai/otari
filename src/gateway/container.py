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
this container after the core defaults are bound, and may rebind any port,
contribute routers and lifespan background tasks of its own, and contribute an
Alembic migration chain for tables of its own. Unset, the defaults stand.
"""

import importlib
import inspect
from collections.abc import Callable, Coroutine, ItemsView
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

from fastapi import APIRouter
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.adapters.billing_adapter import NullBillingAdapter
from gateway.adapters.entitlement_adapter import BaseEntitlementAdapter
from gateway.adapters.growth_signal_adapter import NullGrowthSignalAdapter
from gateway.adapters.identity_provider_adapter import RosterIdentityProviderAdapter
from gateway.adapters.model_provider_adapter import SelfHostedModelProviderAdapter
from gateway.adapters.telemetry_storage_adapter import DatabaseTelemetryStorageAdapter
from gateway.log_config import logger
from gateway.ports.billing_port import BillingPort
from gateway.ports.entitlement_port import EntitlementPort
from gateway.ports.growth_signal_port import GrowthSignalPort
from gateway.ports.identity_provider_port import IdentityProviderPort
from gateway.ports.model_provider_port import ModelProviderPort
from gateway.ports.telemetry_storage_port import TelemetryStoragePort

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig

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

    The additive half of the seam: nothing is swapped here, a surface is added.

    ``capability`` names the licensing axis the surface sits on. Given a name,
    every route the router exposes is served only when the deployment is
    entitled to it, resolved through ``EntitlementPort``, because hiding a link
    in a dashboard is not authorization and a route mounted into this process
    has to refuse for itself. Use it for a surface an overlay licenses per
    deployment.

    ``None`` mounts the router with no entitlement dependency. That is the
    right answer for a contribution that is simply present when the module is
    installed, which is what a plugin is: there is no licensing decision to
    make, and inventing a capability name solely to satisfy the gate would
    invent one.

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

    capability: str | None
    router: APIRouter


@dataclass(frozen=True)
class BackgroundTaskContribution:
    """One periodic worker an overlay runs for the life of the process.

    The lifespan (``gateway.main``) schedules ``start(config)`` as a task after
    Otari's own refreshers, in every mode, and stops it under the same shared
    cancellation bound, so a contributed task that ignores cancellation cannot
    hang shutdown. ``name`` labels the task in the startup summary and in the
    shutdown log, and is unique per container.

    ``start`` is a coroutine function, not the coroutine itself: a coroutine
    object built at register time in a container that never runs (as the test
    suite builds) would only leak a "never awaited" warning.
    """

    name: str
    start: Callable[["GatewayConfig"], Coroutine[Any, Any, None]]


# The version table Otari's own chain stamps (``alembic/env.py`` leaves
# Alembic's default in place, see #921). A contribution may not claim it.
CORE_VERSION_TABLE = "alembic_version"


@dataclass(frozen=True)
class MigrationContribution:
    """One Alembic chain an overlay runs against Otari's database, after Otari's own.

    A module a bootstrap loads may own tables of its own. Its revisions cannot
    join Otari's chain without editing the repo, and a second ``upgrade`` on the
    default version table would fight Otari's over one row, so a contribution
    brings its own script directory and its own ``version_table``, and the two
    histories never interleave. ``init_db`` runs Otari's chain to ``head``
    first, then each contribution's, on the same database URL.

    The contract for the contributed ``env.py``: Otari passes the database URL
    both as ``sqlalchemy.url`` and as ``config.attributes["database_url"]``, and
    passes the declared table name as ``config.attributes["version_table"]``.
    Prefer the attribute for the URL: ``sqlalchemy.url`` is read back through
    configparser, whose interpolation treats a percent sign as a token, so a
    password containing one breaks it. The ``version_table`` attribute is
    offered, not required: a chain may read it, or may hardcode a constant of
    its own. What Otari requires is that the ``version_table`` declared on the
    contribution is the table the chain actually stamps, because Otari uses the
    declared value only to refuse a collision with core's ``alembic_version``
    and with another contribution. A contributed chain must not
    reference a core table by foreign key in a way that would block a core
    migration: the core chain runs first and knows nothing about contributed
    tables, so a core revision that drops or rebuilds a table the contribution
    points at fails on a constraint the core chain did not create.

    ``name`` identifies the chain in the startup log and in errors; it is
    unique among contributions, as is ``version_table``.
    """

    name: str
    script_location: str
    version_table: str


class ContainerError(Exception):
    """Base error for composition-root wiring failures."""


class MigrationContributionError(ContainerError):
    """Raised when a contributed migration chain would collide with another chain."""


class PortNotBoundError(ContainerError):
    """Raised when a port is resolved but no adapter has been bound to it."""

    def __init__(self, port: PortKey[Any]) -> None:
        name = getattr(port, "__name__", repr(port))
        super().__init__(f"No adapter is bound for port {name}")
        self.port = port


class BootstrapError(ContainerError):
    """Raised when the bootstrap module ``OTARI_BOOTSTRAP`` names cannot be loaded."""


class DuplicateBackgroundTaskError(ContainerError):
    """Raised when a second background task is contributed under a name already taken."""

    def __init__(self, name: str) -> None:
        super().__init__(f"A background task named {name!r} is already contributed")
        self.name = name


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
        self._background_task_contributions: dict[str, BackgroundTaskContribution] = {}
        self._migration_contributions: list[MigrationContribution] = []
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

    def contribute_background_task(self, contribution: BackgroundTaskContribution) -> None:
        """Record a background task the lifespan runs beside Otari's own refreshers.

        Raises:
            DuplicateBackgroundTaskError: If a task of the same name is already
                contributed. Two tasks sharing a name would be indistinguishable
                in the shutdown log, and a bootstrap registering the same worker
                twice is a mistake worth refusing at startup.

        """
        if contribution.name in self._background_task_contributions:
            raise DuplicateBackgroundTaskError(contribution.name)
        self._background_task_contributions[contribution.name] = contribution

    def background_task_contributions(self) -> tuple[BackgroundTaskContribution, ...]:
        """Return the recorded background task contributions, in contribution order."""
        return tuple(self._background_task_contributions.values())

    def contribute_migrations(self, contribution: MigrationContribution) -> None:
        """Record an Alembic chain ``init_db`` runs after Otari's own.

        Refused at contribution time rather than at first boot, so a colliding
        bootstrap fails while the container is being built and never reaches
        the database.

        Raises:
            MigrationContributionError: If a field is blank, if the contribution
                claims Otari's own version table, or if it claims a version
                table or name another contribution already holds.

        """
        for field in ("name", "script_location", "version_table"):
            if not getattr(contribution, field).strip():
                msg = f"Migration contribution {contribution.name!r} has a blank {field}"
                raise MigrationContributionError(msg)
        if contribution.version_table == CORE_VERSION_TABLE:
            msg = (
                f"Migration contribution {contribution.name!r} claims {CORE_VERSION_TABLE!r}, "
                "which is Otari's own version table; a contributed chain stamps a table of its own"
            )
            raise MigrationContributionError(msg)
        for recorded in self._migration_contributions:
            if recorded.version_table == contribution.version_table:
                msg = (
                    f"Migration contribution {contribution.name!r} claims version table "
                    f"{contribution.version_table!r}, already held by {recorded.name!r}"
                )
                raise MigrationContributionError(msg)
            if recorded.name == contribution.name:
                msg = f"Migration contribution {contribution.name!r} is already recorded"
                raise MigrationContributionError(msg)
        self._migration_contributions.append(contribution)

    def migration_contributions(self) -> tuple[MigrationContribution, ...]:
        """Return the recorded migration contributions, in contribution order."""
        return tuple(self._migration_contributions)


def _billing_adapter(session: AsyncSession | None) -> BillingPort:
    """Build the core ``BillingPort`` adapter for one request."""
    return NullBillingAdapter(session)


def _entitlement_adapter(session: AsyncSession | None) -> EntitlementPort:
    """Build the core ``EntitlementPort`` adapter for one request."""
    return BaseEntitlementAdapter(session)


def _model_provider_adapter(session: AsyncSession | None) -> ModelProviderPort:
    """Build the core ``ModelProviderPort`` adapter for one request."""
    return SelfHostedModelProviderAdapter(session)


def _telemetry_storage_adapter(session: AsyncSession | None) -> TelemetryStoragePort:
    """Build the core ``TelemetryStoragePort`` adapter for one request."""
    return DatabaseTelemetryStorageAdapter(session)


def _growth_signal_adapter(session: AsyncSession | None) -> GrowthSignalPort:
    """Build the core ``GrowthSignalPort`` adapter for one request."""
    return NullGrowthSignalAdapter(session)


def _identity_provider_adapter(session: AsyncSession | None) -> IdentityProviderPort:
    """Build the core ``IdentityProviderPort`` adapter for one request."""
    return RosterIdentityProviderAdapter(session)


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
    names rebind ports and contribute routers, background tasks and migration
    chains. With no selector the core defaults stand and Otari boots standalone.

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
    # Telemetry storage: the base keeps what the OTLP receiver captures in
    # this deployment's own database, which is where it has always gone. An
    # overlay binds a scale-out store behind the same port.
    container.bind(TelemetryStoragePort, _telemetry_storage_adapter)
    # OAuth sign-in: the base applies its roster policy, so a social identity
    # signs in as an account an operator already added and never creates one.
    # This one is a real implementation rather than a Null Object, because
    # refusing an unknown identity is itself the base's answer.
    container.bind(IdentityProviderPort, _identity_provider_adapter)

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
    contributed = ", ".join(contribution.capability or "ungated" for contribution in container.router_contributions())
    if contributed:
        container.summary += f", contributed routers for {contributed}"
    contributed_tasks = ", ".join(contribution.name for contribution in container.background_task_contributions())
    if contributed_tasks:
        container.summary += f", contributed background tasks {contributed_tasks}"
    chains = ", ".join(contribution.name for contribution in container.migration_contributions())
    if chains:
        container.summary += f", contributed migration chains {chains}"
    logger.info("Composition root: %s", container.summary)
    return container


def _port_name(port: PortKey[Any]) -> str:
    """Return a port's name for a log line."""
    name: str = getattr(port, "__name__", repr(port))
    return name


def _port_names(container: Container) -> str:
    """Return the bound ports' names, for the startup log."""
    return ", ".join(sorted(_port_name(port) for port, _ in container.bindings()))
