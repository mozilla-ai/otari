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
from typing import Any, TypeVar, cast, get_protocol_members

from fastapi import APIRouter
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.adapters.api_key_format_adapter import DefaultApiKeyFormatAdapter
from gateway.adapters.billing_adapter import NullBillingAdapter
from gateway.adapters.code_execution_adapter import build_code_execution_port, verify_code_execution_ready
from gateway.adapters.code_execution_policy_adapter import LocalCodeExecutionPolicy, RemoteCodeExecutionPolicy
from gateway.adapters.entitlement_adapter import BaseEntitlementAdapter
from gateway.adapters.file_storage_adapter import build_file_storage_port
from gateway.adapters.growth_signal_adapter import NullGrowthSignalAdapter
from gateway.adapters.identity_provider_adapter import DeploymentIdentityProviderAdapter
from gateway.adapters.mcp_server_adapter import LocalMcpServers, RemoteMcpServers
from gateway.adapters.model_provider_adapter import SelfHostedModelProviderAdapter
from gateway.adapters.provider_file_adapter import AnyLlmProviderFiles
from gateway.adapters.rate_limit_store_adapter import build_rate_limit_store
from gateway.adapters.telemetry_storage_adapter import DatabaseTelemetryStorageAdapter
from gateway.adapters.web_search_policy_adapter import LocalWebSearchPolicy, RemoteWebSearchPolicy
from gateway.core.config import GatewayConfig
from gateway.core.deployment import Plane, deployment_for
from gateway.core.unit_of_work import UnitOfWork
from gateway.log_config import logger
from gateway.ports.api_key_format_port import ApiKeyFormatPort
from gateway.ports.billing_port import BillingPort
from gateway.ports.code_execution_policy_port import CodeExecutionPolicyPort
from gateway.ports.code_execution_port import CodeExecutionPort
from gateway.ports.entitlement_port import EntitlementPort
from gateway.ports.file_storage_port import FileStoragePort
from gateway.ports.growth_signal_port import GrowthSignalPort
from gateway.ports.identity_provider_port import IdentityProviderPort
from gateway.ports.mcp_server_port import McpServerPort
from gateway.ports.model_provider_port import ModelProviderPort
from gateway.ports.provider_file_port import ProviderFilePort
from gateway.ports.rate_limit_store_port import RateLimitStorePort
from gateway.ports.telemetry_storage_port import TelemetryStoragePort
from gateway.ports.web_search_policy_port import WebSearchPolicyPort
from gateway.repositories.tenancy import UserRepository
from gateway.services.tenancy.membership_listener import MembershipListener
from gateway.services.tenancy.organization_service import OrganizationService
from gateway.services.tenancy.workspace_listener import WorkspaceListener
from gateway.services.tools import WorkspaceSearchKeys

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
# A port whose adapter writes workspace membership also needs the request's Unit
# of Work, because the membership listener writes through its open block and only
# ``get_unit_of_work`` may construct one.
UnitOfWorkPortFactory = Callable[[AsyncSession | None, UnitOfWork | None], T]
MembershipListenerBuilder = Callable[[UnitOfWork], MembershipListener]
WorkspaceListenerBuilder = Callable[[UnitOfWork], WorkspaceListener]
WorkspaceSearchKeysBuilder = Callable[[AsyncSession], WorkspaceSearchKeys]
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


class PortShapeError(ContainerError):
    """Raised when a bootstrap binds an adapter that lacks a method its port declares."""


class Container:
    """A registry mapping each port to the adapter that satisfies it.

    Built once per app with the core bindings plus whatever the configured
    bootstrap rebinds, then read per request. Attached to ``app.state`` rather
    than kept module-global, so two apps in one process (as the test suite
    builds) never share one.
    """

    def __init__(self) -> None:
        self._factories: dict[Any, PortFactory[Any] | UnitOfWorkPortFactory[Any]] = {}
        self._unit_of_work_ports: set[Any] = set()
        self._router_contributions: list[RouterContribution] = []
        # One line naming what this container was built from, logged by
        # build_container and asserted on by tests.
        self.summary = "unbuilt"

    def bind(self, port: PortKey[T], factory: PortFactory[T]) -> None:
        """Bind ``port`` to ``factory``, so a bootstrap can replace a core default.

        A later bind for the same port replaces an earlier one.
        """
        self._factories[port] = factory
        self._unit_of_work_ports.discard(port)

    def bind_with_unit_of_work(self, port: PortKey[T], factory: UnitOfWorkPortFactory[T]) -> None:
        """Bind ``port`` to a factory that also receives the request's Unit of Work."""
        self._factories[port] = factory
        self._unit_of_work_ports.add(port)

    def bindings(self) -> ItemsView[Any, PortFactory[Any] | UnitOfWorkPortFactory[Any]]:
        """Return a snapshot of the (port, factory) pairs bound so far.

        A snapshot rather than a live view, so iterating it stays safe while a
        bootstrap binds. For the composition root itself, which compares a
        bootstrap's bindings against the defaults to report what it rebound.
        Not a resolution path; callers use :meth:`resolve`.
        """
        return dict(self._factories).items()

    def resolve(self, port: PortKey[T], session: AsyncSession | None, *, uow: UnitOfWork | None = None) -> T:
        """Return the adapter bound to ``port``, built for this request's session.

        ``uow`` reaches only a factory bound with :meth:`bind_with_unit_of_work`.

        Raises:
            PortNotBoundError: If no adapter has been bound to ``port``.

        """
        factory = self._factories.get(port)
        if factory is None:
            raise PortNotBoundError(port)
        if port in self._unit_of_work_ports:
            return cast(UnitOfWorkPortFactory[T], factory)(session, uow)
        return cast(PortFactory[T], factory)(session)

    def contribute_router(self, contribution: RouterContribution) -> None:
        """Record a router this build mounts on top of Otari's own."""
        self._router_contributions.append(contribution)

    def router_contributions(self) -> tuple[RouterContribution, ...]:
        """Return the recorded router contributions, in contribution order."""
        return tuple(self._router_contributions)


def _verify_port_shape(container: Container, port: type[Any]) -> None:
    """Refuse to boot on an adapter that lacks a method ``port`` declares.

    A port is a plain ``Protocol`` and a bind checks nothing, so an overlay
    written against an older shape of the port binds cleanly and fails on the
    first request that reaches the missing method, as a 500 with no startup
    signal. Checked for the ports whose adapters build with no session, which
    the hybrid data plane already requires of ``ModelProviderPort``.

    Raises:
        PortShapeError: the bound adapter lacks one of the port's methods.

    """
    adapter = container.resolve(port, None)
    missing = sorted(name for name in get_protocol_members(port) if not hasattr(adapter, name))
    if missing:
        msg = f"{type(adapter).__name__}, bound to {_port_name(port)}, lacks {', '.join(missing)}"
        raise PortShapeError(msg)


def _api_key_format_adapter(session: AsyncSession | None) -> ApiKeyFormatPort:
    """Build the core ``ApiKeyFormatPort`` adapter for one request."""
    return DefaultApiKeyFormatAdapter(session)


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


def _identity_provider_adapter_factory(
    config: GatewayConfig | None,
    membership_listener: MembershipListenerBuilder | None,
    workspace_listener: WorkspaceListenerBuilder | None = None,
) -> UnitOfWorkPortFactory[IdentityProviderPort]:
    """Build the core ``IdentityProviderPort`` factory, bound to this app's ``open_signup`` setting.

    An open signup creates a workspace membership, so the adapter needs the request's Unit of Work too.
    """

    def build(session: AsyncSession | None, uow: UnitOfWork | None) -> IdentityProviderPort:
        if session is None or uow is None:
            msg = f"a session and a unit of work are required to build {_port_name(IdentityProviderPort)}"
            raise ContainerError(msg)
        if membership_listener is None:
            msg = f"a membership listener is required to build {_port_name(IdentityProviderPort)}"
            raise ContainerError(msg)
        return DeploymentIdentityProviderAdapter(
            UserRepository(session),
            OrganizationService(
                session,
                membership_listener=membership_listener(uow),
                uow=uow,
                workspace_listener=workspace_listener(uow) if workspace_listener is not None else None,
            ),
            open_signup=bool(config and config.open_signup),
        )

    return build


def _provider_file_adapter(session: AsyncSession | None) -> ProviderFilePort:
    """Build the core ``ProviderFilePort`` adapter, which holds no state of its own."""
    return AnyLlmProviderFiles()


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


def _code_execution_adapter_factory(config: GatewayConfig | None) -> PortFactory[CodeExecutionPort]:
    """The core ``CodeExecutionPort`` factory, closed over this app's config.

    Config rather than a session, because which adapter runs the code is a
    deployment setting (``sandbox_provider``) and not a per-request fact. A
    container built without config (the test helper's default) resolves this
    port only to raise, which is louder than quietly picking a backend.
    """

    def factory(session: AsyncSession | None) -> CodeExecutionPort:
        del session
        if config is None:
            msg = "CodeExecutionPort needs the deployment config; build the container with it"
            raise ContainerError(msg)
        return build_code_execution_port(config)

    return factory


def _file_storage_port_factory(config: GatewayConfig | None) -> PortFactory[FileStoragePort]:
    """The core ``FileStoragePort`` factory, closed over this app's config.

    Config rather than a session, because which store holds the bytes is a
    deployment setting (``files_backend``) and not a per-request fact.
    The store is built on first resolve and reused, so the retention sweep
    reclaims bytes through the same store the request path wrote them with, and
    a process that never resolves this port opens no client at all.
    A container built without config resolves this port only to raise, which is
    louder than quietly writing to a directory nobody chose.
    """
    store: FileStoragePort | None = None

    def factory(session: AsyncSession | None) -> FileStoragePort:
        del session
        nonlocal store
        if config is None:
            msg = "FileStoragePort needs the deployment config; build the container with it"
            raise ContainerError(msg)
        if store is None:
            store = build_file_storage_port(config)
        return store

    return factory


def _rate_limit_store_port_factory(config: GatewayConfig | None) -> PortFactory[RateLimitStorePort]:
    """The core ``RateLimitStorePort`` factory, closed over this app's config.

    Built on first resolve and reused, so every request counts in one store and
    a process holds one connection pool to it.
    A container built without config resolves this port only to raise.
    """
    store: RateLimitStorePort | None = None

    def factory(session: AsyncSession | None) -> RateLimitStorePort:
        del session
        nonlocal store
        if config is None:
            msg = "RateLimitStorePort needs the deployment config; build the container with it"
            raise ContainerError(msg)
        if store is None:
            store = build_rate_limit_store(config)
        return store

    return factory


def _requires_config(port: PortKey[T]) -> PortFactory[T]:
    """A factory that refuses every resolve, for a container built without config."""

    def factory(session: AsyncSession | None) -> T:
        del session
        msg = f"{_port_name(port)} needs the deployment config; build the container with it"
        raise ContainerError(msg)

    return factory


def _shared(adapter: T) -> PortFactory[T]:
    """A factory that serves the one ``adapter`` to every request.

    NOTE: The adapter must hold no per-request state, because concurrent requests share it.
    """
    return lambda session: adapter


def _with_session(port: PortKey[T], adapter: Callable[[AsyncSession], T]) -> PortFactory[T]:
    """A factory that builds ``adapter`` over each request's own session, which it requires."""

    def factory(session: AsyncSession | None) -> T:
        if session is None:
            msg = f"a session is required where this deployment holds the rows behind {_port_name(port)}"
            raise ContainerError(msg)
        return adapter(session)

    return factory


def _local_web_search_policy(
    search_keys: WorkspaceSearchKeysBuilder | None,
) -> Callable[[AsyncSession], WebSearchPolicyPort]:
    """Build the stored web search policy, which also reads the workspace's own search key."""

    def build(session: AsyncSession) -> WebSearchPolicyPort:
        if search_keys is None:
            msg = f"a search key resolver is required to build {_port_name(WebSearchPolicyPort)}"
            raise ContainerError(msg)
        return LocalWebSearchPolicy(session, search_keys=search_keys(session))

    return build


def _bind_workspace_ports(
    container: Container, config: GatewayConfig | None, search_keys: WorkspaceSearchKeysBuilder | None
) -> None:
    """Bind each workspace port to this deployment's own rows, or to its peer where a peer holds them.

    The planes a deployment serves are fixed for the life of the process, so they are read once, here.
    """
    if config is None:
        container.bind(CodeExecutionPolicyPort, _requires_config(CodeExecutionPolicyPort))
        container.bind(McpServerPort, _requires_config(McpServerPort))
        container.bind(WebSearchPolicyPort, _requires_config(WebSearchPolicyPort))
    elif deployment_for(config).supports(Plane.CONTROL):
        container.bind(CodeExecutionPolicyPort, _with_session(CodeExecutionPolicyPort, LocalCodeExecutionPolicy))
        container.bind(McpServerPort, _with_session(McpServerPort, LocalMcpServers))
        container.bind(WebSearchPolicyPort, _with_session(WebSearchPolicyPort, _local_web_search_policy(search_keys)))
    else:
        container.bind(CodeExecutionPolicyPort, _shared(RemoteCodeExecutionPolicy(config)))
        container.bind(McpServerPort, _shared(RemoteMcpServers(config)))
        container.bind(WebSearchPolicyPort, _shared(RemoteWebSearchPolicy(config)))


def build_container(
    bootstrap_selector: str | None = None,
    config: GatewayConfig | None = None,
    *,
    membership_listener: MembershipListenerBuilder | None = None,
    workspace_listener: WorkspaceListenerBuilder | None = None,
    search_keys: WorkspaceSearchKeysBuilder | None = None,
) -> Container:
    """Build the composition-root container for this deployment.

    Binds the core adapters, then, if a selector is given, lets the bootstrap it
    names rebind ports and contribute routers. With no selector the core
    defaults stand and Otari boots standalone.

    ``membership_listener`` builds the listener the OAuth sign-in adapter's
    organization service notifies. It is a parameter because its builder lives
    in the API layer, which this module cannot import. ``workspace_listener``
    builds what sets up a workspace that adapter's open signup creates, and
    ``search_keys`` the resolver of a workspace's own web search key, for the
    same reason.

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
    # OAuth sign-in: the base applies this deployment's `open_signup` setting.
    container.bind_with_unit_of_work(
        IdentityProviderPort, _identity_provider_adapter_factory(config, membership_listener, workspace_listener)
    )
    # API key format: the base mints the open-source shape and checks every
    # presented key against its own rows. A hosted overlay binds a format that
    # carries a region and a checksum, and routes a key minted elsewhere away.
    container.bind(ApiKeyFormatPort, _api_key_format_adapter)
    # Code execution: the base speaks the published protocol to the backend at
    # ``sandbox_url``, and runs E2B's hosted sandboxes in this process when the
    # deployment asks for them instead. An overlay with its own platform binds
    # a third adapter here and changes nothing above the port.
    container.bind(CodeExecutionPort, _code_execution_adapter_factory(config))
    # Uploaded file bytes: the base writes them to a local directory, an S3
    # bucket or any fsspec filesystem, whichever ``files_backend`` names. An
    # overlay binds a store of its own and changes nothing above the port.
    container.bind(FileStoragePort, _file_storage_port_factory(config))
    # A provider's own files: the base reaches them through any-llm with the
    # credential a request dispatches with. An overlay that must not hand a
    # managed credential to this process binds a transfer of its own.
    container.bind(ProviderFilePort, _provider_file_adapter)
    # A workspace's MCP servers and its web search and code execution policies.
    # An overlay binds a source of its own and changes nothing above the port.
    _bind_workspace_ports(container, config, search_keys)
    # Rate-limit counts: the base keeps them in this process, or in Redis
    # where ``rate_limit_store`` asks for one count shared by every replica.
    container.bind(RateLimitStorePort, _rate_limit_store_port_factory(config))
    if config is not None:
        # Asked once, at build, rather than per request: selecting a hosted
        # provider is itself what publishes code execution on ``/v1/tools``, in
        # the playground menu and to the pricing warning, so a missing extra or
        # credential would otherwise be found by a caller, as a 502 on work it
        # was told would run. A deployment that named a sandbox it cannot lease
        # fails to start, the way one that named a bootstrap it cannot load does.
        verify_code_execution_ready(config)

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
    _verify_port_shape(container, ModelProviderPort)
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
