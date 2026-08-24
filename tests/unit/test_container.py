"""Composition-root container and bootstrap hook.

Covers the mechanism itself: the core defaults every deployment gets, what a
bootstrap may change about them, and how a broken selector is reported. The
end-to-end path (an overlay module rebinding a port and adding a route to a
running app) is in ``tests/integration/test_bootstrap_overlay.py``.
"""

import sys
from collections.abc import Generator
from pathlib import Path
from typing import Protocol, cast

import pytest
from fastapi import APIRouter
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.adapters.billing_adapter import NullBillingAdapter
from gateway.adapters.entitlement_adapter import BaseEntitlementAdapter
from gateway.adapters.growth_signal_adapter import NullGrowthSignalAdapter
from gateway.adapters.identity_provider_adapter import RosterIdentityProviderAdapter
from gateway.adapters.model_provider_adapter import SelfHostedModelProviderAdapter
from gateway.adapters.telemetry_storage_adapter import DatabaseTelemetryStorageAdapter
from gateway.container import (
    BootstrapError,
    Container,
    PortNotBoundError,
    RouterContribution,
    build_container,
)
from gateway.ports.billing_port import BillingPort
from gateway.ports.entitlement_port import EntitlementPort
from gateway.ports.growth_signal_port import GrowthSignalPort
from gateway.ports.identity_provider_port import IdentityProviderPort
from gateway.ports.model_provider_port import ModelProviderPort
from gateway.ports.telemetry_storage_port import TelemetryStoragePort

# The core adapters ignore the session, so a placeholder stands in for one; a
# unit test of the wiring has no database and needs none.
NO_SESSION = cast(AsyncSession, None)


class UnusedPort(Protocol):
    """A port nothing binds, for the not-bound path."""

    async def do_nothing(self) -> None: ...


class _ReboundBilling(NullBillingAdapter):
    """Stands in for an overlay's own billing adapter."""


# Every module name _write_bootstrap has handed out this test, so the autouse
# fixture below can take each back out of sys.modules afterwards.
_WRITTEN: set[str] = set()


@pytest.fixture(autouse=True)
def _forget_written_bootstraps() -> Generator[None]:
    """Leave ``sys.modules`` as the test found it.

    A written bootstrap is imported from ``tmp_path``, which is gone by the next
    test, so the module object must not outlive the test that wrote it.
    ``monkeypatch.delitem(..., raising=False)`` does not manage this on its own:
    on a name that is absent it records no undo entry (see ``MonkeyPatch``), so
    teardown restores whichever module the *first* such test left behind instead
    of clearing it, and that stale module is what survives the suite.
    """
    _WRITTEN.clear()
    yield
    for name in _WRITTEN:
        sys.modules.pop(name, None)
    _WRITTEN.clear()


def _write_bootstrap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, body: str) -> None:
    """Write an importable bootstrap module and put it on ``sys.path``."""
    (tmp_path / f"{name}.py").write_text(body)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(name, None)
    _WRITTEN.add(name)


def test_core_defaults_are_bound_for_every_port() -> None:
    container = build_container()

    assert isinstance(container.resolve(BillingPort, NO_SESSION), NullBillingAdapter)
    assert isinstance(container.resolve(EntitlementPort, NO_SESSION), BaseEntitlementAdapter)
    assert isinstance(container.resolve(ModelProviderPort, NO_SESSION), SelfHostedModelProviderAdapter)
    assert isinstance(container.resolve(GrowthSignalPort, NO_SESSION), NullGrowthSignalAdapter)
    assert isinstance(container.resolve(TelemetryStoragePort, NO_SESSION), DatabaseTelemetryStorageAdapter)
    assert isinstance(container.resolve(IdentityProviderPort, NO_SESSION), RosterIdentityProviderAdapter)


def test_no_selector_contributes_no_routers_and_says_so() -> None:
    container = build_container()

    assert container.router_contributions() == ()
    assert container.summary.startswith("no bootstrap, core defaults for ")
    for port in (
        BillingPort,
        EntitlementPort,
        GrowthSignalPort,
        IdentityProviderPort,
        ModelProviderPort,
        TelemetryStoragePort,
    ):
        assert port.__name__ in container.summary


def test_resolve_refuses_a_port_nothing_bound() -> None:
    container = build_container()

    with pytest.raises(PortNotBoundError, match="UnusedPort"):
        container.resolve(UnusedPort, NO_SESSION)


def test_a_later_bind_replaces_an_earlier_one() -> None:
    container = build_container()
    replacement = _ReboundBilling

    container.bind(BillingPort, replacement)

    assert dict(container.bindings())[BillingPort] is replacement
    assert isinstance(container.resolve(BillingPort, NO_SESSION), _ReboundBilling)


def test_bindings_is_a_snapshot_not_a_live_view() -> None:
    container = build_container()
    before = dict(container.bindings())

    container.bind(BillingPort, _ReboundBilling)

    assert before[BillingPort] is not dict(container.bindings())[BillingPort]


def test_bootstrap_rebinds_a_port_and_contributes_a_router(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_bootstrap(
        tmp_path,
        monkeypatch,
        "probe_bootstrap",
        """
from fastapi import APIRouter

from gateway.container import Container, RouterContribution
from gateway.ports.entitlement_port import EntitlementPort

router = APIRouter()


class ProbeEntitlements:
    def __init__(self, session):
        self.session = session

    async def entitlements(self):
        return {"probe"}


def register(container: Container) -> None:
    container.bind(EntitlementPort, ProbeEntitlements)
    container.contribute_router(RouterContribution(capability="probe", router=router))
""",
    )

    container = build_container("probe_bootstrap:register")

    entitlements = container.resolve(EntitlementPort, NO_SESSION)
    assert type(entitlements).__name__ == "ProbeEntitlements"
    assert not isinstance(entitlements, BaseEntitlementAdapter)
    assert [contribution.capability for contribution in container.router_contributions()] == ["probe"]
    assert container.summary == ("probe_bootstrap:register rebound EntitlementPort, contributed routers for probe")


def test_bootstrap_that_rebinds_nothing_leaves_the_core_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_bootstrap(
        tmp_path,
        monkeypatch,
        "inert_bootstrap",
        "def register(container):\n    return None\n",
    )

    container = build_container("inert_bootstrap:register")

    assert isinstance(container.resolve(BillingPort, NO_SESSION), NullBillingAdapter)
    assert container.summary == "inert_bootstrap:register rebound no ports"


def test_a_selector_with_surrounding_whitespace_still_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_bootstrap(
        tmp_path,
        monkeypatch,
        "spaced_bootstrap",
        "def register(container):\n    return None\n",
    )

    container = build_container("  spaced_bootstrap:register  ")

    assert "rebound no ports" in container.summary


@pytest.mark.parametrize(
    ("selector", "message"),
    [
        ("no_colon_here", "must be 'module:callable'"),
        (":register", "must be 'module:callable'"),
        ("module_only:", "must be 'module:callable'"),
        ("   ", "set but blank"),
    ],
)
def test_a_malformed_selector_refuses_to_boot(selector: str, message: str) -> None:
    with pytest.raises(BootstrapError, match=message):
        build_container(selector)


def test_a_missing_bootstrap_module_is_reported_as_not_found() -> None:
    with pytest.raises(BootstrapError, match="was not found"):
        build_container("gateway_bootstrap_that_does_not_exist:register")


def test_a_bootstrap_whose_own_import_fails_is_reported_as_such(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A module that exists but reaches for something that does not: distinct
    # from a wrong selector, and the message has to say which it was, or an
    # operator debugs the wrong half.
    _write_bootstrap(
        tmp_path,
        monkeypatch,
        "broken_bootstrap",
        "import a_dependency_that_is_not_installed\n\n\ndef register(container):\n    return None\n",
    )

    with pytest.raises(BootstrapError, match="failed to import"):
        build_container("broken_bootstrap:register")


def test_a_bootstrap_missing_its_callable_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_bootstrap(tmp_path, monkeypatch, "empty_bootstrap", "value = 1\n")

    with pytest.raises(BootstrapError, match="has no attribute 'register'"):
        build_container("empty_bootstrap:register")


def test_a_bootstrap_attribute_that_is_not_callable_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_bootstrap(tmp_path, monkeypatch, "value_bootstrap", "register = 1\n")

    with pytest.raises(BootstrapError, match="is not callable"):
        build_container("value_bootstrap:register")


def test_an_async_bootstrap_is_refused_rather_than_silently_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An ``async def register`` is callable, so without an explicit check it
    # passes every guard and the container calls it, gets a coroutine nobody
    # awaits, and boots the plain build with the selector's bindings silently
    # discarded. That is the failure this path exists to refuse, and every port
    # method being async makes it the easy mistake to make.
    _write_bootstrap(
        tmp_path,
        monkeypatch,
        "async_bootstrap",
        "from gateway.ports.billing_port import BillingPort\n\n\n"
        "async def register(container):\n    container.bind(BillingPort, _ReboundBilling)\n",
    )

    with pytest.raises(BootstrapError, match="is async"):
        build_container("async_bootstrap:register")


def test_an_async_callable_object_bootstrap_is_refused_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The hole the iscoroutinefunction guard alone leaves: an instance whose
    # __call__ is async is callable and is *not* a coroutine function, so it
    # passes every guard, gets called, and returns a coroutine whose body never
    # ran. Same silent drop, different route in, so it is refused at the result
    # instead of at the callable.
    _write_bootstrap(
        tmp_path,
        monkeypatch,
        "async_callable_bootstrap",
        "class Register:\n"
        "    async def __call__(self, container):\n"
        "        raise AssertionError('the body must never run')\n"
        "\n\n"
        "register = Register()\n",
    )

    with pytest.raises(BootstrapError, match="returned an awaitable"):
        build_container("async_callable_bootstrap:register")


def test_router_contributions_keep_their_order() -> None:
    container = Container()
    first = RouterContribution(capability="one", router=APIRouter())
    second = RouterContribution(capability="two", router=APIRouter())

    container.contribute_router(first)
    container.contribute_router(second)

    assert container.router_contributions() == (first, second)
