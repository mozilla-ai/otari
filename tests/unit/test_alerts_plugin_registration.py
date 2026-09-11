"""What ``otari_alerts:register`` puts on the container, and what Otari does with it.

The plugin's whole wiring is three contributions, and each one is only real on
the other side of an Otari seam: the router has to answer in a build that
entitles nothing, the chain has to reach head beside Otari's without touching
its version row, and the task has to be started and cancelled by the lifespan.
Those three are what this file covers.

No PostgreSQL and no Docker: the chains run on a throwaway SQLite file, which
is what ``tests/unit/test_contributed_migrations.py`` does for the fixture
chain.
"""

import asyncio
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text

import otari_alerts
from gateway.adapters.entitlement_adapter import BASE_CAPABILITIES
from gateway.api.deps import get_db, get_db_if_needed
from gateway.api.main import _register_contributed_routers
from gateway.container import Container, build_container
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.database import init_db, reset_db
from gateway.main import _create_lifespan

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORE_ALEMBIC = _REPO_ROOT / "alembic"


@pytest.fixture
def container() -> Container:
    """A base build with the plugin's bootstrap loaded, as ``OTARI_BOOTSTRAP`` would."""
    return build_container("otari_alerts:register")


# --------------------------------------------------------------------------
# What register contributes
# --------------------------------------------------------------------------


def test_register_contributes_a_router_a_task_and_a_chain(container: Container) -> None:
    (router_contribution,) = container.router_contributions()
    (task,) = container.background_task_contributions()
    (chain,) = container.migration_contributions()

    assert router_contribution.router is otari_alerts.routes.router
    assert task.name == "budget-alerts"
    assert chain.name == "alerts"
    assert chain.version_table == otari_alerts.VERSION_TABLE == "alerts_alembic_version"


def test_the_router_is_contributed_ungated(container: Container) -> None:
    """No capability, because installing the plugin is the whole decision.

    A capability names a licensing axis, and this surface sits on none: it is
    present exactly when the module is installed.
    """
    (contribution,) = container.router_contributions()
    assert contribution.capability is None


def test_register_binds_no_port(container: Container) -> None:
    """Alerting has one implementation, so it earns no port (ARCHITECTURE.md rule 7)."""
    assert container.summary.startswith("otari_alerts:register rebound no ports")


def test_the_script_location_resolves_from_the_installed_package(container: Container) -> None:
    """A path off ``__file__``, so it is right in a wheel as well as a checkout."""
    (chain,) = container.migration_contributions()
    location = Path(chain.script_location)
    assert location.is_absolute()
    assert (location / "env.py").is_file()
    assert list((location / "versions").glob("*.py"))


# --------------------------------------------------------------------------
# The contributed router, mounted the way Otari mounts it
# --------------------------------------------------------------------------


def test_the_contributed_router_answers_in_a_build_that_entitles_nothing(container: Container) -> None:
    """An ungated contribution reaches its own auth check rather than a 404.

    The base build grants no capability, so a contribution naming one would be
    refused here before the router was ever consulted, and the response would
    be the same 404 a path nothing serves gets. 401 is the proof it is mounted:
    the mount point adds no credential, so what answered is the router's own
    ``verify_master_key``.
    """
    assert not BASE_CAPABILITIES, "the base grants a capability, so this no longer isolates the gate"
    api = APIRouter(prefix=API_ROOT)
    _register_contributed_routers(api, container)
    app = FastAPI()
    app.state.container = container
    app.state.config = GatewayConfig(master_key="sk-test-master")
    app.include_router(api)
    # No database behind this build: what is under test is the mount, and the
    # header check refuses before the session it was handed is ever used.
    app.dependency_overrides[get_db_if_needed] = lambda: None
    app.dependency_overrides[get_db] = lambda: None

    response = TestClient(app).get(f"{API_ROOT}/organizations/me/alert-rules")

    assert response.status_code == 401, response.text


def test_every_contributed_route_sits_under_the_organization_me_prefix(container: Container) -> None:
    api = APIRouter(prefix=API_ROOT)
    _register_contributed_routers(api, container)

    paths = {route.path for route in api.routes}  # type: ignore[attr-defined]

    assert paths == {
        f"{API_ROOT}/organizations/me/alert-rules",
        f"{API_ROOT}/organizations/me/alert-rules/{{rule_id}}",
        f"{API_ROOT}/organizations/me/alert-rules/{{rule_id}}/test",
    }


# --------------------------------------------------------------------------
# The contributed migration chain, beside Otari's own
# --------------------------------------------------------------------------


def _head(script_location: Path) -> str:
    config = Config()
    config.set_main_option("script_location", str(script_location))
    head = ScriptDirectory.from_config(config).get_current_head()
    assert head is not None
    return head


def _version_rows(url: str, version_table: str) -> list[str]:
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            return [row[0] for row in connection.execute(text(f"SELECT version_num FROM {version_table}"))]
    finally:
        engine.dispose()


def test_the_chain_reaches_head_and_stamps_only_its_own_version_table(
    container: Container, tmp_path: Path
) -> None:
    """Two histories, two tables, one database.

    Core's row must hold core's head and nothing else: a contributed chain that
    stamped ``alembic_version`` would make Otari's next upgrade unresolvable.
    """
    url = f"sqlite:///{tmp_path / 'alerts.db'}"

    init_db(
        GatewayConfig(database_url=url, auto_migrate=True),
        migration_contributions=container.migration_contributions(),
    )
    reset_db()

    engine = create_engine(url)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()

    assert {"alert_rules", "alert_deliveries"} <= tables
    assert {"users", "api_keys"} <= tables
    assert _version_rows(url, "alembic_version") == [_head(_CORE_ALEMBIC)]
    assert _version_rows(url, otari_alerts.VERSION_TABLE) == [_head(otari_alerts.ALEMBIC_DIR)]


def test_a_core_only_boot_creates_no_alert_table(tmp_path: Path) -> None:
    """The plugin's tables are not on Otari's metadata, so core's chain cannot make them."""
    url = f"sqlite:///{tmp_path / 'core-only.db'}"

    init_db(GatewayConfig(database_url=url, auto_migrate=True))
    reset_db()

    engine = create_engine(url)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()

    assert "users" in tables
    assert "alert_rules" not in tables
    assert otari_alerts.VERSION_TABLE not in tables


def test_the_chain_is_idempotent(container: Container, tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'twice.db'}"
    config = GatewayConfig(database_url=url, auto_migrate=True)

    for _ in range(2):
        init_db(config, migration_contributions=container.migration_contributions())
        reset_db()

    assert _version_rows(url, otari_alerts.VERSION_TABLE) == [_head(otari_alerts.ALEMBIC_DIR)]


def test_the_chain_downgrades_back_to_base(container: Container, tmp_path: Path) -> None:
    """A real, reversible ``downgrade()``, checked on the engine that is fussiest."""
    url = f"sqlite:///{tmp_path / 'down.db'}"
    init_db(
        GatewayConfig(database_url=url, auto_migrate=True),
        migration_contributions=container.migration_contributions(),
    )
    reset_db()

    config = Config()
    config.set_main_option("script_location", str(otari_alerts.ALEMBIC_DIR))
    config.attributes["database_url"] = url
    config.attributes["version_table"] = otari_alerts.VERSION_TABLE
    config.attributes["configure_logger"] = False
    command.downgrade(config, "base")

    engine = create_engine(url)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()

    assert "alert_rules" not in tables
    assert "alert_deliveries" not in tables
    # Core's schema is untouched by the plugin's downgrade.
    assert "users" in tables


# --------------------------------------------------------------------------
# The contributed background task, started and stopped by the lifespan
# --------------------------------------------------------------------------


def _evaluator_tasks() -> set[asyncio.Task[object]]:
    """Every running task whose coroutine is the plugin's worker.

    The lifespan keeps no public handle on a contributed task, so the task is
    found rather than injected, which also keeps the real coroutine under test
    instead of a stand-in.
    """
    return {
        task
        for task in asyncio.all_tasks()
        if getattr(task.get_coro(), "__qualname__", "") == "run_budget_alert_evaluator"
    }


@pytest.mark.asyncio
async def test_the_evaluator_runs_as_a_contributed_task_and_is_cancelled_at_shutdown(
    container: Container,
    tmp_path: Path,
) -> None:
    """Otari's lifespan owns the worker's life, so the plugin schedules nothing itself.

    The real ``run_budget_alert_evaluator``, not a probe: what is under test is
    that the coroutine the plugin contributed is the one that gets started, and
    that it yields to cancellation rather than having to be abandoned under the
    shutdown bound.
    """
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'lifespan.db'}",
        master_key="sk-test-master",
        model_discovery=False,
        require_pricing=False,
    )
    app = FastAPI()
    app.state.config = config
    app.state.container = container

    assert not _evaluator_tasks()
    async with _create_lifespan()(app):
        await asyncio.sleep(0)  # let the task reach its first await
        (task,) = _evaluator_tasks()
        assert not task.done()

    assert task.cancelled()


@pytest.mark.asyncio
async def test_the_evaluator_returns_immediately_in_hybrid_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hybrid gateway reserves nothing locally, so there are no ceilings to read.

    The task is contributed in every mode, so the mode check is the worker's
    own, and returning beats failing one query a minute forever.
    """
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw-test-token")
    config = GatewayConfig(mode="hybrid", platform={"base_url": "http://platform.test"})

    async with asyncio.timeout(5):
        await otari_alerts.evaluator.run_budget_alert_evaluator(config)


@pytest.mark.asyncio
async def test_a_zero_interval_stops_the_evaluator_starting(monkeypatch: pytest.MonkeyPatch) -> None:
    """``OTARI_ALERT_EVALUATION_INTERVAL_SEC=0`` is the off switch, rules kept."""
    from otari_alerts import config as alert_config

    monkeypatch.setattr(alert_config.settings, "alert_evaluation_interval_sec", 0)

    async with asyncio.timeout(5):
        await otari_alerts.evaluator.run_budget_alert_evaluator(GatewayConfig(master_key="sk-test"))
