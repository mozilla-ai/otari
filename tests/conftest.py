import argparse
import asyncio
import contextlib
import json
import os
import re
import shutil
import sys
import zlib
from collections.abc import AsyncIterator, Callable, Generator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest import mock

import anyio
import dotenv
import pytest
from dotenv import find_dotenv
from dotenv import load_dotenv as read_env_file

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if "gateway" in sys.modules:
    del sys.modules["gateway"]

if TYPE_CHECKING:
    from gateway.services.pricing import ModelsDevPriceIndex

MODELS_DEV_MINI = ROOT / "tests" / "fixtures" / "models_dev_mini.json"


@dataclass(frozen=True)
class Shard:
    """One of ``count`` disjoint slices of a test run, numbered from 1.

    Every test belongs to exactly one shard, so running all ``count`` shards runs the whole suite once.
    """

    index: int
    count: int

    def includes(self, nodeid: str) -> bool:
        # Not hash(): it is salted per process, and every xdist worker must select the same tests.
        return zlib.crc32(nodeid.encode()) % self.count == self.index - 1


def parse_shard(value: str) -> Shard:
    """Parse ``INDEX/COUNT``, such as ``2/4``, into a shard."""
    match = re.fullmatch(r"([0-9]+)/([0-9]+)", value)
    if match is None or not 1 <= int(match[1]) <= int(match[2]):
        raise argparse.ArgumentTypeError(f"expected INDEX/COUNT with 1 <= INDEX <= COUNT, got {value!r}")
    return Shard(index=int(match[1]), count=int(match[2]))


# The only OTARI_* variables a test may see. An app starts with provider copies
# on and refuses to boot without a pepper of its own, so the suite supplies one.
_TEST_OWNED_ENV = {"OTARI_PROVIDER_ACCOUNT_PEPPER": "test-provider-account-pepper-0000"}

# This repository's own env file, which a developer keeps their settings in. The
# suite withholds this one path and reads any other, because an env file beside a
# named config file is a feature of ``load_config`` rather than a leak.
REPOSITORY_ENV_FILE = ROOT / ".env"


def is_hidden_env_name(name: str) -> bool:
    """Whether ``name`` is one of the variables the suite hides.

    NOTE: Matches whatever case the name is written in, because the settings
    class reads the environment case-insensitively and an exported
    ``otari_master_key`` is read as the master key.
    """
    return name.upper().startswith(("OTARI_", "OTEL_"))


def env_file_named(path: str | os.PathLike[str] | None) -> str:
    """The env file ``path`` names, searching upward from the working directory for ``None``.

    NOTE: Resolving ``None`` here rather than leaving it to dotenv is what makes
    the search start from the working directory. Dotenv starts its own from the
    file of whichever function called it, so a caller inside this repository finds
    this repository's env file whatever directory the caller is working in.
    """
    if path is not None:
        return str(path)
    return find_dotenv(usecwd=True)


def is_repository_env_file(path: str | os.PathLike[str] | None) -> bool:
    """Whether ``path`` names this repository's own env file.

    NOTE: Compares the canonical paths. An extra path segment and a symlinked
    checkout both name the same file by another name, and a lexical comparison
    reads each of them as some other file. A relative path resolves against the
    working directory, as it does everywhere else.
    """
    named = env_file_named(path)
    if not named:
        return False
    return Path(named).resolve(strict=False) == REPOSITORY_ENV_FILE.resolve(strict=False)


def _hide_the_machines_own_configuration() -> None:
    """Make a test's answer independent of who runs it.

    Three paths carry a developer's settings into a test.
    One is an exported ``OTARI_*`` or ``OTEL_*`` variable.
    The second is the env file the settings class names, which a bare
    ``GatewayConfig()`` reads even where nothing loads a config file.
    The third is ``load_config``, which reads an env file into the process
    environment, and of those only this repository's own is withheld.

    NOTE: This runs from ``pytest_configure`` rather than from a fixture, because
    a session-scoped fixture builds a config before any fixture of a test's own
    could have cleaned up after the machine.

    NOTE: It holds in this process alone. A test that spawns Otari as a
    subprocess from a directory holding an env file reads it, because the child
    gets neither patch. The scrubbed environment does reach a child.

    NOTE: It covers Otari's own prefixes alone. A provider credential a developer
    exports, such as ``GEMINI_API_KEY``, still reaches a test that reads one.

    NOTE: Both patches stand for the rest of the session. Nothing restores the
    settings class's env file, because no test should see it set.
    """
    for name in [name for name in os.environ if is_hidden_env_name(name)]:
        del os.environ[name]
    os.environ.update(_TEST_OWNED_ENV)

    # Imported after the scrub, so nothing it runs at import time reads what the
    # machine had set.
    from gateway.core import config as config_module

    config_module.GatewayConfig.model_config["env_file"] = None

    # The path positionally or by keyword, every other option by keyword. Taking
    # further positional arguments would let one of them land on ``override``,
    # which is the fourth parameter dotenv declares.
    def load_dotenv(dotenv_path: str | os.PathLike[str] | None = None, **options: Any) -> bool:
        named = env_file_named(dotenv_path)
        if not named or is_repository_env_file(named):
            return False
        # Never ``None``, so dotenv cannot search from this file's own directory.
        return bool(read_env_file(named, **options))

    # The name config resolves is the one dotenv exports, so the patch rebinds
    # that global rather than the function around it. Dotenv's own attribute is
    # rebound too, so a module importing the name later gets the guard as well.
    vars(config_module)["load_dotenv"] = load_dotenv
    vars(dotenv)["load_dotenv"] = load_dotenv


def pytest_configure(config: pytest.Config) -> None:
    _hide_the_machines_own_configuration()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--shard",
        type=parse_shard,
        default=None,
        metavar="INDEX/COUNT",
        help="Run only the tests in shard INDEX of COUNT.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    shard: Shard | None = config.getoption("shard")
    if shard is None:
        return
    selected = [item for item in items if shard.includes(item.nodeid)]
    config.hook.pytest_deselected(items=[item for item in items if not shard.includes(item.nodeid)])
    items[:] = selected


@pytest.fixture(autouse=True)
def _no_background_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop the app lifespan from dialing providers and models.dev for real.

    Every ``TestClient(app)`` runs the lifespan, which starts the discovery and
    catalog refreshers; their first act is to prime the cache from a live dial.
    That is right in production and wrong in a test: it makes real outbound calls
    from every app boot, and it races a test that patches the dial *after*
    startup, so the read then serves whatever the unpatched prime cached.

    Suppressing the refreshers leaves the cache empty, so a read takes the
    cold-provider path and dials once, under whatever the test has patched.
    A test that wants the warm-cache read path seeds the cache itself.

    Lives in the root conftest, not the integration one, because the unit suite
    builds apps too (``tests/unit/test_gateway_root_page.py``,
    ``test_tools_endpoint.py``, ``test_settings_endpoint.py`` and others all use
    ``TestClient(create_app(...))``). Scoping this to ``tests/integration`` left
    every one of those making live models.dev fetches on CI, which is both wrong
    on its own terms and what surfaced the unbounded-shutdown bug that
    ``_stop_refreshers`` now guards against. A test that wants the real refresher
    calls it directly rather than through the lifespan.
    """

    async def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr("gateway.main.run_discovery_refresher", _noop)
    monkeypatch.setattr("gateway.main.run_catalog_refresher", _noop)
    monkeypatch.setattr("gateway.api.routes.health_checks.run_health_check_worker", _noop)
    monkeypatch.setattr("gateway.api.routes.update_check.run_update_check_worker", _noop)


@pytest.fixture(autouse=True)
def _no_models_dev_fetch(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Answer every models.dev fetch as unreachable, and serve no bundled catalog, unless the test patches one in.

    The model listing joins models.dev, and with the refresher suppressed above
    its cache starts empty, so every listing read would otherwise fetch the live
    catalog: an outbound call from any test that lists a model, and limits that
    move whenever models.dev does. The bundled catalog moves at every release,
    so it is off too. A test that wants metadata patches
    ``model_catalog_service._fetch`` or ``_fallback`` itself. The cache is
    cleared on both sides, so one test's answer, a failure included, is never
    the next one's.
    """
    from gateway.services import model_catalog_service
    from gateway.services.pricing_refresh_service import reset_catalog_refresh_state

    async def _unreachable() -> None:
        return None

    reset_catalog_refresh_state()
    monkeypatch.setattr(model_catalog_service, "_fetch", _unreachable)
    monkeypatch.setattr(model_catalog_service, "_fallback", lambda: None)
    yield
    reset_catalog_refresh_state()


@pytest.fixture(autouse=True)
def _reset_default_workspace() -> None:
    """Drop the memoized key workspaces before each test.

    ``workspace_scope`` memoizes each key's workspace, because a usage row is
    written on the hot path and must not pay a lookup for something immutable.
    Immutable within one database, and every test empties its own, so an id
    cached by the previous test can name a row that is gone and the next insert
    then fails its foreign key. Same reason the alias and provider caches have
    resets.
    """
    from gateway.services.workspace_scope import reset_key_workspace_cache

    reset_key_workspace_cache()


@pytest.fixture(autouse=True)
def _reset_guardrail_verdicts() -> None:
    """Drop the guardrail verdicts a worker keeps before each test.

    A verdict is kept by the check and the text it answered, and tests ask the
    same check about the same text with a different fake verdict each, so one
    test's answer would otherwise be the next one's.
    """
    from gateway.services.guardrails import clear_verdict_cache

    clear_verdict_cache()


@pytest.fixture(autouse=True)
def _reset_default_pricing() -> Generator[None, None, None]:
    """Restore process-wide pricing state to its default before each test.

    ``configure_default_pricing`` is set at app startup, so a test that builds an
    app with a different ``default_pricing`` would otherwise leak that state into
    later tests that call ``find_model_pricing`` directly. Reset to off, matching
    the config field's opt-in default; tests that need defaults enable explicitly.

    Also drops the accepted price generations, so a snapshot accepted by one test
    cannot price another's models, and the cached stored rates, which name rows a
    test's emptied database no longer holds.
    """
    from gateway.services.pricing_refresh_service import reset_price_refresh_state
    from gateway.services.pricing_service import (
        configure_default_pricing,
        configure_provider_types,
        reset_pricing_cache,
    )

    configure_default_pricing(False)
    configure_provider_types(None)
    reset_price_refresh_state()
    reset_pricing_cache()
    yield
    configure_default_pricing(False)
    configure_provider_types(None)
    reset_price_refresh_state()
    reset_pricing_cache()


@pytest.fixture
def install_models_dev() -> Callable[[dict[str, Any] | None], "ModelsDevPriceIndex"]:
    """Serve a small models.dev catalog as the only accepted price generation.

    Takes a parsed ``api.json`` document, the shared mini fixture by default, and
    returns its index. The autouse reset above drops it after the test.
    """

    def install(catalog: dict[str, Any] | None = None) -> "ModelsDevPriceIndex":
        from gateway.services.pricing import ModelsDevPriceIndex, PriceGeneration, set_accepted_generations

        document = catalog if catalog is not None else json.loads(MODELS_DEV_MINI.read_text())
        index = ModelsDevPriceIndex.from_document(document)
        set_accepted_generations(
            [PriceGeneration(effective_at=datetime(2020, 1, 1, tzinfo=UTC), index=index)], complete=False
        )
        return index

    return install


@pytest.fixture(autouse=True, scope="session")
def _builtin_fetcher_registered() -> None:
    """Register otari's fetcher as any-fetch's ``builtin`` provider, once per worker.

    ``create_app`` registers it at startup. A test that runs ``builtin`` without an
    app, such as one that constructs the old web backend with its fetch half on,
    must not depend on an earlier test having built one, which a sharded or
    parallel run does not guarantee.
    """
    from gateway.services.tools import register_builtin_fetcher

    register_builtin_fetcher()


@pytest.fixture(scope="session")
def _migrated_sqlite_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A SQLite database migrated to head once per worker, for new databases to start from."""
    from gateway.core.database import _run_migrations

    template = tmp_path_factory.mktemp("migrated-template") / "gateway.db"
    _run_migrations(f"sqlite:///{template}")
    return template


@pytest.fixture(autouse=True)
def _start_new_sqlite_databases_migrated(monkeypatch: pytest.MonkeyPatch, _migrated_sqlite_template: Path) -> None:
    """Start an app's new SQLite database from a migrated copy instead of an empty file.

    Migrating an empty SQLite file takes about a third of a second, and most apps a test builds get a new one.
    Startup still runs the real migration step on the copy, and that step finds nothing left to apply.
    A file that already exists, and any database other than a SQLite file, migrates as it would without this fixture.
    """
    from gateway.core import database

    run_migrations = database._run_migrations

    def run_migrations_from_template(database_url: str) -> None:
        new_file = _new_sqlite_file(database_url)
        if new_file is not None:
            shutil.copyfile(_migrated_sqlite_template, new_file)
        run_migrations(database_url)

    monkeypatch.setattr(database, "_run_migrations", run_migrations_from_template)


@pytest.fixture(autouse=True)
def _no_relocated_claude_config_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide a ``CLAUDE_CONFIG_DIR`` the developer's own shell exports.

    NOTE: The variable moves Claude Code's settings and session history off the
    home directory, so a fake home does not hide them and a test asserting the
    default location would read the developer's real files. A test that wants
    the variable set does so itself.
    """
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)


@pytest.fixture
def isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An empty home directory, so neither a test nor a subprocess it runs reads the developer's own files."""
    home = tmp_path_factory.mktemp("home")
    # Windows reads USERPROFILE, not HOME.
    for name in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(home))
    return home


@pytest.fixture(autouse=True)
def _restore_environ() -> Generator[None]:
    """Undo what a test wrote to ``os.environ`` without going through monkeypatch.

    NOTE: Loading a config reads an env file into the process environment, and
    monkeypatch reverts only a name the test touched first, so a credential a test
    loaded would otherwise reach every test after it.
    """
    with mock.patch.dict(os.environ, clear=False):
        yield


@pytest.fixture
def test_owned_env_names() -> frozenset[str]:
    """The ``OTARI_*`` names the suite sets for itself."""
    return frozenset(_TEST_OWNED_ENV)


@pytest.fixture
def no_otari_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the suite's own ``OTARI_*`` variables too, leaving the prefix empty.

    The suite already hides the machine's own settings, so this is for a test that
    must see no Otari configuration at all, such as one running `otari hook` and
    checking it reports a test's gates to no real server.
    """
    for name in [name for name in os.environ if is_hidden_env_name(name)]:
        monkeypatch.delenv(name)


class TaskGroupMcpTransport:
    """A fake MCP transport that holds an anyio task group open, as the streamable HTTP client does.

    ``may_open`` and ``may_close`` gate opening and closing.
    ``ignore_cancel`` makes both gates outlast a cancellation.
    """

    def __init__(self) -> None:
        self.may_open = asyncio.Event()
        self.may_open.set()
        self.may_close = asyncio.Event()
        self.may_close.set()
        self.ignore_cancel = False
        self.close_error: BaseException | None = None
        self.entered_in: asyncio.Task[Any] | None = None
        self.exited_in: list[asyncio.Task[Any] | None] = []
        # What the transport's block exited with, ``None`` for a clean exit.
        self.exited_with: list[BaseException | None] = []

    async def _pass(self, gate: asyncio.Event) -> None:
        while True:
            try:
                await gate.wait()
                return
            except asyncio.CancelledError:
                if not self.ignore_cancel:
                    raise

    @contextlib.asynccontextmanager
    async def __call__(self, *args: Any, **kwargs: Any) -> AsyncIterator[tuple[object, object]]:
        await self._pass(self.may_open)
        async with anyio.create_task_group():
            self.entered_in = asyncio.current_task()
            try:
                yield object(), object()
            except BaseException as exc:
                self.exited_with.append(exc)
                raise
            self.exited_with.append(None)
            await self._pass(self.may_close)
            self.exited_in.append(asyncio.current_task())
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture
def mcp_task_group_transport(monkeypatch: pytest.MonkeyPatch) -> TaskGroupMcpTransport:
    """Serve every MCP connection through one :class:`TaskGroupMcpTransport`."""
    transport = TaskGroupMcpTransport()
    monkeypatch.setattr("gateway.services.mcp_client.streamable_http_client", transport)
    return transport


def _new_sqlite_file(database_url: str) -> Path | None:
    from sqlalchemy.engine import make_url

    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or not url.database or url.database == ":memory:" or "uri" in url.query:
        return None
    path = Path(url.database)
    return path if path.parent.is_dir() and not path.exists() else None


@pytest.fixture(autouse=True)
def _cheap_password_hashing(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hash passwords at bcrypt's lowest cost, unless the test is marked ``production_password_cost``.

    Each hash or check at the production cost takes about a fifth of a second.
    A hash verifies the same way at any cost, so no other test needs the production cost.
    """
    if request.node.get_closest_marker("production_password_cost") is not None:
        return
    monkeypatch.setattr("gateway.services.password_service._BCRYPT_ROUNDS", 4)
    # The stand-in hash is cached per process, so a test must not reuse one minted at another cost.
    monkeypatch.setattr("gateway.services.password_service._absent_password_hash", None)


def seed_workspace_id(db: Any) -> Any:
    """The workspace a directly-built request-plane row belongs to.

    Fixtures that insert ``UsageLog`` or ``APIKey`` rows through a sync session
    skip the routes that would resolve a workspace for them, and the column is
    NOT NULL. The migration seeds a default; this finds it, and creates one on a
    schema built by ``create_all`` rather than by migrations.

    Imports inside the function: this module puts ``src`` on ``sys.path`` at
    import time, so a top-level ``gateway`` import here would run before that.
    """
    from gateway.models.tenancy import Organization, Workspace

    workspace = (
        db.query(Workspace)
        .join(Organization, Organization.id == Workspace.organization_id)
        .filter(Organization.slug == "default")
        .first()
    )
    if workspace is not None:
        return workspace.id

    organization = db.query(Organization).filter(Organization.slug == "default").first()
    if organization is None:
        organization = Organization(name="Default organization", slug="default")
        db.add(organization)
        db.flush()
    workspace = Workspace(name="Default workspace", organization_id=organization.id)
    db.add(workspace)
    db.flush()
    return workspace.id
