import argparse
import re
import shutil
import sys
import zlib
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if "gateway" in sys.modules:
    del sys.modules["gateway"]


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
    ``_stop_refresher`` now guards against. A test that wants the real refresher
    calls it directly rather than through the lifespan.
    """

    async def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr("gateway.main.run_discovery_refresher", _noop)
    monkeypatch.setattr("gateway.main.run_catalog_refresher", _noop)


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
def _reset_default_pricing() -> Generator[None, None, None]:
    """Restore process-wide pricing state to its default before each test.

    ``configure_default_pricing`` is set at app startup, so a test that builds an
    app with a different ``default_pricing`` would otherwise leak that state into
    later tests that call ``find_model_pricing`` directly. Reset to off, matching
    the config field's opt-in default; tests that need defaults enable explicitly.

    Also clear the memoized genai-prices resolutions so a real price cached by one
    test cannot mask another test that patches ``calc_price`` to fail.
    """
    from gateway.services.pricing_refresh_service import reset_price_refresh_state
    from gateway.services.pricing_service import configure_default_pricing, configure_provider_types

    configure_default_pricing(False)
    configure_provider_types(None)
    reset_price_refresh_state()
    yield
    configure_default_pricing(False)
    configure_provider_types(None)
    reset_price_refresh_state()


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
