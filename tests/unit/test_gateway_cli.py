import logging
import sys
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pytest
import uvicorn
from click.testing import CliRunner

import gateway.cli as gateway_cli
from gateway.core.config import GatewayConfig


@dataclass
class ServeCapture:
    """Records observable side effects of the serve command under test."""

    log_level: int | None = None
    uvicorn_calls: int = 0


@pytest.fixture
def serve_stubs(monkeypatch: pytest.MonkeyPatch) -> ServeCapture:
    """Stub out config loading, app creation, and the uvicorn server for serve tests.

    Captures the log level passed to setup_logger and how many times uvicorn.run
    was invoked, so tests can assert CLI behavior without starting a real server
    or touching a database.
    """
    captured = ServeCapture()

    def fake_load_config(config_path: str | None = None) -> GatewayConfig:
        return GatewayConfig(master_key="test-master-key")

    def fake_setup_logger(level: int) -> None:
        captured.log_level = level

    def fake_create_app(config: GatewayConfig) -> object:
        return object()

    def fake_uvicorn_run(*args: object, **kwargs: object) -> None:
        captured.uvicorn_calls += 1

    monkeypatch.setattr(gateway_cli, "load_config", fake_load_config)
    monkeypatch.setattr(gateway_cli, "setup_logger", fake_setup_logger)
    monkeypatch.setattr(gateway_cli, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)
    return captured


def test_serve_log_level_symbolic_name(serve_stubs: ServeCapture) -> None:
    result = CliRunner().invoke(gateway_cli.serve, ["--log-level", "info"])
    assert result.exit_code == 0, result.output
    assert serve_stubs.log_level == logging.INFO
    assert serve_stubs.uvicorn_calls == 1


def test_serve_log_level_symbolic_uppercase(serve_stubs: ServeCapture) -> None:
    result = CliRunner().invoke(gateway_cli.serve, ["--log-level", "DEBUG"])
    assert result.exit_code == 0, result.output
    assert serve_stubs.log_level == logging.DEBUG


def test_serve_log_level_numeric_backcompat(serve_stubs: ServeCapture) -> None:
    result = CliRunner().invoke(gateway_cli.serve, ["--log-level", "20"])
    assert result.exit_code == 0, result.output
    assert serve_stubs.log_level == 20


def test_serve_log_level_invalid_is_rejected(serve_stubs: ServeCapture) -> None:
    result = CliRunner().invoke(gateway_cli.serve, ["--log-level", "bogus"])
    assert result.exit_code != 0
    assert "not a valid log level" in result.output
    assert serve_stubs.uvicorn_calls == 0


def test_serve_default_workers_starts_server(serve_stubs: ServeCapture) -> None:
    result = CliRunner().invoke(gateway_cli.serve, [])
    assert result.exit_code == 0, result.output
    assert serve_stubs.uvicorn_calls == 1


def test_serve_workers_greater_than_one_is_rejected(serve_stubs: ServeCapture) -> None:
    result = CliRunner().invoke(gateway_cli.serve, ["--workers", "4"])
    assert result.exit_code != 0
    assert "does not support running more than one worker" in result.output
    assert serve_stubs.uvicorn_calls == 0


def test_main_invokes_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_cli() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(gateway_cli, "cli", fake_cli)
    monkeypatch.setattr(sys, "argv", ["gateway", "serve"])

    gateway_cli.main()

    assert called


def test_gateway_config_defaults_to_sqlite() -> None:
    config = GatewayConfig()
    assert config.database_url == "sqlite:///./otari.db"
    assert config.bootstrap_api_key is True


def test_gen_secret_key_prints_a_usable_fernet_key() -> None:
    from cryptography.fernet import Fernet

    result = CliRunner().invoke(gateway_cli.cli, ["gen-secret-key"])
    assert result.exit_code == 0
    key = result.output.strip()
    # Round-trips through Fernet, so it is a valid key the secret box can use.
    box = Fernet(key.encode())
    assert box.decrypt(box.encrypt(b"x")) == b"x"


# -- the out-of-band migration path ---------------------------------------------
#
# ``auto_migrate=false`` is the posture that most needs these commands: it is
# the deployment whose tables no boot will create. A bootstrap's chains have to
# come through here too, or a plugin's tables exist only where DDL at boot is
# allowed.


@dataclass
class MigrationCapture:
    """What the CLI asked the shared chain runner to do."""

    database_url: str | None = None
    revision: str | None = None
    chains: tuple[str, ...] = ()
    init_chains: tuple[str, ...] = ()


@pytest.fixture
def migration_stubs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[MigrationCapture]:
    """A bootstrap contributing one chain, with the runner and ``init_db`` stubbed.

    The chains are real ``MigrationContribution``s off a real container, since
    it is the CLI's reading of the container that is under test; only the part
    that would touch a database is replaced.
    """
    import gateway.core.database as database_module
    import gateway.db as db_module

    captured = MigrationCapture()

    (tmp_path / "chain_bootstrap.py").write_text(
        "from gateway.container import MigrationContribution\n\n\n"
        "def register(container):\n"
        "    container.contribute_migrations(\n"
        "        MigrationContribution(name='demo', script_location='/tmp/demo', version_table='demo_version')\n"
        "    )\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("chain_bootstrap", None)

    def fake_load_config(config_path: str | None = None) -> GatewayConfig:
        return GatewayConfig(master_key="test-master-key", bootstrap="chain_bootstrap:register")

    def fake_run_migrations(
        database_url: str,
        contributions: object = (),
        *,
        revision: str = "head",
    ) -> None:
        captured.database_url = database_url
        captured.revision = revision
        captured.chains = tuple(c.name for c in contributions)  # type: ignore[attr-defined]

    def fake_init_db(config: GatewayConfig, *, migration_contributions: object = ()) -> None:
        captured.init_chains = tuple(c.name for c in migration_contributions)  # type: ignore[attr-defined]

    monkeypatch.setattr(gateway_cli, "load_config", fake_load_config)
    monkeypatch.setattr(database_module, "run_migrations", fake_run_migrations)
    monkeypatch.setattr(db_module, "init_db", fake_init_db)
    yield captured
    sys.modules.pop("chain_bootstrap", None)


def test_migrate_runs_the_contributed_chains_too(migration_stubs: MigrationCapture) -> None:
    result = CliRunner().invoke(gateway_cli.cli, ["migrate"])

    assert result.exit_code == 0, result.output
    assert migration_stubs.revision == "head"
    assert migration_stubs.chains == ("demo",)
    assert "demo" in result.output


def test_migrate_to_a_pinned_revision_leaves_the_contributed_chains_alone(
    migration_stubs: MigrationCapture,
) -> None:
    """A revision names one in Otari's chain; a contributed history knows nothing about it."""
    result = CliRunner().invoke(gateway_cli.cli, ["migrate", "--revision", "abc123"])

    assert result.exit_code == 0, result.output
    assert migration_stubs.revision == "abc123"
    assert migration_stubs.chains == ()
    assert "leaving the contributed chains alone" in result.output


def test_migrate_to_heads_still_runs_the_contributed_chains(migration_stubs: MigrationCapture) -> None:
    """``heads`` names the same target as ``head``: Otari's chain is single-headed."""
    result = CliRunner().invoke(gateway_cli.cli, ["migrate", "--revision", "heads"])

    assert result.exit_code == 0, result.output
    assert migration_stubs.chains == ("demo",)


def test_migrate_reports_a_bad_bootstrap_selector_instead_of_a_traceback(
    migration_stubs: MigrationCapture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A selector that cannot load is the operator's typo, not a bug."""
    monkeypatch.setattr(
        gateway_cli,
        "load_config",
        lambda config_path=None: GatewayConfig(master_key="k", bootstrap="no_such_module:register"),
    )

    result = CliRunner().invoke(gateway_cli.cli, ["migrate"])

    assert result.exit_code == 1
    assert "Could not load the configured bootstrap" in result.output
    assert "Traceback" not in result.output


def test_migrate_reports_a_failed_chain_instead_of_a_traceback(
    migration_stubs: MigrationCapture, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gateway.core.database as database_module

    def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("revision d5b7f9a1c3e6 is not present")

    monkeypatch.setattr(database_module, "run_migrations", explode)

    result = CliRunner().invoke(gateway_cli.cli, ["migrate"])

    assert result.exit_code == 1
    assert "Migration failed: revision d5b7f9a1c3e6 is not present" in result.output


def test_init_db_creates_the_contributed_chains_tables_too(migration_stubs: MigrationCapture) -> None:
    result = CliRunner().invoke(gateway_cli.cli, ["init-db"])

    assert result.exit_code == 0, result.output
    assert migration_stubs.init_chains == ("demo",)
