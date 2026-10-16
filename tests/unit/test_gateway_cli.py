import logging
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
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
    uvicorn_kwargs: dict[str, object] = field(default_factory=dict)
    config: GatewayConfig = field(default_factory=lambda: GatewayConfig(master_key="test-master-key"))


@pytest.fixture
def serve_stubs(monkeypatch: pytest.MonkeyPatch) -> ServeCapture:
    """Stub out config loading, app creation, and the uvicorn server for serve tests.

    Captures the log level passed to setup_logger and how many times uvicorn.run
    was invoked, so tests can assert CLI behavior without starting a real server
    or touching a database.
    """
    captured = ServeCapture()

    def fake_load_config(config_path: str | None = None) -> GatewayConfig:
        return captured.config

    def fake_setup_logger(level: int) -> None:
        captured.log_level = level

    def fake_create_app(config: GatewayConfig) -> object:
        return object()

    def fake_uvicorn_run(*args: object, **kwargs: object) -> None:
        captured.uvicorn_calls += 1
        captured.uvicorn_kwargs = kwargs

    monkeypatch.setattr(gateway_cli, "load_config", fake_load_config)
    monkeypatch.setattr(gateway_cli, "setup_logger", fake_setup_logger)
    monkeypatch.setattr("gateway.main.create_app", fake_create_app)
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


def test_serve_leaves_forwarded_allow_ips_to_uvicorn_when_unset(serve_stubs: ServeCapture) -> None:
    result = CliRunner().invoke(gateway_cli.serve, [])
    assert result.exit_code == 0, result.output
    assert serve_stubs.uvicorn_kwargs["forwarded_allow_ips"] is None


def test_serve_passes_forwarded_allow_ips_to_uvicorn(serve_stubs: ServeCapture) -> None:
    serve_stubs.config = GatewayConfig(master_key="test-master-key", forwarded_allow_ips="*")
    result = CliRunner().invoke(gateway_cli.serve, [])
    assert result.exit_code == 0, result.output
    assert serve_stubs.uvicorn_kwargs["forwarded_allow_ips"] == "*"


def test_main_invokes_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_cli() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(gateway_cli, "cli", fake_cli)
    monkeypatch.setattr(sys, "argv", ["gateway", "serve"])

    gateway_cli.main()

    assert called


def test_cli_import_does_not_load_server_application() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import gateway.cli, sys; assert 'gateway.main' not in sys.modules"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="The extraction address-space limit is enforced on Linux")
def test_serve_can_spawn_memory_bounded_extraction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Spawn reloads the launcher in the child, including its top-level CLI import.
    launcher = tmp_path / "extraction_cli.py"
    launcher.write_text(
        textwrap.dedent(
            """\
            from gateway.cli import main

            if __name__ == "__main__":
                import asyncio
                import uvicorn
                from gateway.services.web_extraction import ExtractionSupervisor

                async def extract():
                    supervisor = ExtractionSupervisor()
                    try:
                        result = await supervisor.extract_html(
                            "<html><body><article><h1>CLI extraction probe</h1>"
                            "<p>The real worker must extract this document within its "
                            "unchanged memory limit.</p></article></body></html>"
                        )
                        assert "CLI extraction probe" in result.text, result.text
                    finally:
                        supervisor.close()

                def run_extraction(app, **kwargs):
                    asyncio.run(extract())
                    print("CLI_EXTRACTION_OK")

                uvicorn.run = run_extraction
                main()
            """
        )
    )
    config = tmp_path / "config.yml"
    config.write_text("mode: standalone\nmaster_key: test-master-key\ndatabase_url: 'sqlite:///:memory:'\n")
    for name in list(os.environ):
        if name.startswith(("OTARI_", "GATEWAY_")):
            monkeypatch.delenv(name)
    result = subprocess.run(
        [sys.executable, str(launcher), "serve", "--config", str(config)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CLI_EXTRACTION_OK" in result.stdout


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


def test_gen_provider_account_pepper_prints_a_pepper_the_gateway_accepts() -> None:
    first = CliRunner().invoke(gateway_cli.cli, ["gen-provider-account-pepper"])
    second = CliRunner().invoke(gateway_cli.cli, ["gen-provider-account-pepper"])

    assert first.exit_code == 0
    pepper = first.output.strip()
    assert GatewayConfig(provider_account_pepper=pepper).provider_account_pepper == pepper
    assert pepper != second.output.strip()


_SECRET_DATABASE_URL = "postgresql+psycopg://u:sekret@h:5432/db?password=alsosekret&sslpassword=pem123"


def test_init_db_does_not_echo_database_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gateway.db.init_db", lambda config: None)

    result = CliRunner().invoke(gateway_cli.cli, ["init-db", "--database-url", _SECRET_DATABASE_URL])

    assert result.exit_code == 0, result.output
    assert "Initializing database: postgresql+psycopg://u:***@h:5432/db?password=***&sslpassword=***" in result.output
    for secret in ("sekret", "alsosekret", "pem123"):
        assert secret not in result.output


def test_migrate_does_not_echo_database_secrets_but_passes_them_to_alembic(monkeypatch: pytest.MonkeyPatch) -> None:
    alembic_env: dict[str, str] = {}

    def fake_run(args: list[str], *, env: dict[str, str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        alembic_env.update(env)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/alembic")
    monkeypatch.setattr("subprocess.run", fake_run)

    result = CliRunner().invoke(gateway_cli.cli, ["migrate", "--database-url", _SECRET_DATABASE_URL])

    assert result.exit_code == 0, result.output
    assert "Running migrations on: postgresql+psycopg://u:***@h:5432/db?password=***&sslpassword=***" in result.output
    for secret in ("sekret", "alsosekret", "pem123"):
        assert secret not in result.output
    assert alembic_env["OTARI_DATABASE_URL"] == _SECRET_DATABASE_URL


def test_init_db_redacts_a_database_url_with_a_malformed_port_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("gateway.db.init_db", lambda config: None)

    result = CliRunner().invoke(gateway_cli.cli, ["init-db", "--database-url", "postgresql://u:sekret@h:bad/db"])

    assert result.exit_code == 0, result.output
    assert "Initializing database: postgresql://u:***@h:bad/db" in result.output
