import logging
import sys
from dataclasses import dataclass

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
