import sys

import pytest

import gateway.cli as gateway_cli
from gateway.core.config import GatewayConfig


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
    assert config.database_url == "sqlite:///./otari-gateway.db"
    assert config.bootstrap_api_key is True
