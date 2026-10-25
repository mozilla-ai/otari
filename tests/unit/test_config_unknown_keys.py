"""Unknown top-level config keys are warned about at load, not refused."""

import logging
from pathlib import Path

import pytest

from gateway.core.config import load_config


def test_unknown_top_level_keys_are_warned_about(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "config.yml"
    config_file.write_text("require_api_key: true\nrate_limit_rpm: 30\nport: 9000\nzzz_typo: 1\n")

    with caplog.at_level(logging.WARNING):
        config = load_config(str(config_file))

    assert config.port == 9000
    assert config.rate_limit_rpm == 30
    warnings = [r.getMessage() for r in caplog.records if "unknown config keys" in r.getMessage()]
    assert warnings == ["Ignoring unknown config keys (no setting reads them): require_api_key, zzz_typo"]


def test_a_config_of_known_keys_logs_no_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "config.yml"
    config_file.write_text("port: 9000\n")

    with caplog.at_level(logging.WARNING):
        load_config(str(config_file))

    assert "unknown config keys" not in caplog.text
