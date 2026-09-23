"""The light settings reader `otari hook` uses in place of gateway.core.config.load_config."""

import os
from pathlib import Path

import pytest

from otari_agent import settings
from otari_agent.settings import HookSettings, load_settings


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # python-dotenv writes straight into os.environ, so hand it a copy to write into,
    # with the developer's own OTARI_* variables removed.
    environ = {key: value for key, value in os.environ.items() if not key.startswith("OTARI_")}
    monkeypatch.setattr(os, "environ", environ)
    monkeypatch.chdir(tmp_path)


def test_defaults_match_the_gateway_bind_defaults() -> None:
    assert load_settings() == HookSettings(host="0.0.0.0", port=8000, master_key=None)  # noqa: S104


def test_yaml_top_level_keys_are_read(tmp_path: Path) -> None:
    config = tmp_path / "config.yml"
    config.write_text("host: gw.internal\nport: 9100\nmaster_key: from-yaml\nproviders: {}\n", encoding="utf-8")
    assert load_settings(str(config)) == HookSettings(host="gw.internal", port=9100, master_key="from-yaml")


def test_non_empty_env_beats_yaml_and_empty_env_is_unset(tmp_path: Path) -> None:
    config = tmp_path / "config.yml"
    config.write_text("host: gw.internal\nport: 9100\nmaster_key: from-yaml\n", encoding="utf-8")
    os.environ["OTARI_MASTER_KEY"] = "from-env"
    os.environ["OTARI_PORT"] = ""
    assert load_settings(str(config)) == HookSettings(host="gw.internal", port=9100, master_key="from-env")


def test_dotenv_beside_the_config_file_loads_before_the_cwd_one(tmp_path: Path) -> None:
    config_dir = tmp_path / "deploy"
    config_dir.mkdir()
    (config_dir / "config.yml").write_text("port: 9100\n", encoding="utf-8")
    (config_dir / ".env").write_text("OTARI_MASTER_KEY=from-config-dir\n", encoding="utf-8")
    (tmp_path / ".env").write_text("OTARI_MASTER_KEY=from-cwd\nOTARI_HOST=cwd.internal\n", encoding="utf-8")
    resolved = load_settings(str(config_dir / "config.yml"))
    assert resolved == HookSettings(host="cwd.internal", port=9100, master_key="from-config-dir")


def test_a_variable_already_in_the_environment_beats_every_dotenv(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("OTARI_MASTER_KEY=from-dotenv\n", encoding="utf-8")
    os.environ["OTARI_MASTER_KEY"] = "from-process"
    assert load_settings().master_key == "from-process"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("host: [unclosed\n", "could not read"),
        ("- a\n- list\n", "must hold a YAML mapping"),
        ("port: abc\n", "port must be an integer"),
    ],
)
def test_unusable_config_raises_value_error(tmp_path: Path, body: str, message: str) -> None:
    config = tmp_path / "config.yml"
    config.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_settings(str(config))


def test_a_missing_config_file_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="could not read"):
        load_settings(str(tmp_path / "absent.yml"))


def test_api_constants_match_the_gateway() -> None:
    from gateway.core.config import API_KEY_HEADER, API_ROOT

    assert (settings.API_KEY_HEADER, settings.API_ROOT) == (API_KEY_HEADER, API_ROOT)
