"""The suite's isolation from the machine it runs on.

These cover ``_hide_the_machines_own_configuration`` in the root conftest, which
runs from ``pytest_configure``.
"""

import os
from pathlib import Path

import pytest

import conftest
from conftest import REPOSITORY_ENV_FILE, is_hidden_env_name, is_repository_env_file
from gateway.core import config as config_module
from gateway.core.config import GatewayConfig, load_config


def test_the_settings_class_reads_no_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OTARI_MASTER_KEY=from-an-env-file\n", encoding="utf-8")

    assert GatewayConfig().master_key is None


def test_an_env_file_a_test_names_is_still_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The carve-out. Withholding every env file would take out ``load_config``
    # reading one beside the config file it was given.
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OTARI_MASTER_KEY=from-a-test-of-its-own\n", encoding="utf-8")
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("{}\n", encoding="utf-8")
    monkeypatch.delenv("OTARI_MASTER_KEY", raising=False)

    assert load_config(str(config_file)).master_key == "from-a-test-of-its-own"


def test_a_name_in_any_case_is_hidden() -> None:
    # The settings class reads the environment case-insensitively, so a lowercase
    # name is as much a leak as an uppercase one.
    assert is_hidden_env_name("otari_master_key") is True
    assert is_hidden_env_name("Otari_Master_Key") is True
    assert is_hidden_env_name("otel_exporter_otlp_endpoint") is True
    assert is_hidden_env_name("ANTHROPIC_API_KEY") is False


def test_naming_no_env_file_is_held_to_the_same_rule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A call naming nothing would otherwise reach the repository's own file. The
    # file stands in for it, because a checkout carries no env file of its own.
    stand_in = tmp_path / ".env"
    stand_in.write_text("OTARI_MASTER_KEY=from-the-repositorys-own-file\n", encoding="utf-8")
    monkeypatch.setattr(conftest, "REPOSITORY_ENV_FILE", stand_in)
    monkeypatch.chdir(tmp_path)

    assert is_repository_env_file(None) is True


def test_naming_no_env_file_reads_none_of_this_repositorys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Dotenv resolves a path of its own from the file of whoever called it, so a
    # caller inside this repository finds this repository's env file whatever
    # directory it is working in. The guard resolves the path itself instead.
    monkeypatch.chdir(tmp_path)
    before = {name for name in os.environ if is_hidden_env_name(name)}
    # Through the module's own binding, which is the one the suite rebinds.
    guarded_load_dotenv = vars(config_module)["load_dotenv"]

    assert guarded_load_dotenv() is False
    assert {name for name in os.environ if is_hidden_env_name(name)} == before


@pytest.mark.parametrize(
    "spelling",
    [
        # The control, and the only shape a lexical comparison also gets right.
        pytest.param(REPOSITORY_ENV_FILE, id="as-a-path"),
        pytest.param(str(REPOSITORY_ENV_FILE), id="as-a-string"),
        pytest.param(REPOSITORY_ENV_FILE.parent / "docs" / ".." / ".env", id="through-a-parent-segment"),
    ],
)
def test_this_repositorys_env_file_is_recognized_however_it_is_spelled(spelling: str | Path) -> None:
    assert is_repository_env_file(spelling) is True


def test_it_is_recognized_through_a_symlinked_checkout(tmp_path: Path) -> None:
    # The shape a lexical comparison misses most quietly, because a checkout
    # reached through a link has an entirely different prefix.
    linked_root = tmp_path / "checkout"
    try:
        linked_root.symlink_to(REPOSITORY_ENV_FILE.parent, target_is_directory=True)
    except OSError as error:  # pragma: no cover - a link needs a privilege on Windows
        pytest.skip(f"this machine cannot make a symlink: {error}")

    assert is_repository_env_file(linked_root / ".env") is True


def test_an_env_file_elsewhere_is_not_mistaken_for_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert is_repository_env_file(tmp_path / ".env") is False
    # Naming nothing from a directory with no env file above it finds none.
    assert is_repository_env_file(None) is False


def test_only_the_suites_own_variables_are_set(test_owned_env_names: frozenset[str]) -> None:
    present = {name for name in os.environ if is_hidden_env_name(name)}

    assert present == test_owned_env_names
