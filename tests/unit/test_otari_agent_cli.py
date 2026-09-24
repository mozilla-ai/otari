"""The `otari` console script as the otari-agent distribution ships it."""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

import gateway.cli as gateway_cli
import otari_agent
import otari_agent.hook as hook_cli
from otari_agent.cli import SERVER_COMMANDS, OtariGroup, cli

# Spelled out rather than read from either side, so a command dropped from
# SERVER_COMMANDS or from the gateway group fails here instead of vanishing.
_EXPECTED_SERVER_COMMANDS = frozenset({"serve", "init-db", "migrate", "gen-secret-key", "routing"})

# What a Homebrew install of the light CLI does not have. The gateway is the
# reason the split exists; the rest is what gateway.core.config drags in.
_SERVER_STACK = ("gateway", "uvicorn", "any_llm", "sqlalchemy", "sqlmodel", "fastapi", "pydantic", "pydantic_settings")


def _run_isolated(code: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False, env=env
    )


# `--version` is bound when otari_agent.cli is imported, so each version test
# gets an interpreter of its own with the environment it asserts on.
_VERSION_CODE = (
    "from click.testing import CliRunner\n"
    "from otari_agent.cli import cli\n"
    "result = CliRunner().invoke(cli, ['--version'])\n"
    "assert result.exit_code == 0, result.output\n"
    "print(result.output, end='')\n"
)


def test_importing_the_light_cli_loads_no_server_stack() -> None:
    code = (
        "import sys\n"
        "import otari_agent.cli, otari_agent.hook, otari_agent.usage_import, otari_agent.settings\n"
        f"loaded = {{name.split('.')[0] for name in sys.modules}} & set({_SERVER_STACK!r})\n"
        "assert not loaded, sorted(loaded)\n"
    )
    result = _run_isolated(code)
    assert result.returncode == 0, result.stdout + result.stderr


def test_hook_help_does_not_attach_the_gateway() -> None:
    # The dev venv has the gateway installed; the hook path must still not import it.
    code = (
        "import sys\n"
        "from click.testing import CliRunner\n"
        "from otari_agent.cli import cli\n"
        "result = CliRunner().invoke(cli, ['hook', '--help'])\n"
        "assert result.exit_code == 0, result.output\n"
        "assert 'gateway' not in sys.modules\n"
    )
    result = _run_isolated(code)
    assert result.returncode == 0, result.stdout + result.stderr


def test_help_lists_both_command_sets_when_the_gateway_is_installed() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    listed = set(re.findall(r"^\s{2}(\S+)", result.output, re.MULTILINE))
    assert {"hook", "import"} | _EXPECTED_SERVER_COMMANDS <= listed


def test_a_server_command_resolves_through_the_light_group() -> None:
    result = CliRunner().invoke(cli, ["serve", "--help"])
    assert result.exit_code == 0, result.output
    assert "Start the Otari server" in result.output


@pytest.mark.parametrize("name", sorted(_EXPECTED_SERVER_COMMANDS))
def test_a_server_command_without_the_gateway_says_why(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda module, *args: None if module == "gateway" else find_spec(module, *args)
    )
    result = CliRunner().invoke(OtariGroup(), [name])
    assert result.exit_code == 2
    assert f"'{name}' is a server command" in result.output
    assert "does not include the otari gateway" in result.output


def test_an_unknown_command_without_the_gateway_keeps_the_usual_error(monkeypatch: pytest.MonkeyPatch) -> None:
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda module, *args: None if module == "gateway" else find_spec(module, *args)
    )
    result = CliRunner().invoke(OtariGroup(), ["server"])
    assert result.exit_code == 2
    assert "No such command 'server'" in result.output


def test_version_reads_the_stamp_when_the_deployment_names_none() -> None:
    env = {name: value for name, value in os.environ.items() if name != "OTARI_VERSION"}
    result = _run_isolated(_VERSION_CODE, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == f"otari, version {otari_agent.__version__}"


def test_version_prefers_the_deployment_version() -> None:
    # The Docker image installs the CLI unstamped and sets OTARI_VERSION to its tag.
    result = _run_isolated(_VERSION_CODE, env={**os.environ, "OTARI_VERSION": "v9.9.9"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "otari, version v9.9.9"


def test_register_attaches_every_server_command() -> None:
    group = click.Group("probe")
    gateway_cli.register(group)
    assert set(group.commands) == _EXPECTED_SERVER_COMMANDS


def test_the_light_cli_names_every_server_command() -> None:
    assert SERVER_COMMANDS == _EXPECTED_SERVER_COMMANDS


def test_binary_path_prefers_the_invoked_link_unresolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A Homebrew layout: bin/otari is a symlink into a versioned Cellar directory.
    cellar = tmp_path / "Cellar" / "otari" / "1.2.3" / "libexec" / "bin"
    cellar.mkdir(parents=True)
    (cellar / "otari").write_text("#!/bin/sh\n")
    (tmp_path / "bin").mkdir()
    link = tmp_path / "bin" / "otari"
    link.symlink_to(cellar / "otari")
    monkeypatch.setattr(sys, "argv", [str(link), "hook", "setup"])
    assert hook_cli._otari_binary_path() == str(link)


def test_binary_path_falls_back_to_the_interpreter_sibling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["pytest", "tests/unit"])
    assert hook_cli._otari_binary_path() == str(Path(sys.executable).with_name("otari"))


@pytest.mark.parametrize("entry", ["otari_agent.cli:main", "gateway.cli:main"])
def test_both_entry_points_still_exist(entry: str) -> None:
    module_name, attribute = entry.split(":")
    module = __import__(module_name, fromlist=[attribute])
    assert callable(getattr(module, attribute))
