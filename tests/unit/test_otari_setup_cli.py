"""`otari setup` and the commands that start a tool: what is written, what is run, and where the key is not."""

import json
import stat
import tomllib
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner

from otari_agent import setup as otari_setup
from otari_agent.cli import cli
from otari_agent.setup import KeyStore, pick_models

URL = "https://api.otari.example"
KEY = "otk_v1_us_secret-value"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OTARI_SETUP_HOME", str(tmp_path))
    monkeypatch.setenv("OTARI_SETUP_NO_KEYCHAIN", "1")
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("OTARI_URL", raising=False)
    return tmp_path


def test_the_default_model_is_a_coding_model_the_workspace_serves() -> None:
    assert pick_models(["a/b", "z-ai/glm-5.2", "moonshotai/kimi-k3"], None)[0] == "moonshotai/kimi-k3"


def test_a_requested_model_leads_only_when_it_is_served() -> None:
    assert pick_models(["a/b", "c/d"], "c/d") == ["c/d", "a/b"]
    assert pick_models(["a/b", "c/d"], "not/served") == ["a/b", "c/d"]


def test_the_key_file_is_owner_only_and_round_trips(home: Path) -> None:
    store = KeyStore(URL)
    store.save(KEY)
    path = home / ".config" / "otari" / "keys" / "api.otari.example.key"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert store.load() == KEY
    store.forget()
    assert store.load() is None


def test_no_tool_file_holds_the_key(home: Path) -> None:
    store = KeyStore(URL)
    store.save(KEY)
    written = [
        configure(URL, store, ["vendor/model"])
        for configure in otari_setup._CONFIGURE.values()  # noqa: SLF001
    ]
    for item in written:
        for path in item.files:
            assert KEY not in path.read_text(encoding="utf-8")
            assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0


def test_claude_code_gets_a_settings_file_of_its_own_that_reads_the_key_by_command(home: Path) -> None:
    store = KeyStore(URL)
    otari_setup._configure_claude_code(URL, store, ["vendor/model", "other/model"])  # noqa: SLF001
    settings = json.loads((home / ".claude" / "otari.json").read_text(encoding="utf-8"))
    assert settings["apiKeyHelper"] == store.helper()
    assert settings["env"]["ANTHROPIC_BASE_URL"] == f"{URL}/api"
    assert settings["env"]["ANTHROPIC_MODEL"] == "vendor/model"
    assert not (home / ".claude" / "settings.json").exists()


def test_codex_gets_a_profile_file_whose_auth_is_a_command(home: Path) -> None:
    store = KeyStore(URL)
    otari_setup._configure_codex(URL, store, ["vendor/model"])  # noqa: SLF001
    profile = tomllib.loads((home / ".codex" / "otari.config.toml").read_text(encoding="utf-8"))
    provider = profile["model_providers"]["otari"]
    assert profile["model"] == "vendor/model"
    assert provider["base_url"] == f"{URL}/api/v1"
    assert provider["wire_api"] == "responses"
    assert [provider["auth"]["command"], *provider["auth"]["args"]] == store.helper_argv()
    assert not (home / ".codex" / "config.toml").exists()


def test_opencode_reads_the_key_from_the_environment(home: Path) -> None:
    store = KeyStore(URL)
    otari_setup._configure_opencode(URL, store, ["vendor/model"])  # noqa: SLF001
    config = json.loads((home / ".config" / "otari" / "opencode.json").read_text(encoding="utf-8"))
    assert config["provider"]["otari"]["options"] == {"baseURL": f"{URL}/api/v1", "apiKey": "{env:OTARI_API_KEY}"}


@pytest.fixture
def launched(home: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[str], dict[str, str]]]:
    """A machine `otari setup` has run on, with the hand-over to the tool recorded instead of made."""
    store = KeyStore(URL)
    store.save(KEY)
    written = [configure(URL, store, ["vendor/model"]) for configure in otari_setup._CONFIGURE.values()]  # noqa: SLF001
    otari_setup._write_manifest(URL, written, "vendor/model")  # noqa: SLF001
    calls: list[tuple[str, list[str], dict[str, str]]] = []
    monkeypatch.setattr(otari_setup.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(otari_setup, "_exec", lambda binary, argv, env: calls.append((binary, argv, env)))
    return calls


def test_otari_claude_starts_claude_code_with_its_otari_settings(
    home: Path, launched: list[tuple[str, list[str], dict[str, str]]]
) -> None:
    result = CliRunner().invoke(cli, ["claude", "-p", "hello", "--help"])
    assert result.exit_code == 0, result.output
    binary, argv, env = launched[0]
    assert binary == "claude"
    assert argv == ["claude", "--settings", str(home / ".claude" / "otari.json"), "-p", "hello", "--help"]
    assert "OTARI_API_KEY" not in env


def test_otari_codex_starts_codex_on_the_otari_profile(launched: list[tuple[str, list[str], dict[str, str]]]) -> None:
    assert CliRunner().invoke(cli, ["codex", "exec", "--full-auto"]).exit_code == 0
    assert launched[0][:2] == ("codex", ["codex", "--profile", "otari", "exec", "--full-auto"])


def test_otari_opencode_hands_opencode_the_key_and_its_config(
    home: Path, launched: list[tuple[str, list[str], dict[str, str]]]
) -> None:
    assert CliRunner().invoke(cli, ["opencode", "run", "hi"]).exit_code == 0
    binary, argv, env = launched[0]
    assert (binary, argv) == ("opencode", ["opencode", "run", "hi"])
    assert env["OTARI_API_KEY"] == KEY
    assert env["OPENCODE_CONFIG"] == str(home / ".config" / "otari" / "opencode.json")


def test_the_start_hint_names_otari_only_when_it_outlives_this_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(otari_setup.shutil, "which", lambda _: "/opt/homebrew/bin/otari")
    assert otari_setup._invocation() == "otari"  # noqa: SLF001
    monkeypatch.setattr(otari_setup.shutil, "which", lambda _: f"{otari_setup.sys.prefix}/bin/otari")
    assert otari_setup._invocation() == "uvx --from otari-agent otari"  # noqa: SLF001
    monkeypatch.setattr(otari_setup.shutil, "which", lambda _: None)
    assert otari_setup._invocation() == "uvx --from otari-agent otari"  # noqa: SLF001


def test_a_tool_is_not_started_before_setup_has_run(home: Path) -> None:
    result = CliRunner().invoke(cli, ["claude"])
    assert result.exit_code != 0
    assert "otari setup" in result.output


def test_setup_stops_before_the_browser_where_the_gateway_has_not_registered_the_cli() -> None:
    class Client:
        def get(self, url: str, params: dict[str, str]) -> httpx.Response:
            assert params["client_id"] == "otari-cli"
            return httpx.Response(404, json={"detail": "Not Found"})

    with pytest.raises(otari_setup.SetupError, match="otari-cli"):
        otari_setup._require_cli_app(Client(), URL)  # type: ignore[arg-type]  # noqa: SLF001


def test_setup_needs_to_be_told_which_otari(home: Path) -> None:
    result = CliRunner().invoke(cli, ["setup"])
    assert result.exit_code != 0
    assert "--url" in result.output


def test_logout_removes_what_setup_wrote(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = KeyStore(URL)
    store.save(KEY)
    item = otari_setup._configure_codex(URL, store, ["vendor/model"])  # noqa: SLF001
    otari_setup._write_manifest(URL, [item], "vendor/model")  # noqa: SLF001
    revoked: list[str] = []

    class Client:
        def __init__(self, **_: object) -> None: ...
        def __enter__(self) -> "Client":
            return self

        def __exit__(self, *_: object) -> None: ...
        def post(self, url: str, data: dict[str, str]) -> None:
            revoked.append(data["token"])

    monkeypatch.setattr(otari_setup.httpx, "Client", Client)
    result = CliRunner().invoke(cli, ["logout"])
    assert result.exit_code == 0
    assert revoked == [KEY]
    assert store.load() is None
    assert not item.files[0].exists()
