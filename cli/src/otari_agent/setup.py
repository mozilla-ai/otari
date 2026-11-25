"""`otari setup`: one browser approval, then every coding tool on this machine works with Otari.

The command asks the gateway for a key through its OAuth app flow (PKCE, with a
loopback callback), so the only thing a person does is approve a spending limit
in the browser. The key is stored once, in the macOS Keychain or an owner-only
file, and each coding tool's settings refer to it by a command or an environment
variable rather than holding it.

Nothing the person already has is edited: every tool gets a settings file of
its own, and `otari claude`, `otari codex` and `otari opencode` start the tool
with it, so a plain `claude` or `codex` behaves as it did before.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import http.server
import json
import os
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import httpx

CLIENT_ID = "otari-cli"
KEYCHAIN_SERVICE = "Otari"
API_ROOT = "/api/v1"
DEFAULT_LIMIT = "10"
DEFAULT_EXPIRATION = "30-days"
EXPIRATIONS = ("1-hour", "1-day", "7-days", "30-days", "90-days", "180-days", "1-year", "never")
APPROVAL_TIMEOUT_SEC = 300
# The first request to a model that has to be loaded can take a while.
TEST_TIMEOUT_SEC = 180
TEST_MESSAGE = "Reply with one short sentence that says the setup worked."
# How many models a tool with a picker of its own is given to choose from.
OFFERED_MODELS = 7
# Open-weight models that code well, best first, as the hosted catalog names them.
CODING_MODELS = (
    "z-ai/glm-5.3",
    "moonshotai/kimi-k3.1-pro",
    "deepseek/deepseek-v4.1-flash-1017",
    "minimax/minimax-m3",
    "moonshotai/kimi-k3",
    "deepseek/deepseek-v4-pro",
    "z-ai/glm-5.2",
)
GUIDES = "https://docs.otari.ai/guides/integrations"


class SetupError(click.ClickException):
    """A failure the person can act on, shown without a traceback."""


# --------------------------------------------------------------------------- paths


def _home() -> Path:
    return Path(os.environ.get("OTARI_SETUP_HOME") or Path.home())


def _state_dir() -> Path:
    return _home() / ".config" / "otari"


def _manifest_path() -> Path:
    return _state_dir() / "setup.json"


def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IRUSR | stat.S_IWUSR
    # Created owner-only rather than narrowed afterwards, so the file is never readable by others.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
    path.chmod(mode)


def _tilde(path: Path) -> str:
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- key store


@dataclass(frozen=True)
class KeyStore:
    """Where the key lives, and the shell command that prints it for a tool that asks."""

    url: str

    @property
    def uses_keychain(self) -> bool:
        return (
            sys.platform == "darwin"
            and shutil.which("security") is not None
            and not os.environ.get("OTARI_SETUP_NO_KEYCHAIN")
        )

    @property
    def _file(self) -> Path:
        host = urllib.parse.urlparse(self.url).netloc.replace(":", "_") or "otari"
        return _state_dir() / "keys" / f"{host}.key"

    @property
    def where(self) -> str:
        return f"your macOS Keychain ({KEYCHAIN_SERVICE})" if self.uses_keychain else _tilde(self._file)

    def helper_argv(self) -> list[str]:
        """The command a tool runs to read the key; it prints the key and nothing else."""
        if self.uses_keychain:
            return ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", self.url, "-w"]
        return ["cat", str(self._file)]

    def helper(self) -> str:
        return shlex.join(self.helper_argv())

    def save(self, key: str) -> None:
        if self.uses_keychain:
            # Through stdin, not argv, so the key never shows in the process list.
            command = shlex.join(
                ["add-generic-password", "-U", "-s", KEYCHAIN_SERVICE, "-a", self.url, "-l", "Otari API key", "-w", key]
            )
            done = subprocess.run(  # noqa: S603
                ["security", "-i"], input=command + "\n", text=True, capture_output=True, check=False
            )
            if done.returncode == 0:
                return
            raise SetupError("Could not store the key in your Keychain. Set OTARI_SETUP_NO_KEYCHAIN=1 to use a file.")
        _write_private(self._file, key)

    def load(self) -> str | None:
        if self.uses_keychain:
            done = subprocess.run(self.helper_argv(), text=True, capture_output=True, check=False)  # noqa: S603
            return done.stdout.strip() or None if done.returncode == 0 else None
        try:
            return self._file.read_text(encoding="utf-8").strip() or None
        except FileNotFoundError:
            return None

    def forget(self) -> None:
        if self.uses_keychain:
            subprocess.run(  # noqa: S603
                ["security", "delete-generic-password", "-s", KEYCHAIN_SERVICE, "-a", self.url],
                capture_output=True,
                check=False,
            )
        with contextlib.suppress(FileNotFoundError):
            self._file.unlink()


# --------------------------------------------------------------------------- approval


@dataclass
class Approval:
    """What the browser sent back to the loopback callback."""

    params: dict[str, str] = field(default_factory=dict)
    done: threading.Event = field(default_factory=threading.Event)
    page: str = ""


_CALLBACK_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Otari CLI</title>{refresh}
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{ color-scheme: light dark; }}
  body {{ margin: 0; min-height: 100vh; display: grid; place-items: center;
    font: 16px/1.5 ui-sans-serif, system-ui, sans-serif; background: Canvas; color: CanvasText; }}
  main {{ max-width: 26rem; padding: 2rem; text-align: center; }}
  h1 {{ font-size: 1.25rem; margin: 0 0 .5rem; }}
  p {{ margin: 0; opacity: .7; }}
  a {{ color: inherit; }}
</style></head>
<body><main><h1>{title}</h1><p>{detail}</p></main></body></html>
"""


def _callback_page(title: str, detail: str, *, onward: str | None = None) -> str:
    refresh = f'<meta http-equiv="refresh" content="2;url={onward}">' if onward else ""
    return _CALLBACK_PAGE.format(title=title, detail=detail, refresh=refresh)


def _serve_callback(approval: Approval, onward: str) -> tuple[http.server.HTTPServer, str]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/callback":
                self.send_error(404)
                return
            approval.params = {name: values[0] for name, values in urllib.parse.parse_qs(parsed.query).items()}
            if "code" in approval.params:
                page = _callback_page(
                    "Otari CLI is connected",
                    "Go back to your terminal. This tab returns to Otari in a moment.",
                    onward=onward,
                )
            else:
                page = _callback_page("Nothing was connected", "You can close this tab and run otari setup again.")
            body = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            approval.done.set()

        def log_message(self, *_: Any) -> None:
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/callback"


def _origin(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _approve(
    client: httpx.Client, url: str, *, limit: str, expiration: str, open_browser: bool, say: Any
) -> tuple[str, list[str], str]:
    """Run the browser approval and return the key, its scopes and the page that manages it."""
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state = secrets.token_urlsafe(24)
    approval = Approval()
    server, redirect_uri = _serve_callback(approval, onward=_origin(url) + "/")
    try:
        query = urllib.parse.urlencode(
            {
                "client_id": CLIENT_ID,
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "state": state,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "limit": limit,
                "expiration": expiration,
            }
        )
        link = f"{url}{API_ROOT}/oauth/apps/authorize?{query}"
        say("Approve the key in your browser:")
        say(f"  {link}")
        if open_browser:
            webbrowser.open(link)
        say(f"Waiting for your approval (up to {APPROVAL_TIMEOUT_SEC // 60} minutes)...")
        if not approval.done.wait(APPROVAL_TIMEOUT_SEC):
            raise SetupError("No approval arrived in time. Run otari setup again when you are ready.")
    finally:
        # Let the page finish sending before the listener goes away.
        time.sleep(0.2)
        server.shutdown()
        server.server_close()

    params = approval.params
    if params.get("state") != state:
        raise SetupError("The approval did not come from this setup run. Nothing was stored.")
    if "error" in params:
        if params["error"] == "access_denied":
            raise SetupError("You declined in the browser. Nothing was created.")
        raise SetupError(f"Otari refused the approval: {params['error']}.")
    if params.get("iss", "").rstrip("/") != _origin(url):
        sender = params.get("iss") or "an unnamed server"
        raise SetupError(f"The approval came from {sender}, not {_origin(url)}. Nothing was stored.")
    response = client.post(
        f"{url}{API_ROOT}/oauth/apps/token",
        data={
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "redirect_uri": redirect_uri,
            "code": params["code"],
            "code_verifier": verifier,
        },
    )
    if response.status_code != httpx.codes.OK:
        raise SetupError("Otari did not issue the key. Run otari setup again.")
    body = response.json()
    return body["access_token"], str(body.get("scope", "")).split(), str(body.get("manage_url", ""))


# --------------------------------------------------------------------------- gateway reads


def _require_cli_app(client: httpx.Client, url: str) -> None:
    """Stop before the browser opens when this gateway has not registered the CLI as an app."""
    response = client.get(
        f"{url}{API_ROOT}/oauth/apps/clients",
        params={"client_id": CLIENT_ID, "redirect_uri": "http://127.0.0.1/callback"},
    )
    if response.status_code != httpx.codes.OK:
        raise SetupError(
            f"{url} does not accept `otari setup` yet. Its operator registers the CLI as the "
            f'`{CLIENT_ID}` app first: see "Connect an app with OAuth" in the Otari docs.'
        )


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _key_works(client: httpx.Client, url: str, key: str) -> bool:
    response = client.get(f"{url}{API_ROOT}/models", headers=_auth(key))
    return response.status_code == httpx.codes.OK


def _models(client: httpx.Client, url: str, key: str) -> list[str]:
    response = client.get(f"{url}{API_ROOT}/models", headers=_auth(key))
    if response.status_code != httpx.codes.OK:
        return []
    return [str(model["id"]) for model in response.json().get("data", []) if model.get("id")]


def pick_models(served: list[str], wanted: str | None) -> list[str]:
    """The models to offer a tool, the default first; empty when the workspace serves none."""
    preferred = [model for model in CODING_MODELS if model in served]
    rest = [model for model in served if model not in preferred]
    offered = preferred + rest
    if wanted in offered:
        offered = [wanted] + [model for model in offered if model != wanted]
    return offered[:OFFERED_MODELS]


def _limit_line(client: httpx.Client, url: str, key: str) -> str | None:
    response = client.get(f"{url}{API_ROOT}/key/budget", headers=_auth(key))
    if response.status_code != httpx.codes.OK:
        return None
    for limit in response.json().get("limits", []):
        if limit.get("scope") == "api_token" and limit.get("limit_usd") is not None:
            return f"${limit['remaining_usd']:.2f} of ${limit['limit_usd']:.2f} left on this key"
    return None


@dataclass(frozen=True)
class TestResult:
    ok: bool
    model: str
    reply: str = ""
    cost_usd: float | None = None
    request_id: str | None = None
    error: str = ""


def _test_request(client: httpx.Client, url: str, key: str, model: str) -> TestResult:
    try:
        response = client.post(
            f"{url}{API_ROOT}/chat/completions",
            headers=_auth(key),
            json={"model": model, "messages": [{"role": "user", "content": TEST_MESSAGE}]},
            timeout=TEST_TIMEOUT_SEC,
        )
    except httpx.HTTPError as failure:
        return TestResult(ok=False, model=model, error=f"could not reach Otari ({type(failure).__name__})")
    request_id = response.headers.get("otari-request-id")
    if response.status_code != httpx.codes.OK:
        detail = ""
        with contextlib.suppress(ValueError):
            detail = str(response.json().get("detail", ""))
        return TestResult(
            ok=False, model=model, request_id=request_id, error=detail or f"Otari answered {response.status_code}"
        )
    reply = str(response.json()["choices"][0]["message"].get("content") or "").strip()
    cost: float | None = None
    if request_id:
        # The usage row is written after the response, so it can trail it by a moment.
        for _ in range(8):
            usage = client.get(f"{url}{API_ROOT}/usage/requests/{request_id}", headers=_auth(key))
            if usage.status_code == httpx.codes.OK and usage.json().get("cost_usd") is not None:
                cost = float(usage.json()["cost_usd"])
                break
            time.sleep(0.5)
    return TestResult(ok=True, model=model, reply=reply, cost_usd=cost, request_id=request_id)


# --------------------------------------------------------------------------- coding tools


@dataclass(frozen=True)
class Configured:
    """One coding tool's outcome: the files written for it and the command `otari <binary>` starts."""

    tool: str
    label: str
    binary: str
    files: list[Path]


@dataclass(frozen=True)
class Tool:
    id: str
    label: str
    binary: str
    guide: str


TOOLS: tuple[Tool, ...] = (
    Tool("claude-code", "Claude Code", "claude", f"{GUIDES}/claude-code"),
    Tool("codex", "Codex", "codex", f"{GUIDES}/codex"),
    Tool("opencode", "OpenCode", "opencode", f"{GUIDES}/opencode"),
)
# Tools Otari has a guide for and this command does not configure yet.
GUIDE_ONLY: tuple[Tool, ...] = (
    Tool("pi", "Pi", "pi", f"{GUIDES}/pi"),
    Tool("hermes", "Hermes", "hermes", f"{GUIDES}/hermes"),
)


def _claude_settings() -> Path:
    return _home() / ".claude" / "otari.json"


def _codex_profile() -> Path:
    return Path(os.environ.get("CODEX_HOME") or _home() / ".codex") / "otari.config.toml"


def _opencode_config() -> Path:
    return _state_dir() / "opencode.json"


def _configure_claude_code(url: str, store: KeyStore, models: list[str]) -> Configured:
    model = models[0]
    settings = _claude_settings()
    document = {
        # Claude Code runs this to get the key, so the key is not in this file.
        "apiKeyHelper": store.helper(),
        "env": {
            "ANTHROPIC_BASE_URL": f"{url}/api",
            "ANTHROPIC_MODEL": model,
            # Background and subagent calls ask for these aliases, which would
            # otherwise name Claude models this workspace may not serve.
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        },
        "modelPicker": {
            "replaceBuiltInOptions": True,
            "options": [{"model": name, "label": name} for name in models],
        },
    }
    _write_private(settings, json.dumps(document, indent=2) + "\n")
    return Configured("claude-code", "Claude Code", "claude", [settings])


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _configure_codex(url: str, store: KeyStore, models: list[str]) -> Configured:
    profile = _codex_profile()
    command, *arguments = store.helper_argv()
    lines = [
        "# Written by `otari setup`. Used only by `otari codex` (`codex --profile otari`).",
        f"model = {_toml_string(models[0])}",
        'model_provider = "otari"',
        "",
        "[model_providers.otari]",
        'name = "Otari"',
        f"base_url = {_toml_string(url + API_ROOT)}",
        'wire_api = "responses"',
        "",
        "# Codex runs this to get the key, so the key is not in this file.",
        "[model_providers.otari.auth]",
        f"command = {_toml_string(command)}",
        f"args = [{', '.join(_toml_string(argument) for argument in arguments)}]",
        "",
    ]
    _write_private(profile, "\n".join(lines))
    return Configured("codex", "Codex", "codex", [profile])


def _configure_opencode(url: str, store: KeyStore, models: list[str]) -> Configured:
    config = _opencode_config()
    document = {
        "$schema": "https://opencode.ai/config.json",
        "model": f"otari/{models[0]}",
        "provider": {
            "otari": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Otari",
                "options": {"baseURL": f"{url}{API_ROOT}", "apiKey": "{env:OTARI_API_KEY}"},
                # Without a model listed, opencode answers "Provider not found".
                "models": {name: {} for name in models},
            }
        },
    }
    _write_private(config, json.dumps(document, indent=2) + "\n")
    return Configured("opencode", "OpenCode", "opencode", [config])


_CONFIGURE = {
    "claude-code": _configure_claude_code,
    "codex": _configure_codex,
    "opencode": _configure_opencode,
}


def _installed(tools: tuple[Tool, ...]) -> list[Tool]:
    return [tool for tool in tools if shutil.which(tool.binary) is not None]


# --------------------------------------------------------------------------- manifest


def _read_manifest() -> dict[str, Any]:
    try:
        loaded = json.loads(_manifest_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_manifest(url: str, configured: list[Configured], model: str | None) -> None:
    files = sorted({str(path) for item in configured for path in item.files})
    _write_private(_manifest_path(), json.dumps({"url": url, "model": model, "files": files}, indent=2) + "\n")


# --------------------------------------------------------------------------- commands


def _resolve_url(url: str | None) -> str:
    resolved = url or os.environ.get("OTARI_URL") or _read_manifest().get("url")
    if not resolved:
        raise SetupError("Say which Otari to set up with --url, for example --url https://api.otari.ai.")
    return str(resolved).rstrip("/")


@click.command("setup")
@click.option("--url", help="The Otari API address. Also read from OTARI_URL.")
@click.option(
    "--tool",
    "tools",
    multiple=True,
    type=click.Choice([tool.id for tool in TOOLS]),
    help="Configure only this tool. Repeat for several. Default: every supported tool installed here.",
)
@click.option("--limit", default=DEFAULT_LIMIT, show_default=True, help="Spending limit in USD to suggest for the key.")
@click.option(
    "--expiration",
    default=DEFAULT_EXPIRATION,
    show_default=True,
    type=click.Choice(EXPIRATIONS),
    help="Key lifetime to suggest. You can change both on the approval page.",
)
@click.option("--model", help="The model the tools start on. Default: the first coding model your workspace serves.")
@click.option("--new-key", is_flag=True, help="Ask for a new key even if a working one is stored.")
@click.option("--no-browser", is_flag=True, help="Print the approval link without opening it.")
@click.option("--no-test", is_flag=True, help="Skip the test request.")
@click.option("--yes", is_flag=True, help="Accepted for scripts and agents. The command never asks a question.")
@click.option("--json", "as_json", is_flag=True, help="Print one JSON object when done. It never contains the key.")
def setup(
    url: str | None,
    tools: tuple[str, ...],
    limit: str,
    expiration: str,
    model: str | None,
    new_key: bool,
    no_browser: bool,
    no_test: bool,
    yes: bool,
    as_json: bool,
) -> None:
    """Connect this machine's coding tools to Otari with one browser approval.

    Creates a key with a spending limit you approve, stores it in your Keychain,
    writes a settings file for each coding tool found, and sends one test
    request. Your existing settings for those tools are left as they are.
    """
    del yes
    base = _resolve_url(url)
    store = KeyStore(base)

    def say(text: str = "") -> None:
        # Progress goes to stderr in JSON mode, so stdout stays one object. Flushed
        # at once: an agent's shell call shows nothing else while it waits.
        click.echo(text, err=as_json)
        (sys.stderr if as_json else sys.stdout).flush()

    with httpx.Client(timeout=30, follow_redirects=False) as client:
        try:
            client.get(f"{base}{API_ROOT}/models").status_code
        except httpx.HTTPError:
            raise SetupError(f"Could not reach Otari at {base}. Check the address and your connection.") from None

        say("Otari setup")
        say()
        key = None if new_key else store.load()
        reused = key is not None and _key_works(client, base, key)
        manage_url = ""
        if reused:
            say(f"Using the key already stored in {store.where}.")
        else:
            _require_cli_app(client, base)
            key, _scopes, manage_url = _approve(
                client, base, limit=limit, expiration=expiration, open_browser=not no_browser, say=say
            )
            store.save(key)
            say(f"Approved. The key ending in {key[-4:]} is stored in {store.where}.")
        assert key is not None  # noqa: S101
        limit_line = _limit_line(client, base, key)
        if limit_line:
            say(f"  {limit_line}")
        say()

        served = _models(client, base, key)
        offered = pick_models(served, model)
        if model and offered and offered[0] != model:
            say(f"Your workspace does not serve {model}, so the tools start on {offered[0]}.")
        selected = [tool for tool in TOOLS if tool.id in tools] if tools else _installed(TOOLS)
        configured: list[Configured] = []
        if not offered:
            say("Your workspace serves no models yet, so no tool was configured.")
        else:
            say("Coding tools")
            otari = _invocation()
            for tool in selected:
                item = _CONFIGURE[tool.id](base, store, offered)
                configured.append(item)
                say(f"  {tool.label:<12} ready. Start it with: {otari} {tool.binary}")
            for tool in _installed(GUIDE_ONLY):
                say(f"  {tool.label:<12} found, not set up by this command yet: {tool.guide}")
            if not selected:
                say("  No supported coding tool found on this machine.")
            elif configured:
                say("  Your own settings for these tools were not changed.")
                if otari != "otari":
                    say("  To shorten those commands, install otari: uv tool install otari-agent")
        _write_manifest(base, configured, offered[0] if offered else None)
        say()

        result: TestResult | None = None
        if offered and not no_test:
            say(f"Test request to {offered[0]}...")
            result = _test_request(client, base, key, offered[0])
            if result.ok:
                say(f'  It worked. The model replied: "{result.reply[:200]}"')
                details = []
                if result.cost_usd is not None:
                    details.append(f"cost ${result.cost_usd:.6f}")
                after = _limit_line(client, base, key)
                if after:
                    details.append(after)
                if details:
                    say("  " + ", ".join(details))
            else:
                say(f"  The test request failed: {result.error}")
            say()

    if manage_url:
        say(f"See and revoke this key at {manage_url}")
    if as_json:
        click.echo(
            json.dumps(
                {
                    "status": "ok" if result is None or result.ok else "test_failed",
                    "url": base,
                    "key_stored_in": store.where,
                    "key_reused": reused,
                    "limit": limit_line,
                    "model": offered[0] if offered else None,
                    "configured": [
                        {
                            "tool": item.tool,
                            "files": [str(path) for path in item.files],
                            "start": f"{_invocation()} {item.binary}",
                        }
                        for item in configured
                    ],
                    "test": None
                    if result is None
                    else {
                        "ok": result.ok,
                        "reply": result.reply,
                        "cost_usd": result.cost_usd,
                        "request_id": result.request_id,
                        "error": result.error,
                    },
                    "manage_url": manage_url or None,
                }
            )
        )
    if result is not None and not result.ok:
        raise click.exceptions.Exit(1)


def _invocation() -> str:
    """How this machine runs `otari`: by name when it is installed, through uvx when it is not."""
    found = shutil.which("otari")
    # Under uvx the command is on PATH only while this run lasts, from its own environment.
    installed = found is not None and Path(found).parent != Path(sys.prefix) / "bin"
    return "otari" if installed else "uvx --from otari-agent otari"


# Replaced in tests: the launch commands hand the process over to the coding tool.
_exec = os.execvpe


def _launch(tool: Tool, config: Path, arguments: list[str], *, needs_key_in_env: bool = False) -> None:
    """Start a coding tool on Otari with the settings `otari setup` wrote for it."""
    url = _read_manifest().get("url")
    key = KeyStore(str(url)).load() if url else None
    if not url or key is None or not config.exists():
        raise SetupError(f"{tool.label} is not set up for Otari on this machine. Run otari setup first.")
    if shutil.which(tool.binary) is None:
        raise SetupError(f"{tool.label} is not installed: no `{tool.binary}` command was found.")
    env = dict(os.environ)
    if needs_key_in_env:
        # Only for a tool that can read its key no other way; the others run a command for it.
        env["OTARI_API_KEY"] = key
        env["OPENCODE_CONFIG"] = str(config)
    _exec(tool.binary, [tool.binary, *arguments], env)


# Everything after the command name belongs to the tool, its own --help included.
_PASS_THROUGH = {"ignore_unknown_options": True, "allow_extra_args": True, "help_option_names": []}
_BY_ID = {tool.id: tool for tool in TOOLS}


@click.command("claude", context_settings=_PASS_THROUGH)
@click.argument("arguments", nargs=-1, type=click.UNPROCESSED)
def claude(arguments: tuple[str, ...]) -> None:
    """Start Claude Code on Otari. Arguments go to `claude` unchanged."""
    settings = _claude_settings()
    _launch(_BY_ID["claude-code"], settings, ["--settings", str(settings), *arguments])


@click.command("codex", context_settings=_PASS_THROUGH)
@click.argument("arguments", nargs=-1, type=click.UNPROCESSED)
def codex(arguments: tuple[str, ...]) -> None:
    """Start Codex on Otari. Arguments go to `codex` unchanged."""
    _launch(_BY_ID["codex"], _codex_profile(), ["--profile", "otari", *arguments])


@click.command("opencode", context_settings=_PASS_THROUGH)
@click.argument("arguments", nargs=-1, type=click.UNPROCESSED)
def opencode(arguments: tuple[str, ...]) -> None:
    """Start OpenCode on Otari. Arguments go to `opencode` unchanged."""
    _launch(_BY_ID["opencode"], _opencode_config(), list(arguments), needs_key_in_env=True)


@click.command("logout")
@click.option("--url", help="The Otari API address. Default: the one the last setup used.")
def logout(url: str | None) -> None:
    """Revoke the stored key and remove everything `otari setup` wrote."""
    base = _resolve_url(url)
    store = KeyStore(base)
    key = store.load()
    if key is not None:
        with contextlib.suppress(httpx.HTTPError), httpx.Client(timeout=30) as client:
            client.post(f"{base}{API_ROOT}/oauth/apps/revoke", data={"token": key})
        store.forget()
        click.echo("The key is revoked and removed from this machine.")
    for name in _read_manifest().get("files", []):
        with contextlib.suppress(FileNotFoundError):
            Path(name).unlink()
    with contextlib.suppress(FileNotFoundError):
        _manifest_path().unlink()
    for folder in (_state_dir() / "keys", _state_dir()):
        # Only when empty: anything else in there is not this command's to remove.
        with contextlib.suppress(OSError):
            folder.rmdir()
    click.echo("Removed the settings files otari setup wrote.")
