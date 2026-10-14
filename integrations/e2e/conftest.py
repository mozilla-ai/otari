"""Boot a real Otari in front of the scripted mock provider, once per session.

Otari runs as a subprocess from the repository's own environment, the way a
self-hoster runs it, on a throwaway SQLite database. The frameworks under test
are installed only in this directory's environment, so nothing they pull in
reaches the gateway's lock.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mock_provider
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
MASTER_KEY = "framework-e2e-master-key"
HEALTH_TIMEOUT_SECONDS = 90
# Set to an OpenAI key to serve the snippets from the real OpenAI API instead of the mock.
LIVE_OPENAI_KEY = os.environ.get("OTARI_E2E_LIVE_OPENAI_KEY")


@dataclass(frozen=True)
class Gateway:
    """A running Otari and the API key the snippets authenticate with."""

    base_url: str
    api_key: str
    provider: mock_provider.MockProvider
    log_path: Path
    live: bool

    def env(self) -> dict[str, str]:
        """The environment a docs snippet runs in: what the pages tell a reader to export."""
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OTARI_", "OPENAI_"))}
        env["OTARI_BASE_URL"] = f"{self.base_url}/api/v1"
        env["OTARI_API_KEY"] = self.api_key
        return env

    def tail(self, lines: int = 60) -> str:
        """Return the end of the gateway log, which is where the reason for a failure is."""
        if not self.log_path.exists():
            return ""
        return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])


def _free_port() -> int:
    """Return a port that is free on localhost now, for the gateway to bind."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(method: str, url: str, payload: dict[str, Any] | None = None) -> Any:
    """Call the gateway's admin API with the master key and return the decoded JSON."""
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Otari-Key", MASTER_KEY)
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def _otari_command() -> list[str]:
    """``OTARI_E2E_EXECUTABLE`` if set, otherwise the repository's ``otari`` through uv."""
    if executable := os.environ.get("OTARI_E2E_EXECUTABLE"):
        return [executable]
    uv = shutil.which("uv")
    if uv is None:
        pytest.fail("uv is not on PATH, so the gateway cannot be started from the repository")
    return [uv, "run", "--project", str(REPO_ROOT), "--frozen", "--no-dev", "otari"]


@pytest.fixture(scope="session")
def gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    """Serve Otari for the session, in front of the mock provider or, in live mode, OpenAI."""
    workdir = tmp_path_factory.mktemp("otari")
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = workdir / "gateway.log"

    with mock_provider.running() as provider:
        config = {
            "database_url": f"sqlite:///{workdir / 'otari.db'}",
            "host": "127.0.0.1",
            "port": port,
            "master_key": MASTER_KEY,
            # The mock's models have no rates, and pricing is fail-closed by default.
            "require_pricing": False,
            "providers": {
                "openai": {"provider_type": "openai", "api_key": LIVE_OPENAI_KEY}
                if LIVE_OPENAI_KEY
                else {"provider_type": "openai", "api_base": provider.base_url, "api_key": "mock-provider-key"},
            },
        }
        config_path = workdir / "config.yml"
        config_path.write_text(json.dumps(config))
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("OTARI_") and k not in {"DATABASE_URL", "VIRTUAL_ENV", "PYTHONPATH"}
        }
        # The gateway's own tools (alembic among them) must resolve in its environment, not this one.
        own_bin = str(Path(sys.executable).parent)
        env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep) if p != own_bin)
        env["OTARI_SECRET_KEY"] = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()  # a Fernet key
        env["OTARI_PROVIDER_ACCOUNT_PEPPER"] = secrets.token_urlsafe(32)

        # From the repository root: `otari migrate` runs alembic, which reads ./alembic.ini.
        command = _otari_command()
        with log_path.open("wb") as log:
            subprocess.run(
                [*command, "migrate", "--config", str(config_path)],
                env=env,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=log,
                check=True,
            )
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                [*command, "serve", "--config", str(config_path)], env=env, cwd=REPO_ROOT, stdout=log, stderr=log
            )
        try:
            deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
            while True:
                if process.poll() is not None:
                    pytest.fail(f"Otari exited with {process.returncode}:\n{log_path.read_text(errors='replace')}")
                try:
                    _request("GET", f"{base_url}/api/v1/health")
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        pytest.fail(f"Otari was not healthy within {HEALTH_TIMEOUT_SECONDS}s")
                    time.sleep(0.5)

            _request("POST", f"{base_url}/api/v1/users", {"user_id": "framework-e2e", "alias": "Framework e2e"})
            key = _request("POST", f"{base_url}/api/v1/keys", {"key_name": "framework-e2e", "user_id": "framework-e2e"})
            yield Gateway(
                base_url=base_url, api_key=key["key"], provider=provider, log_path=log_path, live=bool(LIVE_OPENAI_KEY)
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=20)
