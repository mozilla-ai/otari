#!/usr/bin/env python3
"""Boot Otari as the OSS edition and smoke it end to end.

ARCHITECTURE.md promises that every milestone ships a standalone OSS edition:
Otari must boot with only its own adapters bound, no overlay bootstrap and no
platform token, and still serve a real request. Without one automated check that
proves it, the promise is an assertion, and the first change that quietly makes
an OSS path depend on an enterprise adapter merges green and breaks self-hosting.

This is that check. It boots the packaged CLI as a subprocess (not an in-process
TestClient) so a failure to *start* counts as a failure, then walks the path a
self-hoster walks on day one:

1. ``/health``, ``/health/liveness``, ``/health/readiness`` answer.
2. Readiness reports no ``mode``, which is what standalone looks like: hybrid
   mode stamps ``mode: hybrid`` on both health payloads, so the absence is the
   assertion that no platform token selected the other edition behind our back.
3. ``/v1/bootstrap`` answers with no credential and reports ``standalone``. It is
   the first request a browser makes, and a second statement of which edition
   booted.
4. Create a user, then an API key for it (master-key admin surface).
5. Register a BYO provider credential at runtime, which is stored encrypted.
6. Create a routing policy whose default candidate fails and whose ``on_failure``
   candidate is that BYO provider.
7. Send one chat completion at the policy. It can only succeed by failing over to
   the BYO candidate and presenting the stored key to the provider, so a 200 with
   the expected body proves credential storage, routing, fallback, and dispatch
   all work in the OSS edition.
8. The usage row for that request is readable back through ``/v1/usage``.
9. Mail, which this deployment never configured, reports itself unavailable and
   names what would turn it on, and a send is refused rather than accepted and
   dropped. That is the state every self-hoster who wants no email is in, so it
   is the state the gate has to prove still boots and still says so honestly.

Both provider endpoints are a mock OpenAI-compatible server this script runs, so
the gate needs no provider secret and runs identically on a fork PR.

Standard library only, and no dev dependencies, on purpose: CI runs it against a
runtime-only environment (``uv sync --frozen --no-dev``), so an enterprise-only
or dev-only import that reached an OSS code path fails here rather than passing
on a developer machine that happens to have it installed.

Usage:
    uv run --frozen --no-dev python scripts/oss_edition_smoke.py
    uv run --frozen --no-dev python scripts/oss_edition_smoke.py --database-url postgresql://...
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal

MASTER_KEY = "oss-edition-smoke-master-key"
# One header carries both credentials the smoke presents: the master key on the
# admin calls, and the created API key on the completion.
KEY_HEADER = "Otari-Key"
MODEL = "oss-edition-smoke-model"
REPLY = "oss-edition-smoke-ok"

# Where the mock provider serves each behavior. The failing prefix is what the
# policy's default candidate points at, the working prefix what its on_failure
# candidate points at.
FAILING_PREFIX = "/failing"
WORKING_PREFIX = "/working"

# Everything the gateway reads from the environment is cleared before it starts,
# so the config file this script writes is the only thing that decides what boots.
# A developer's exported OTARI_MODE or OTARI_AI_TOKEN would otherwise select
# hybrid mode (GatewayConfig.effective_mode) and quietly turn the gate into a
# check of the other edition; OTARI_DATABASE_URL or the CLI's DATABASE_URL would
# point the run at a database nobody meant to smoke. OTARI_BOOTSTRAP, the overlay
# selector ARCHITECTURE.md describes, is covered by the same sweep on the day it
# exists.
SCRUBBED_ENV_PREFIXES = ("OTARI_",)
SCRUBBED_ENV_VARS = frozenset({"DATABASE_URL"})

HEALTH_TIMEOUT_SECONDS = 90
REQUEST_TIMEOUT_SECONDS = 60


class SmokeFailure(Exception):
    """A smoke step did not do what the OSS edition is supposed to do."""


@dataclass(frozen=True)
class Names:
    """The identifiers one run creates, all suffixed with a per-run id.

    Nothing here is cleaned up: CI runs against an empty database and throws it
    away. The suffix is what keeps a local re-run against a database that kept
    the last run's rows from failing on a 409 that says nothing about the OSS
    edition.
    """

    user_id: str
    policy: str
    failing_instance: str
    byo_instance: str
    byo_key: str

    @classmethod
    def for_run(cls, run_id: str) -> Names:
        # Provider instance and policy names may not contain ':' or '/', which a
        # hex run id cannot produce.
        return cls(
            user_id=f"oss-edition-smoke-user-{run_id}",
            policy=f"oss-edition-smoke-{run_id}",
            failing_instance=f"oss_smoke_failing_{run_id}",
            byo_instance=f"oss_smoke_byo_{run_id}",
            byo_key=f"oss-edition-smoke-byo-{secrets.token_hex(8)}",
        )


def log(message: str) -> None:
    """Print a progress line, unbuffered so it interleaves correctly in CI logs."""
    print(message, flush=True)


# --------------------------------------------------------------------------- #
# Mock provider
# --------------------------------------------------------------------------- #


class _MockProviderServer(ThreadingHTTPServer):
    """An OpenAI-compatible provider that fails on one path and answers on another.

    Records what it was asked for so the smoke can assert *how* a request was
    served: a completion that never touched the failing path was not routed
    through the fallback chain, and would pass a weaker assertion.
    """

    daemon_threads = True

    def __init__(self, address: tuple[str, int], expected_api_key: str) -> None:
        super().__init__(address, _MockProviderHandler)
        self.expected_api_key = expected_api_key
        self.failing_calls = 0
        self.working_calls = 0
        self.unauthorized_calls = 0
        self._lock = threading.Lock()

    def record(self, kind: Literal["failing", "working", "unauthorized"]) -> None:
        with self._lock:
            if kind == "failing":
                self.failing_calls += 1
            elif kind == "working":
                self.working_calls += 1
            else:
                self.unauthorized_calls += 1


class _MockProviderHandler(BaseHTTPRequestHandler):
    """Serve ``POST <prefix>/v1/chat/completions`` for the two mock instances."""

    protocol_version = "HTTP/1.1"
    # Narrowed from the base class's socketserver.BaseServer so the recorder and
    # the expected key below are typed.
    server: _MockProviderServer

    # do_POST, log_message: http.server dictates both spellings.
    def do_POST(self) -> None:  # noqa: N802
        # Drain the request body so the connection stays usable for keep-alive.
        self._read_body()
        if not self.path.endswith("/chat/completions"):
            self._respond(404, {"error": {"message": f"mock provider has no route {self.path}"}})
            return

        if self.path.startswith(FAILING_PREFIX):
            self.server.record("failing")
            # 500 rather than a 4xx: a caller-fault status is the one class the
            # gateway is right not to fail over on, so it would make the
            # fallback assertion below untestable.
            self._respond(500, {"error": {"message": "mock provider is deliberately down"}})
            return

        if not self.path.startswith(WORKING_PREFIX):
            self._respond(404, {"error": {"message": f"mock provider has no route {self.path}"}})
            return

        if self.headers.get("Authorization") != f"Bearer {self.server.expected_api_key}":
            # The gateway only has this key because it decrypted the credential
            # stored in step 4, so rejecting anything else is what makes the
            # completion prove the BYO credential was resolved and used.
            self.server.record("unauthorized")
            self._respond(401, {"error": {"message": "mock provider was not presented the stored BYO key"}})
            return

        self.server.record("working")
        self._respond(200, _completion_payload())

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Stay quiet: the gateway's own log is the interesting one on a failure."""

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _respond(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _completion_payload() -> dict[str, Any]:
    """A minimally complete chat completion, shaped as the OpenAI SDK expects."""
    return {
        "id": "chatcmpl-oss-edition-smoke",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": REPLY},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


@contextmanager
def mock_provider(expected_api_key: str) -> Iterator[_MockProviderServer]:
    """Run the mock provider on a free port for the duration of the block."""
    server = _MockProviderServer(("127.0.0.1", 0), expected_api_key)
    thread = threading.Thread(target=server.serve_forever, name="mock-provider", daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# --------------------------------------------------------------------------- #
# OSS-edition configuration and environment
# --------------------------------------------------------------------------- #


def oss_edition_env(base_env: dict[str, str], secret_key: str) -> dict[str, str]:
    """Return the environment the OSS edition boots in.

    Settings are dropped rather than overridden with a benign value: an override
    still leaves the variable present, and a future setting that reads "present"
    as "on" would flip the edition without changing this function.
    """
    env = {
        name: value
        for name, value in base_env.items()
        if name not in SCRUBBED_ENV_VARS and not name.startswith(SCRUBBED_ENV_PREFIXES)
    }
    # Provider credentials are encrypted at rest, so storing one (step 4) needs a
    # secret key. Generated per run and thrown away with the database.
    env["OTARI_SECRET_KEY"] = secret_key
    return env


def oss_edition_config(*, database_url: str, port: int, mock_base_url: str, names: Names) -> dict[str, Any]:
    """Return the config file contents for the OSS edition under smoke.

    No ``mode``, no ``platform`` block, and no provider anyone has to hold a key
    for: this is the config a self-hoster could write on day one.
    """
    return {
        "database_url": database_url,
        "host": "127.0.0.1",
        "port": port,
        "master_key": MASTER_KEY,
        # The mock's model has no real rates, and pricing is fail-closed by
        # default (402 for an unpriced model), which would reject the completion
        # for a reason that has nothing to do with the OSS boot this gate is
        # about. Pricing has its own tests.
        "require_pricing": False,
        "providers": {
            names.failing_instance: {
                "provider_type": "openai-compatible",
                "api_base": f"{mock_base_url}{FAILING_PREFIX}/v1",
                "api_key": "oss-edition-smoke-failing-instance-key",
            }
        },
    }


def write_config(path: Path, config: dict[str, Any]) -> None:
    """Write the config file.

    Serialized as JSON, which the loader's ``yaml.safe_load`` accepts (YAML is a
    superset), so nothing here has to hand-quote YAML.
    """
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #


def _request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    """Send one request and return its status and decoded body (None if not JSON)."""
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=body, method=method)
    request.add_header("Accept", "application/json")
    if body is not None:
        request.add_header("Content-Type", "application/json")
    for name, value in (headers or {}).items():
        request.add_header(name, value)

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return response.status, _decode(response.read())
    except urllib.error.HTTPError as error:
        return error.code, _decode(error.read())


def _decode(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _free_port() -> int:
    with closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    return port


# --------------------------------------------------------------------------- #
# Gateway process
# --------------------------------------------------------------------------- #


def _otari_executable() -> str:
    executable = shutil.which("otari")
    if executable is None:
        raise SmokeFailure(
            "The 'otari' CLI is not on PATH. Run this through the project environment, "
            "for example: uv run --frozen --no-dev python scripts/oss_edition_smoke.py"
        )
    return executable


@contextmanager
def gateway(config_path: Path, env: dict[str, str], base_url: str, log_path: Path) -> Iterator[None]:
    """Migrate the database, serve the OSS edition, and wait for it to be healthy."""
    otari = _otari_executable()

    log("Migrating the database")
    with log_path.open("wb") as log_file:
        migrate = subprocess.run(
            [otari, "migrate", "--config", str(config_path)],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if migrate.returncode != 0:
        raise SmokeFailure(f"'otari migrate' exited {migrate.returncode}")

    log("Starting the OSS edition")
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            [otari, "serve", "--config", str(config_path)],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    try:
        _await_health(process, base_url)
        yield
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=20)


def _await_health(process: subprocess.Popen[bytes], base_url: str) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SmokeFailure(f"The OSS edition exited with code {process.returncode} before becoming healthy")
        try:
            status, _ = _request("GET", f"{base_url}/health")
        except OSError:
            status = 0
        if status == 200:
            return
        time.sleep(0.5)
    raise SmokeFailure(f"The OSS edition did not answer /health within {HEALTH_TIMEOUT_SECONDS}s")


def _tail(log_path: Path, lines: int = 80) -> str:
    """Return the end of the gateway log, which is where the reason for a failure is."""
    if not log_path.exists():
        return "=== no gateway log ==="
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    return "=== gateway log (tail) ===\n" + "\n".join(tail)


# --------------------------------------------------------------------------- #
# Smoke steps
# --------------------------------------------------------------------------- #


def _expect(status: int, expected: int, what: str, body: Any) -> None:
    if status != expected:
        raise SmokeFailure(f"{what}: expected HTTP {expected}, got {status}: {body!r}")


def check_health(base_url: str) -> None:
    """Assert the three probes answer, and that this is the standalone edition."""
    status, body = _request("GET", f"{base_url}/health")
    _expect(status, 200, "GET /health", body)
    if not isinstance(body, dict) or body.get("status") != "healthy":
        raise SmokeFailure(f"GET /health did not report healthy: {body!r}")
    if "mode" in body:
        raise SmokeFailure(
            f"GET /health reports mode {body['mode']!r}. Only hybrid mode stamps a mode, so this "
            "process is not the OSS edition."
        )

    status, body = _request("GET", f"{base_url}/health/liveness")
    _expect(status, 200, "GET /health/liveness", body)

    status, body = _request("GET", f"{base_url}/health/readiness")
    _expect(status, 200, "GET /health/readiness", body)
    if not isinstance(body, dict) or body.get("database") != "connected":
        raise SmokeFailure(f"GET /health/readiness did not report a connected database: {body!r}")
    if "mode" in body:
        raise SmokeFailure(
            f"GET /health/readiness reports mode {body['mode']!r}, so this process is not the OSS edition."
        )
    log("Health probes answer, and the edition is standalone")


def check_bootstrap(base_url: str) -> None:
    """Assert the deployment bootstrap answers, uncredentialed, as standalone.

    This is the first request any browser makes, before it holds a credential, so
    it is checked without one. It is also a second, independent statement of
    which edition booted: an enterprise or platform-connected build answers
    something other than ``standalone`` here.
    """
    status, body = _request("GET", f"{base_url}/v1/bootstrap")
    _expect(status, 200, "GET /v1/bootstrap", body)
    if not isinstance(body, dict):
        raise SmokeFailure(f"GET /v1/bootstrap did not return an object: {body!r}")
    if body.get("deployment_type") != "standalone" or body.get("session_type") != "local_operator":
        raise SmokeFailure(f"GET /v1/bootstrap does not describe the OSS edition: {body!r}")
    if not body.get("surfaces"):
        raise SmokeFailure(f"GET /v1/bootstrap reports no management surfaces: {body!r}")
    log("The deployment bootstrap answers without a credential, as standalone")


def check_mail_is_honestly_unavailable(base_url: str, admin: dict[str, str]) -> None:
    """Assert a deployment with no mail says so, and refuses instead of dropping a send.

    Mail is optional (otari#648), and this smoke configures none, so this is the
    self-hoster's default state rather than an edge case. Two things have to
    hold in it: the surface reports what is missing, and a send is refused up
    front. A 200 here would mean the deployment accepted a message nobody would
    ever receive.
    """
    status, body = _request("GET", f"{base_url}/v1/settings/mail", headers=admin)
    _expect(status, 200, "GET /v1/settings/mail", body)
    if not isinstance(body, dict):
        raise SmokeFailure(f"GET /v1/settings/mail did not return an object: {body!r}")
    if body.get("transport") != "none" or body.get("ready") is not False:
        raise SmokeFailure(f"GET /v1/settings/mail reports mail on a deployment with none: {body!r}")
    if not body.get("missing"):
        raise SmokeFailure(f"GET /v1/settings/mail names nothing to configure: {body!r}")

    status, body = _request(
        "POST",
        f"{base_url}/v1/settings/mail/test",
        headers=admin,
        payload={"to": "smoke@example.com"},
    )
    if status != 503:
        raise SmokeFailure(f"POST /v1/settings/mail/test returned {status}, expected a 503 refusal: {body!r}")
    log("Mail is unconfigured, reports what is missing, and refuses a send rather than dropping it")


def create_key(base_url: str, admin: dict[str, str], names: Names) -> str:
    """Create a user and an API key for it, and return the raw key."""
    status, body = _request(
        "POST",
        f"{base_url}/v1/users",
        headers=admin,
        payload={"user_id": names.user_id, "alias": "OSS edition smoke"},
    )
    _expect(status, 200, "POST /v1/users", body)

    status, body = _request(
        "POST",
        f"{base_url}/v1/keys",
        headers=admin,
        payload={"key_name": "oss-edition-smoke", "user_id": names.user_id},
    )
    _expect(status, 200, "POST /v1/keys", body)
    key = body.get("key") if isinstance(body, dict) else None
    if not isinstance(key, str) or not key:
        raise SmokeFailure(f"POST /v1/keys returned no key: {body!r}")
    log("Created a user and an API key")
    return key


def register_byo_provider(base_url: str, admin: dict[str, str], names: Names, *, mock_base_url: str) -> None:
    """Store a BYO provider credential at runtime, the way an operator would."""
    status, body = _request(
        "POST",
        f"{base_url}/v1/provider-credentials",
        headers=admin,
        payload={
            "instance": names.byo_instance,
            "provider_type": "openai-compatible",
            "api_base": f"{mock_base_url}{WORKING_PREFIX}/v1",
            "api_key": names.byo_key,
        },
    )
    _expect(status, 201, "POST /v1/provider-credentials", body)
    log("Stored a BYO provider credential")


def create_fallback_policy(base_url: str, admin: dict[str, str], names: Names) -> None:
    """Create a policy whose default candidate is down and whose fallback is the BYO one."""
    status, body = _request(
        "POST",
        f"{base_url}/v1/routing/policies",
        headers=admin,
        payload={
            "name": names.policy,
            "spec": {
                "select": [{"default": f"{names.failing_instance}:{MODEL}"}],
                "on_failure": [f"{names.byo_instance}:{MODEL}"],
            },
        },
    )
    _expect(status, 200, "POST /v1/routing/policies", body)
    log("Created a routing policy with a fallback candidate")


def run_completion(base_url: str, key: str, provider: _MockProviderServer, names: Names) -> None:
    """Send one completion at the policy and assert it was served by the fallback."""
    status, body = _request(
        "POST",
        f"{base_url}/v1/chat/completions",
        headers={KEY_HEADER: key},
        payload={
            "model": names.policy,
            "messages": [{"role": "user", "content": "Is the OSS edition alive?"}],
        },
    )
    _expect(status, 200, "POST /v1/chat/completions", body)
    content = ""
    if isinstance(body, dict):
        choices = body.get("choices") or []
        if choices and isinstance(choices[0], dict):
            content = (choices[0].get("message") or {}).get("content") or ""
    if REPLY not in content:
        raise SmokeFailure(f"The completion did not come from the mock provider: {body!r}")

    if provider.unauthorized_calls:
        raise SmokeFailure(
            f"The mock provider rejected {provider.unauthorized_calls} call(s): the gateway did not "
            "present the stored BYO key."
        )
    if provider.failing_calls == 0:
        raise SmokeFailure(
            "The failing candidate was never called, so the completion was not fallback-routed and this "
            "step proves less than it claims."
        )
    if provider.working_calls != 1:
        raise SmokeFailure(f"Expected exactly one call to the fallback candidate, got {provider.working_calls}")
    log("A fallback-routed completion was served by the BYO provider")


def check_usage_recorded(base_url: str, admin: dict[str, str], names: Names) -> None:
    """Assert the completion was recorded, which is the OSS control plane's own job."""
    status, rows = _request("GET", f"{base_url}/v1/usage?limit=10", headers=admin)
    _expect(status, 200, "GET /v1/usage", rows)
    if not isinstance(rows, list) or not rows:
        raise SmokeFailure(f"GET /v1/usage recorded nothing for the completion: {rows!r}")
    served = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("user_id") == names.user_id
        and row.get("provider") == names.byo_instance
        and row.get("status") == "success"
        and row.get("total_tokens")
    ]
    if not served:
        raise SmokeFailure(
            f"GET /v1/usage has no successful row for {names.user_id} on the BYO provider "
            f"{names.byo_instance!r}: {rows!r}"
        )
    log("Usage for the completion is readable back")


def smoke(base_url: str, provider: _MockProviderServer, mock_base_url: str, names: Names) -> None:
    admin = {KEY_HEADER: MASTER_KEY}
    check_health(base_url)
    check_bootstrap(base_url)
    key = create_key(base_url, admin, names)
    register_byo_provider(base_url, admin, names, mock_base_url=mock_base_url)
    create_fallback_policy(base_url, admin, names)
    run_completion(base_url, key, provider, names)
    check_usage_recorded(base_url, admin, names)
    check_mail_is_honestly_unavailable(base_url, admin)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--database-url",
        default=None,
        help="Database for the run (default: a throwaway SQLite file). CI points this at PostgreSQL.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="otari-oss-smoke-") as workdir_name:
        workdir = Path(workdir_name)
        config_path = workdir / "oss-edition.yml"
        log_path = workdir / "gateway.log"
        database_url = args.database_url or f"sqlite:///{workdir / 'oss-edition.db'}"
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        names = Names.for_run(secrets.token_hex(4))
        # A Fernet key is urlsafe-base64 of 32 random bytes, which is why this
        # needs no cryptography import.
        secret_key = base64.urlsafe_b64encode(os.urandom(32)).decode()

        try:
            with mock_provider(names.byo_key) as provider:
                mock_base_url = f"http://127.0.0.1:{provider.server_address[1]}"
                write_config(
                    config_path,
                    oss_edition_config(
                        database_url=database_url,
                        port=port,
                        mock_base_url=mock_base_url,
                        names=names,
                    ),
                )
                env = oss_edition_env(dict(os.environ), secret_key)
                with gateway(config_path, env, base_url, log_path):
                    smoke(base_url, provider, mock_base_url, names)
        except SmokeFailure as failure:
            # Every failure prints the gateway's own log, not just the ones raised
            # while starting it: a step that fails against a running gateway is
            # exactly when its traceback is the thing worth reading.
            log(f"\nOSS-edition smoke FAILED: {failure}\n\n{_tail(log_path)}")
            return 1
        except Exception as error:
            log(f"\nOSS-edition smoke FAILED: {type(error).__name__}: {error}\n\n{_tail(log_path)}")
            return 1

    log("\nOSS-edition smoke passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
