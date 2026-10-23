"""Report each gate `otari hook` checked to an Otari server, as OTLP spans on the agent's session.

Reports go to the Otari server ``OTARI_URL`` names, with ``OTARI_API_KEY``, the
variables the rest of the ``otari`` command reads. They land on the agent's
session by ``session.id``. Without them, a harness's own OpenTelemetry trace
export (``OTEL_TRACES_EXPORTER=otlp``) is followed, endpoint and headers both;
Claude Code passes no ``OTEL_*`` variable to its hooks, so it needs the first.
Nothing configured reports nothing, and ``OTARI_HOOK_REPORT=off`` turns it off.

A hook runs on every tool call, so a report never waits on the network. Spans
are written to an outbox under ``~/.otari/`` and a detached process sends them.
The outbox holds span bodies and the endpoint only: the headers, which carry the
credential, are read from the environment at send time and never written down.

A span carries ids and outcomes, never a gate's message or the evidence it read.
It spans the time its gate took, and a judge gate's span carries the tokens its
model calls used, so the server can price them.
What each gate is and says goes once per version of the guardrail, as its
definition, to the server ``OTARI_URL`` names; a check names that version by
its fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.parse
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from otari_agent import __version__
from otari_agent.domain.definitions import GateDefinition, definition_body

SCOPE = "otari.hook"
# A judge's calls on a model other than its gate span's: priced, never read as a gate.
USAGE_SPAN = "otari.gate.usage"
# Past these an outbox stops growing: a server that never answers must not fill the disk.
_MAX_PENDING = 1_000
_MAX_AGE_S = 24 * 3600
_SEND_TIMEOUT_S = 5.0
# A lock file older than this is a dead sender's; and how often a sender looks again after releasing it.
_STALE_LOCK_S = 120
_SENDER_ROUNDS = 3


@dataclass(frozen=True)
class ModelUsage:
    """The tokens a judge gate's model calls used, on the model that answered."""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True)
class GateReport:
    gate_id: str
    gate_type: str
    origin: str
    enforcement: str
    outcome: str
    blocked: bool
    # When the gate ran; without them it is reported as taking the whole check.
    started_ns: int | None = None
    ended_ns: int | None = None
    # One per model a judge's calls ran on, the most used first: its gate span carries that one.
    usage: tuple[ModelUsage, ...] = ()


@dataclass(frozen=True)
class CheckReport:
    """One `otari hook` invocation: the session and tool call it checked, and each gate that applied."""

    harness: str
    session_id: str
    moment: str
    policy_digest: str
    started_ns: int
    gates: tuple[GateReport, ...]
    tool_call_id: str | None = None
    tool_name: str | None = None


def endpoint(env: Mapping[str, str]) -> str | None:
    """The OTLP/HTTP traces URL to report to, or None when nothing names one."""
    if env.get("OTARI_HOOK_REPORT", "").strip().lower() in ("off", "0", "false"):
        return None
    server = env.get("OTARI_URL", "").strip()
    if server:
        return f"{server.rstrip('/')}/otlp/v1/traces"
    exporters = {name.strip() for name in env.get("OTEL_TRACES_EXPORTER", "").split(",")}
    if "otlp" not in exporters:
        return None
    traces = env.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "").strip()
    if traces:
        return traces
    base = env.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    return f"{base.rstrip('/')}/v1/traces" if base else None


def headers(env: Mapping[str, str]) -> dict[str, str]:
    """The headers for the endpoint :func:`endpoint` chose.

    OTel's are in its environment format: comma-separated, percent-encoded ``key=value`` pairs.
    """
    if env.get("OTARI_URL", "").strip():
        key = env.get("OTARI_API_KEY", "").strip()
        return {"Authorization": f"Bearer {key}"} if key else {}
    raw = env.get("OTEL_EXPORTER_OTLP_TRACES_HEADERS") or env.get("OTEL_EXPORTER_OTLP_HEADERS") or ""
    parsed: dict[str, str] = {}
    for pair in raw.split(","):
        key, sep, value = pair.partition("=")
        if sep and key.strip():
            parsed[urllib.parse.unquote(key.strip())] = urllib.parse.unquote(value.strip())
    return parsed


def definitions_url(env: Mapping[str, str]) -> str | None:
    """Where a guardrail's definition goes: only an Otari server named by ``OTARI_URL`` takes one."""
    if endpoint(env) is None:
        return None
    server = env.get("OTARI_URL", "").strip()
    return f"{server.rstrip('/')}/api/v1/hooks/guardrails" if server else None


def _attribute(key: str, value: str | bool | int) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    return {"key": key, "value": {"stringValue": value}}


def _usage_attributes(usage: ModelUsage | None) -> list[dict[str, Any]]:
    """A judge's model calls, in the GenAI names a server prices imported usage by."""
    if usage is None:
        return []
    return [
        _attribute("gen_ai.provider.name", usage.provider),
        _attribute("gen_ai.request.model", usage.model),
        _attribute("gen_ai.usage.input_tokens", usage.input_tokens),
        _attribute("gen_ai.usage.output_tokens", usage.output_tokens),
        _attribute("gen_ai.usage.cache_read_tokens", usage.cache_read_tokens),
        _attribute("gen_ai.usage.cache_write_tokens", usage.cache_write_tokens),
    ]


def otlp_body(report: CheckReport, *, ended_ns: int) -> dict[str, Any]:
    """The OTLP/JSON export of one check: a span per gate, all on the agent's session.

    A judge whose calls ran on more than one model reports each further model on
    a child span of its gate, so each model's tokens are priced at its own rate.
    """
    trace_id = secrets.token_hex(16)
    shared = [
        _attribute("session.id", report.session_id),
        _attribute("otari.harness", report.harness),
        _attribute("otari.gate.moment", report.moment),
        _attribute("otari.policy.digest", report.policy_digest),
    ]
    if report.tool_call_id:
        shared.append(_attribute("gen_ai.tool.call.id", report.tool_call_id))
    if report.tool_name:
        shared.append(_attribute("gen_ai.tool.name", report.tool_name))
    spans: list[dict[str, Any]] = []
    for gate in report.gates:
        span_id = secrets.token_hex(8)
        window = {
            "kind": 1,
            "startTimeUnixNano": str(gate.started_ns or report.started_ns),
            "endTimeUnixNano": str(gate.ended_ns or ended_ns),
        }
        main, *others = gate.usage or (None,)
        spans.append(
            {
                "traceId": trace_id,
                "spanId": span_id,
                "name": "otari.gate",
                **window,
                "attributes": [
                    *shared,
                    _attribute("otari.gate.id", gate.gate_id),
                    _attribute("otari.gate.type", gate.gate_type),
                    _attribute("otari.gate.origin", gate.origin),
                    _attribute("otari.gate.enforcement", gate.enforcement),
                    _attribute("otari.gate.outcome", gate.outcome),
                    _attribute("otari.gate.blocked", gate.blocked),
                    *_usage_attributes(main),
                ],
            }
        )
        spans.extend(
            {
                "traceId": trace_id,
                "spanId": secrets.token_hex(8),
                "parentSpanId": span_id,
                "name": USAGE_SPAN,
                **window,
                "attributes": [*shared, _attribute("otari.gate.id", gate.gate_id), *_usage_attributes(usage)],
            }
            for usage in others
        )
    resource = [_attribute("service.name", "otari-hook"), _attribute("service.version", __version__)]
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": resource},
                "scopeSpans": [{"scope": {"name": SCOPE, "version": __version__}, "spans": spans}],
            }
        ]
    }


def default_outbox() -> Path:
    return Path.home() / ".otari" / "outbox"


def _identity(env: Mapping[str, str]) -> str | None:
    """Which server and credential a report goes out under, as a name to queue it by.

    A hash, so the queue's path never holds the credential. Every session on the
    machine shares one outbox, and each queues under its own identity, so a sender
    only ever sends what was queued with the headers it was started with.
    """
    url = endpoint(env)
    if url is None:
        return None
    material = json.dumps([url, sorted(headers(env).items())])
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def queue_for(env: Mapping[str, str], outbox: Path | None = None) -> Path | None:
    """The directory this environment's reports queue in, or None when it reports nothing."""
    identity = _identity(env)
    return None if identity is None else (outbox or default_outbox()) / identity


def _settled_marker(queue: Path, digest: str) -> Path:
    """The file that says this server took this version of the guardrail under this credential."""
    return queue.parent.parent / "reported" / queue.name / digest


def _write_entry(queue: Path, name: str, entry: Mapping[str, Any]) -> None:
    # Unique per writer: two hooks queueing the same name must not race on one partial file.
    partial = queue / f".{name}.{secrets.token_hex(4)}"
    partial.write_text(json.dumps(entry), encoding="utf-8")
    partial.replace(queue / name)


def enqueue(
    report: CheckReport,
    env: Mapping[str, str],
    *,
    definition: Iterable[GateDefinition] = (),
    outbox: Path | None = None,
    spawn: bool = True,
) -> bool:
    """Queue one check for sending and start a sender; never raises, never waits on the network.

    The guardrail's definition is queued with it until this server has taken that
    version; it goes ahead of the check, so a check never names a version its
    server cannot explain. False when the check was not queued: reporting is off,
    the check had no gate to report, or the outbox could not be written.
    """
    queue = queue_for(env, outbox)
    url = endpoint(env)
    if queue is None or url is None or not report.gates:
        return False
    try:
        queue.mkdir(parents=True, exist_ok=True, mode=0o700)
        _trim(queue)
    except OSError:
        return False
    definitions = definitions_url(env)
    if definitions is not None:
        try:
            body = definition_body(definition)
            if body["gates"] and not _settled_marker(queue, body["digest"]).exists():
                # Named for its version, so every check before it is delivered queues it once, not again.
                _write_entry(
                    queue,
                    f"0-definition-{body['digest']}.json",
                    {"url": definitions, "body": body, "kind": "definition"},
                )
        except OSError:
            pass
    try:
        _write_entry(
            queue,
            f"{time.time_ns()}-{secrets.token_hex(4)}.json",
            {"url": url, "body": otlp_body(report, ended_ns=time.time_ns()), "kind": "spans"},
        )
    except OSError:
        return False
    if spawn:
        _spawn_sender(queue)
    return True


def _pending(queue: Path) -> list[Path]:
    return sorted(path for path in queue.glob("*.json") if not path.name.startswith("."))


def _trim(queue: Path) -> None:
    """Drop what is too old to matter, including partial files a killed hook left, then the oldest past the cap."""
    cutoff = time.time() - _MAX_AGE_S
    for path in queue.iterdir():
        try:
            if path.name != ".lock" and path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            continue
    pending = _pending(queue)
    for path in pending[: max(0, len(pending) - _MAX_PENDING + 1)]:
        path.unlink(missing_ok=True)


def _spawn_sender(queue: Path) -> None:
    """Start a sender that outlives the hook.

    Isolated (``-I``) and run from the queue, so nothing in the agent's working
    directory can stand in for a module the sender imports while it holds the credential.
    """
    detach: dict[str, Any] = (
        {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    try:
        subprocess.Popen(  # noqa: S603 - this interpreter, this module, no shell
            [sys.executable, "-I", "-m", "otari_agent.report", str(queue)],
            cwd=queue,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            **detach,
        )
    except Exception:  # noqa: BLE001, S110 - a sender that cannot start leaves the report for the next one
        pass


def send_pending(queue: Path, env: Mapping[str, str], post: Any = None) -> bool:
    """Send every queued export, oldest first; False when the server could not take one yet.

    Delivered exports are removed. One the server could not take yet stays, and
    sending stops there for the next sender: a network failure, 408, 429, a 5xx,
    or a 401/403, which a corrected key fixes. Any other refusal drops that one
    export and goes on, and so does a 403 on a guardrail version: a key the server
    takes spans from may still not be allowed to declare guardrails, and holding
    the queue there would hold every check behind it. A guardrail version counts
    as taken only when the server answers 2xx. An entry that is not an export is dropped.
    """
    if post is None:
        import httpx

        post = httpx.post
    request_headers = {"Content-Type": "application/json", **headers(env)}
    for path in _pending(queue):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            url, body = entry["url"], entry["body"]
        except (OSError, ValueError, KeyError, TypeError):
            path.unlink(missing_ok=True)
            continue
        try:
            response = post(url, json=body, headers=request_headers, timeout=_SEND_TIMEOUT_S)
        except Exception:  # noqa: BLE001 - any transport failure leaves the export queued
            return False
        status = response.status_code
        definition = entry.get("kind") == "definition"
        if status in (401, 408, 429) or status >= 500 or (status == 403 and not definition):
            return False
        if 200 <= status < 300 and definition:
            marker = _settled_marker(queue, str(body.get("digest", "")))
            try:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
            except OSError:
                pass
        path.unlink(missing_ok=True)
    return True


def _lock(queue: Path) -> Any:
    """Take the queue's sender lock without waiting, or return None when another sender holds it."""
    try:
        import fcntl
    except ImportError:
        # No flock (Windows): a lock file, taken over once it is old enough to be a dead sender's.
        path = queue / ".lock"
        try:
            if time.time() - path.stat().st_mtime > _STALE_LOCK_S:
                path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except OSError:
            return None
        return path
    handle = open(queue / ".lock", "w")  # noqa: SIM115 - held until _unlock
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def _unlock(held: Any) -> None:
    if isinstance(held, Path):
        held.unlink(missing_ok=True)
    else:
        held.close()


def _send_locked(queue: Path) -> None:
    """Send under an exclusive lock, so concurrent hooks start one sender, not one each.

    A hook that queued while the lock was held started a sender that gave up, so
    once the lock is released the queue is looked at again.
    """
    for _ in range(_SENDER_ROUNDS):
        held = _lock(queue)
        if held is None:
            return
        try:
            delivered = True
            while delivered and _pending(queue):
                delivered = send_pending(queue, os.environ)
        finally:
            _unlock(held)
        if not delivered or not _pending(queue):
            return


if __name__ == "__main__":
    _send_locked(Path(sys.argv[1]) if len(sys.argv) > 1 else default_outbox())
