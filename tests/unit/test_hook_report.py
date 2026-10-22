"""`otari hook` reports the gates it checked to the agent's Otari server, without ever waiting on it."""

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import otari_agent.hook as hook_cli
import otari_agent.report as report
from otari_agent.domain.definitions import (
    MAX_DEFINITION_GATES,
    MAX_GATE_MESSAGE_LENGTH,
    GateDefinition,
    bounded,
    definition_digest,
)
from otari_agent.report import CheckReport, GateReport, ModelUsage

pytestmark = pytest.mark.usefixtures("isolated_home", "no_otari_env")

_EXPORTING = {
    "OTEL_TRACES_EXPORTER": "otlp",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:8000/otlp/",
    "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Bearer%20tk-secret,x-extra=1",
}
_GATE = GateReport(
    gate_id="no-changelog", gate_type="path", origin="repo", enforcement="required", outcome="fail", blocked=True
)
_CHECK = CheckReport(
    harness="claude-code",
    session_id="77192d4d-b969-44eb-bb05-e7ca84f6dfab",
    moment="pre_tool_use.edit_target",
    policy_digest="0123456789abcdef",
    started_ns=1_792_340_397_929_000_000,
    gates=(_GATE,),
    tool_call_id="toolu_01Y5eZnM7cScX5y8GyV7fT7q",
    tool_name="Edit",
)


def _attributes(span: dict[str, Any]) -> dict[str, Any]:
    return {item["key"]: next(iter(item["value"].values())) for item in span["attributes"]}


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def _queue(env: dict[str, str], outbox: Path) -> Path:
    queue = report.queue_for(env, outbox)
    assert queue is not None
    return queue


def _answering(*statuses: int) -> tuple[list[str], Any]:
    """A stand-in for httpx.post that answers each request with the next status, recording the URLs."""
    sent: list[str] = []
    answers = iter(statuses)

    def post(url: str, *, json: Any, headers: dict[str, str], timeout: float) -> _Response:
        sent.append(url)
        return _Response(next(answers))

    return sent, post


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, None),
        ({**_EXPORTING, "OTEL_TRACES_EXPORTER": "console"}, None),
        ({**_EXPORTING, "OTARI_HOOK_REPORT": "off"}, None),
        (_EXPORTING, "http://localhost:8000/otlp/v1/traces"),
        (
            {**_EXPORTING, "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "https://otari.ai/otlp/v1/traces"},
            "https://otari.ai/otlp/v1/traces",
        ),
    ],
)
def test_reports_go_where_the_harness_exports_its_traces(env: dict[str, str], expected: str | None) -> None:
    assert report.endpoint(env) == expected


def test_headers_are_read_in_the_otel_environment_format() -> None:
    assert report.headers(_EXPORTING) == {"Authorization": "Bearer tk-secret", "x-extra": "1"}


def test_otari_url_is_what_a_harness_that_strips_otel_variables_reports_to() -> None:
    env = {**_EXPORTING, "OTARI_URL": "http://localhost:8200/", "OTARI_API_KEY": "tk-own"}

    assert report.endpoint(env) == "http://localhost:8200/otlp/v1/traces"
    assert report.headers(env) == {"Authorization": "Bearer tk-own"}


def test_a_check_is_a_span_per_gate_on_the_agents_session_with_no_message() -> None:
    body = report.otlp_body(_CHECK, ended_ns=_CHECK.started_ns + 5_000_000)

    [resource] = body["resourceSpans"]
    [scope] = resource["scopeSpans"]
    assert scope["scope"]["name"] == "otari.hook"
    [span] = scope["spans"]
    assert _attributes(span) == {
        "session.id": _CHECK.session_id,
        "otari.harness": "claude-code",
        "otari.gate.moment": "pre_tool_use.edit_target",
        "otari.policy.digest": "0123456789abcdef",
        "gen_ai.tool.call.id": "toolu_01Y5eZnM7cScX5y8GyV7fT7q",
        "gen_ai.tool.name": "Edit",
        "otari.gate.id": "no-changelog",
        "otari.gate.type": "path",
        "otari.gate.origin": "repo",
        "otari.gate.enforcement": "required",
        "otari.gate.outcome": "fail",
        "otari.gate.blocked": True,
    }


def test_a_gate_spans_its_own_run_and_carries_what_its_model_calls_used() -> None:
    judged = GateReport(
        gate_id="writing-style",
        gate_type="judge",
        origin="repo",
        enforcement="advisory",
        outcome="pass",
        blocked=False,
        started_ns=_CHECK.started_ns + 1_000,
        ended_ns=_CHECK.started_ns + 40_000_000_000,
        usage=(
            ModelUsage(
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                input_tokens=9,
                output_tokens=111,
                cache_write_tokens=6_684,
            ),
        ),
    )
    body = report.otlp_body(
        CheckReport(**{**_CHECK.__dict__, "gates": (_GATE, judged)}), ended_ns=_CHECK.started_ns + 1
    )

    path, judge = body["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert (path["startTimeUnixNano"], path["endTimeUnixNano"]) == (str(_CHECK.started_ns), str(_CHECK.started_ns + 1))
    assert (judge["startTimeUnixNano"], judge["endTimeUnixNano"]) == (
        str(judged.started_ns),
        str(judged.ended_ns),
    )
    assert "gen_ai.request.model" not in _attributes(path)
    assert {
        key: value for key, value in _attributes(judge).items() if key.startswith("gen_ai.") and "tool" not in key
    } == {
        "gen_ai.provider.name": "anthropic",
        "gen_ai.request.model": "claude-haiku-4-5-20251001",
        "gen_ai.usage.input_tokens": "9",
        "gen_ai.usage.output_tokens": "111",
        "gen_ai.usage.cache_read_tokens": "0",
        "gen_ai.usage.cache_write_tokens": "6684",
    }


def test_each_further_model_a_judge_ran_on_goes_on_a_usage_span_under_its_gate() -> None:
    judged = GateReport(
        gate_id="writing-style",
        gate_type="judge",
        origin="repo",
        enforcement="advisory",
        outcome="pass",
        blocked=False,
        usage=(
            ModelUsage("anthropic", "claude-sonnet-4-5", 300, 20),
            ModelUsage("anthropic", "claude-haiku-4-5-20251001", 9, 111),
        ),
    )
    body = report.otlp_body(CheckReport(**{**_CHECK.__dict__, "gates": (judged,)}), ended_ns=_CHECK.started_ns + 1)

    gate, side = body["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert gate["name"] == "otari.gate" and side["name"] == report.USAGE_SPAN
    assert side["parentSpanId"] == gate["spanId"]
    assert _attributes(gate)["gen_ai.request.model"] == "claude-sonnet-4-5"
    attributes = _attributes(side)
    assert attributes["gen_ai.request.model"] == "claude-haiku-4-5-20251001"
    assert (attributes["gen_ai.usage.input_tokens"], attributes["gen_ai.usage.output_tokens"]) == ("9", "111")
    assert attributes["session.id"] == _CHECK.session_id and attributes["otari.gate.id"] == "writing-style"
    assert "otari.gate.outcome" not in attributes


def test_a_queued_report_holds_no_credential(tmp_path: Path) -> None:
    assert report.enqueue(_CHECK, _EXPORTING, outbox=tmp_path, spawn=False)

    [queued] = list(_queue(_EXPORTING, tmp_path).glob("*.json"))
    text = queued.read_text(encoding="utf-8")
    assert "tk-secret" not in text and "tk-secret" not in str(queued)
    assert json.loads(text)["url"] == "http://localhost:8000/otlp/v1/traces"


def test_nothing_is_queued_without_an_export_or_a_gate(tmp_path: Path) -> None:
    assert not report.enqueue(_CHECK, {}, outbox=tmp_path, spawn=False)
    assert not report.enqueue(CheckReport(**{**_CHECK.__dict__, "gates": ()}), _EXPORTING, outbox=tmp_path, spawn=False)
    assert not any(tmp_path.rglob("*.json"))


def test_sending_removes_what_was_taken_and_keeps_what_was_not(tmp_path: Path) -> None:
    for _ in range(3):
        report.enqueue(_CHECK, _EXPORTING, outbox=tmp_path, spawn=False)
    answers = iter([200, 400, 503])
    seen: list[dict[str, str]] = []

    def post(url: str, *, json: Any, headers: dict[str, str], timeout: float) -> _Response:
        seen.append(headers)
        return _Response(next(answers))

    queue = _queue(_EXPORTING, tmp_path)
    assert not report.send_pending(queue, _EXPORTING, post)

    assert len(list(queue.glob("*.json"))) == 1
    assert seen[0]["Authorization"] == "Bearer tk-secret"


def test_a_server_that_cannot_be_reached_keeps_every_report(tmp_path: Path) -> None:
    report.enqueue(_CHECK, _EXPORTING, outbox=tmp_path, spawn=False)

    def post(*args: Any, **kwargs: Any) -> _Response:
        raise OSError("connection refused")

    queue = _queue(_EXPORTING, tmp_path)
    assert not report.send_pending(queue, _EXPORTING, post)
    assert len(list(queue.glob("*.json"))) == 1


def test_the_outbox_stops_growing_at_its_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report, "_MAX_PENDING", 3)
    for _ in range(5):
        report.enqueue(_CHECK, _EXPORTING, outbox=tmp_path, spawn=False)

    assert len(list(_queue(_EXPORTING, tmp_path).glob("*.json"))) == 3


_GATES_YAML = (
    "schema_version: '1.0'\npolicy:\n  id: test\ngates:\n"
    "  - id: no-changelog\n"
    "    type: path\n"
    "    runs: [pre_tool_use.edit_target]\n"
    "    enforcement: required\n"
    "    forbidden: ['CHANGELOG.md']\n"
    "    message: CHANGELOG.md is generated.\n"
    "  - id: no-secrets\n"
    "    type: path\n"
    "    runs: [pre_tool_use.read_target]\n"
    "    enforcement: required\n"
    "    forbidden: ['.env']\n"
    "    message: m\n"
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    (tmp_path / hook_cli.GUARDRAIL_FILE).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / hook_cli.GUARDRAIL_FILE).write_text(_GATES_YAML, encoding="utf-8")
    return tmp_path


def _edit(repo: Path) -> dict[str, Any]:
    return {
        "hook_event_name": "PreToolUse",
        "session_id": "77192d4d-b969-44eb-bb05-e7ca84f6dfab",
        "tool_use_id": "toolu_01Y5eZnM7cScX5y8GyV7fT7q",
        "cwd": str(repo),
        "tool_name": "Edit",
        "tool_input": {"file_path": str(repo / "CHANGELOG.md")},
    }


def test_the_hook_reports_the_gates_that_applied_to_the_tool_call(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _EXPORTING.items():
        monkeypatch.setenv(key, value)
    spawned: list[Path] = []
    monkeypatch.setattr(report, "_spawn_sender", spawned.append)

    result = CliRunner().invoke(hook_cli.hook, [], input=json.dumps(_edit(repo)))

    assert result.exit_code == 2
    queue = report.queue_for(_EXPORTING)
    assert queue is not None
    [queued] = list(queue.glob("*.json"))
    [span] = json.loads(queued.read_text(encoding="utf-8"))["body"]["resourceSpans"][0]["scopeSpans"][0]["spans"]
    attributes = _attributes(span)
    assert (attributes["otari.gate.id"], attributes["otari.gate.outcome"], attributes["otari.gate.blocked"]) == (
        "no-changelog",
        "fail",
        True,
    )
    assert attributes["gen_ai.tool.call.id"] == "toolu_01Y5eZnM7cScX5y8GyV7fT7q"
    assert spawned == [queue]


def test_a_report_that_fails_never_changes_the_verdict(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _EXPORTING.items():
        monkeypatch.setenv(key, value)

    def broken(*args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("disk full")

    monkeypatch.setattr(hook_cli, "enqueue", broken)

    result = CliRunner().invoke(hook_cli.hook, [], input=json.dumps(_edit(repo)))

    assert result.exit_code == 2
    assert "CHANGELOG.md is generated." in result.output


def test_a_hook_with_no_export_configured_reports_nothing(repo: Path) -> None:
    result = CliRunner().invoke(hook_cli.hook, [], input=json.dumps(_edit(repo)))

    assert result.exit_code == 2
    assert not report.default_outbox().exists()


_DEFINITION = (
    GateDefinition(
        gate_id="no-changelog",
        gate_type="path",
        enforcement="required",
        origin="repo",
        runs=("pre_tool_use.edit_target",),
        message="CHANGELOG.md is generated.",
    ),
)
_OTARI = {"OTARI_URL": "http://localhost:8200", "OTARI_API_KEY": "tk-own"}


def test_a_guardrail_version_goes_once_and_ahead_of_its_checks(tmp_path: Path) -> None:
    outbox = tmp_path / "outbox"
    report.enqueue(_CHECK, _OTARI, definition=_DEFINITION, outbox=outbox, spawn=False)
    report.enqueue(_CHECK, _OTARI, definition=_DEFINITION, outbox=outbox, spawn=False)
    sent, post = _answering(*[200] * 4)
    queue = _queue(_OTARI, outbox)

    assert report.send_pending(queue, _OTARI, post)
    report.enqueue(_CHECK, _OTARI, definition=_DEFINITION, outbox=outbox, spawn=False)
    assert report.send_pending(queue, _OTARI, post)

    assert sent == [
        "http://localhost:8200/api/v1/hooks/guardrails",
        "http://localhost:8200/otlp/v1/traces",
        "http://localhost:8200/otlp/v1/traces",
        "http://localhost:8200/otlp/v1/traces",
    ]


def test_a_definition_goes_only_to_a_server_otari_url_names(tmp_path: Path) -> None:
    report.enqueue(_CHECK, _EXPORTING, definition=_DEFINITION, outbox=tmp_path, spawn=False)

    [queued] = list(_queue(_EXPORTING, tmp_path).glob("*.json"))
    assert json.loads(queued.read_text(encoding="utf-8"))["url"].endswith("/v1/traces")


def test_a_definitions_fingerprint_is_the_checks(tmp_path: Path) -> None:
    report.enqueue(_CHECK, _OTARI, definition=_DEFINITION, outbox=tmp_path, spawn=False)

    [definition] = list(_queue(_OTARI, tmp_path).glob("0-definition-*.json"))
    assert json.loads(definition.read_text(encoding="utf-8"))["body"]["digest"] == definition_digest(_DEFINITION)


@pytest.mark.parametrize("status", [401, 403, 404, 422])
def test_a_definition_the_server_refused_is_sent_again_later(tmp_path: Path, status: int) -> None:
    queue = _queue(_OTARI, tmp_path)
    report.enqueue(_CHECK, _OTARI, definition=_DEFINITION, outbox=tmp_path, spawn=False)
    _, post = _answering(status, 200)
    report.send_pending(queue, _OTARI, post)

    report.enqueue(_CHECK, _OTARI, definition=_DEFINITION, outbox=tmp_path, spawn=False)

    assert list(queue.glob("0-definition-*.json"))


def test_a_rejected_key_holds_the_queue_for_a_corrected_one(tmp_path: Path) -> None:
    queue = _queue(_OTARI, tmp_path)
    report.enqueue(_CHECK, _OTARI, outbox=tmp_path, spawn=False)
    sent, post = _answering(401)

    assert not report.send_pending(queue, _OTARI, post)

    assert len(sent) == 1 and len(list(queue.glob("*.json"))) == 1


def test_a_key_that_may_not_declare_guardrails_still_delivers_its_checks(tmp_path: Path) -> None:
    queue = _queue(_OTARI, tmp_path)
    report.enqueue(_CHECK, _OTARI, definition=_DEFINITION, outbox=tmp_path, spawn=False)
    sent, post = _answering(403, 200)

    assert report.send_pending(queue, _OTARI, post)

    assert sent == ["http://localhost:8200/api/v1/hooks/guardrails", "http://localhost:8200/otlp/v1/traces"]
    assert not list(queue.glob("*.json"))


def test_a_definition_refused_for_a_bad_key_holds_the_queue(tmp_path: Path) -> None:
    queue = _queue(_OTARI, tmp_path)
    report.enqueue(_CHECK, _OTARI, definition=_DEFINITION, outbox=tmp_path, spawn=False)
    sent, post = _answering(401)

    assert not report.send_pending(queue, _OTARI, post)

    assert len(sent) == 1 and len(list(queue.glob("*.json"))) == 2


def test_each_server_and_key_queue_apart_so_a_key_only_goes_where_it_was_meant(tmp_path: Path) -> None:
    staging = {"OTARI_URL": "https://staging.example", "OTARI_API_KEY": "tk-staging"}
    production = {"OTARI_URL": "https://otari.example", "OTARI_API_KEY": "tk-production"}
    report.enqueue(_CHECK, staging, outbox=tmp_path, spawn=False)
    report.enqueue(_CHECK, production, outbox=tmp_path, spawn=False)
    sent: list[tuple[str, str]] = []

    def post(url: str, *, json: Any, headers: dict[str, str], timeout: float) -> _Response:
        sent.append((url, headers["Authorization"]))
        return _Response(200)

    assert report.send_pending(_queue(production, tmp_path), production, post)

    assert sent == [("https://otari.example/otlp/v1/traces", "Bearer tk-production")]
    assert list(_queue(staging, tmp_path).glob("*.json"))


def test_an_entry_that_is_not_an_export_is_dropped_not_left_blocking(tmp_path: Path) -> None:
    queue = _queue(_OTARI, tmp_path)
    queue.mkdir(parents=True)
    (queue / "0-broken.json").write_text("null", encoding="utf-8")
    report.enqueue(_CHECK, _OTARI, outbox=tmp_path, spawn=False)
    sent, post = _answering(200)

    assert report.send_pending(queue, _OTARI, post)

    assert len(sent) == 1 and not list(queue.glob("*.json"))


def test_a_guardrail_too_large_for_the_server_is_not_sent(tmp_path: Path) -> None:
    gates = tuple(
        GateDefinition(
            gate_id=f"g{index}", gate_type="path", enforcement="required", origin="repo", runs=(), message=""
        )
        for index in range(MAX_DEFINITION_GATES + 1)
    )

    report.enqueue(_CHECK, _OTARI, definition=gates, outbox=tmp_path, spawn=False)

    assert not list(_queue(_OTARI, tmp_path).glob("0-definition-*.json"))


def test_a_long_gate_message_is_cut_before_it_is_fingerprinted() -> None:
    [gate] = bounded([GateDefinition("g", "path", "required", "repo", (), "x" * (MAX_GATE_MESSAGE_LENGTH + 10))])

    assert len(gate.message) == MAX_GATE_MESSAGE_LENGTH


def test_the_sender_runs_isolated_from_the_agents_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr("subprocess.Popen", lambda argv, **options: started.append((argv, options)))

    report._spawn_sender(tmp_path)

    [(argv, options)] = started
    assert argv[1] == "-I" and options["cwd"] == tmp_path


def test_a_sender_that_cannot_start_never_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise ValueError("no interpreter")

    monkeypatch.setattr("subprocess.Popen", refuse)

    report._spawn_sender(tmp_path)


def test_a_codex_check_reports_on_its_thread(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _OTARI.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(report, "_spawn_sender", lambda queue: None)
    payload = {
        **_edit(repo),
        "tool_name": "apply_patch",
        "tool_input": {"command": "*** Begin Patch\n*** Update File: CHANGELOG.md\n@@\n-a\n+b\n*** End Patch\n"},
    }

    CliRunner().invoke(hook_cli.hook, ["--harness", "codex"], input=json.dumps(payload))

    queue = report.queue_for(_OTARI)
    assert queue is not None
    [queued] = [path for path in queue.glob("*.json") if not path.name.startswith("0-definition")]
    [span] = json.loads(queued.read_text(encoding="utf-8"))["body"]["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert _attributes(span)["otari.harness"] == "codex"


def test_reporting_leaves_an_advisory_answer_on_stdout_alone(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Claude Code reads a hook's stdout as JSON; a report must add nothing to it."""
    for key, value in _OTARI.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(report, "_spawn_sender", lambda queue: None)
    (repo / hook_cli.GUARDRAIL_FILE).write_text(_GATES_YAML.replace("required", "advisory"), encoding="utf-8")

    result = CliRunner().invoke(hook_cli.hook, [], input=json.dumps(_edit(repo)))

    assert result.exit_code == 0
    assert "systemMessage" in json.loads(result.stdout)


_STOP_GATES_YAML = (
    "schema_version: '1.0'\n"
    "policy:\n  id: test\n"
    "gates:\n"
    "  - id: no-changelog\n"
    "    type: path\n"
    "    runs: [stop.working_tree]\n"
    "    enforcement: required\n"
    "    forbidden: ['CHANGELOG.md']\n"
    "    message: CHANGELOG.md is generated.\n"
    "  - id: follows-pattern\n"
    "    type: judge\n"
    "    runs: [stop.session]\n"
    "    enforcement: advisory\n"
    "    rubric: Does this change follow the repository's conventions?\n"
    "    message: Does not follow the pattern.\n"
)
_JUDGE_TAKES_S = 0.2


def test_a_stop_check_times_its_judge_apart_and_reports_what_the_judge_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / hook_cli.GUARDRAIL_FILE).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / hook_cli.GUARDRAIL_FILE).write_text(_STOP_GATES_YAML, encoding="utf-8")
    for key, value in _EXPORTING.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(report, "_spawn_sender", lambda queue: None)
    envelope = {
        "type": "result",
        "result": json.dumps({"outcome": "pass", "reasoning": "fine"}),
        "modelUsage": {
            "claude-haiku-4-5-20251001": {
                "inputTokens": 9,
                "outputTokens": 111,
                "cacheReadInputTokens": 0,
                "cacheCreationInputTokens": 6_684,
            }
        },
    }

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "git":
            changed = " M README.md\n" if cmd[1] == "status" else "+ changed\n" if cmd[1] == "diff" else ""
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=changed, stderr="")
        time.sleep(_JUDGE_TAKES_S)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=json.dumps(envelope), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)
    payload = {"hook_event_name": "Stop", "session_id": "77192d4d-b969-44eb-bb05-e7ca84f6dfab", "cwd": str(tmp_path)}

    result = CliRunner().invoke(hook_cli.hook, [], input=json.dumps(payload))

    assert result.exit_code == 0, result.output
    queue = report.queue_for(_EXPORTING)
    assert queue is not None
    [queued] = list(queue.glob("*.json"))
    spans = json.loads(queued.read_text(encoding="utf-8"))["body"]["resourceSpans"][0]["scopeSpans"][0]["spans"]
    by_gate = {_attributes(span)["otari.gate.id"]: span for span in spans}

    def took_s(span: dict[str, Any]) -> float:
        return (int(span["endTimeUnixNano"]) - int(span["startTimeUnixNano"])) / 1e9

    assert took_s(by_gate["follows-pattern"]) >= _JUDGE_TAKES_S
    assert took_s(by_gate["no-changelog"]) < _JUDGE_TAKES_S
    assert _attributes(by_gate["follows-pattern"])["gen_ai.usage.cache_write_tokens"] == "6684"
    assert "gen_ai.request.model" not in _attributes(by_gate["no-changelog"])
