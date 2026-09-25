"""Unit tests for judge-gate CLI backend selection: a gate's own `judge_cli`,
`--judge-cli`/`OTARI_HOOK_JUDGE_CLI`, and the invoking harness's own default,
in that precedence order (see `_hook_collect_judge_verdicts` in
`gateway.cli`). Complements `tests/unit/test_hook_cli.py` (which already
covers the `claude` backend's own call shape end to end) by covering the
selection mechanism itself and the `codex exec` backend's own call shape.
"""

import json
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

import otari_agent.hook as hook_cli
from otari_agent.domain.policy import parse_policy
from otari_agent.settings import HookSettings


def _guardrail_path(root: Path) -> Path:
    """`.otari/guardrails.yml` under `root`, with its parent directory created."""
    path = root / hook_cli.GUARDRAIL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._payload


@pytest.fixture(autouse=True)
def _judge_log_in_tmp_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(hook_cli, "_hook_judge_log_path", lambda: tmp_path / "judge-calls.log")


@pytest.fixture(autouse=True)
def _config_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hook_cli, "load_settings", lambda config_path=None: HookSettings(master_key="test-master-key"))


def _gates_yaml(judge_cli: str | None = None) -> str:
    judge_cli_line = f"    judge_cli: {judge_cli}\n" if judge_cli is not None else ""
    return (
        "schema_version: '1.0'\n"
        "policy:\n  id: test\n"
        "gates:\n"
        "  - id: g\n"
        "    type: judge\n"
        "    runs: [stop.session]\n"
        "    enforcement: advisory\n"
        "    rubric: r\n"
        f"{judge_cli_line}"
        "    message: m\n"
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    return tmp_path


def _git_status_and_diff_run() -> Callable[..., subprocess.CompletedProcess[str]]:
    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected subprocess.run call before the judge CLI itself: {cmd}")

    return fake_run


def _invoke(payload: dict[str, Any], *, harness: str = "claude-code", extra: list[str] | None = None) -> Any:
    args = ["--api-key", "test-key", "--harness", harness, *(extra or [])]
    return CliRunner().invoke(hook_cli.hook, args, input=json.dumps(payload))


def _capture_post(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    return captured


def test_claude_code_harness_defaults_to_the_claude_backend(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    _guardrail_path(repo).write_text(_gates_yaml(), encoding="utf-8")
    called_with: list[str] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        result = _git_status_and_diff_run()(cmd, **kwargs) if cmd[0] == "git" else None
        if result is not None:
            return result
        called_with.append(cmd[0])
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=json.dumps({"outcome": "pass", "reasoning": "ok"}), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}" if name in ("claude", "codex") else None)
    captured = _capture_post(monkeypatch)

    result = _invoke({"hook_event_name": "Stop", "cwd": str(repo)}, harness="claude-code")
    assert result.exit_code == 0, result.output
    assert called_with == ["/usr/bin/claude"]
    assert captured["json"]["judge_results"] == [{"gate_id": "g", "outcome": "pass", "reasoning": "ok"}]


def test_codex_harness_defaults_to_the_codex_backend(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    _guardrail_path(repo).write_text(_gates_yaml(), encoding="utf-8")
    called_with: list[str] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        result = _git_status_and_diff_run()(cmd, **kwargs) if cmd[0] == "git" else None
        if result is not None:
            return result
        called_with.append(cmd[0])
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=json.dumps({"outcome": "pass", "reasoning": "ok"}), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}" if name in ("claude", "codex") else None)
    captured = _capture_post(monkeypatch)

    result = _invoke({"hook_event_name": "Stop", "cwd": str(repo)}, harness="codex")
    assert result.exit_code == 0, result.output
    assert called_with == ["/usr/bin/codex"]
    assert captured["json"]["judge_results"] == [{"gate_id": "g", "outcome": "pass", "reasoning": "ok"}]


def test_codex_exec_is_invoked_read_only_and_non_interactive(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    _guardrail_path(repo).write_text(_gates_yaml(), encoding="utf-8")

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "git":
            return _git_status_and_diff_run()(cmd, **kwargs)
        assert cmd[0] == "/usr/bin/codex"
        assert cmd[1:3] == ["exec", "-"], "prompt goes over stdin, via the '-' pseudo-argument, not a trailing arg"
        assert "--model" not in cmd, (
            "no --judge-model given and no stable 'small codex model' to default to (see "
            "_HOOK_JUDGE_DEFAULT_MODEL's own comment): --model is left off, not guessed"
        )
        assert cmd[cmd.index("--sandbox") + 1] == "read-only"
        assert cmd[cmd.index("--ask-for-approval") + 1] == "never"
        assert "--skip-git-repo-check" in cmd, "the judge workdir is a plain directory, not a Git repo"
        assert "--ephemeral" in cmd, "a one-shot judge call must not leave a rollout file behind"
        assert "input" in kwargs, "the prompt is piped over stdin, matching claude -p's own choice"
        assert kwargs.get("cwd") == hook_cli._hook_judge_workdir()
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=json.dumps({"outcome": "fail", "reasoning": "no"}), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    captured = _capture_post(monkeypatch)

    result = _invoke({"hook_event_name": "Stop", "cwd": str(repo)}, harness="codex")
    assert result.exit_code == 0, result.output
    assert captured["json"]["judge_results"] == [{"gate_id": "g", "outcome": "fail", "reasoning": "no"}]


def test_claude_gets_the_haiku_default_model_with_no_override(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    _guardrail_path(repo).write_text(_gates_yaml(), encoding="utf-8")

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "git":
            return _git_status_and_diff_run()(cmd, **kwargs)
        assert cmd[cmd.index("--model") + 1] == hook_cli._HOOK_JUDGE_DEFAULT_MODEL
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=json.dumps({"outcome": "pass", "reasoning": "ok"}), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)
    _capture_post(monkeypatch)

    result = _invoke({"hook_event_name": "Stop", "cwd": str(repo)}, harness="claude-code")
    assert result.exit_code == 0, result.output


def test_judge_model_flag_overrides_the_default_for_the_codex_backend(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    _guardrail_path(repo).write_text(_gates_yaml(), encoding="utf-8")

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "git":
            return _git_status_and_diff_run()(cmd, **kwargs)
        assert cmd[cmd.index("--model") + 1] == "gpt-6-astra"
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=json.dumps({"outcome": "pass", "reasoning": "ok"}), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    _capture_post(monkeypatch)

    result = _invoke(
        {"hook_event_name": "Stop", "cwd": str(repo)}, harness="codex", extra=["--judge-model", "gpt-6-astra"]
    )
    assert result.exit_code == 0, result.output


def test_a_gates_own_judge_cli_overrides_the_harness_default(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """A gate authored to require codex gets codex even from a Claude Code hook."""
    _guardrail_path(repo).write_text(_gates_yaml(judge_cli="codex"), encoding="utf-8")
    called_with: list[str] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "git":
            return _git_status_and_diff_run()(cmd, **kwargs)
        called_with.append(cmd[0])
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=json.dumps({"outcome": "pass", "reasoning": "ok"}), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}" if name in ("claude", "codex") else None)
    _capture_post(monkeypatch)

    result = _invoke({"hook_event_name": "Stop", "cwd": str(repo)}, harness="claude-code")
    assert result.exit_code == 0, result.output
    assert called_with == ["/usr/bin/codex"]


def test_judge_cli_flag_overrides_the_harness_default_but_not_a_gates_own(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    _guardrail_path(repo).write_text(_gates_yaml(judge_cli="claude"), encoding="utf-8")
    called_with: list[str] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "git":
            return _git_status_and_diff_run()(cmd, **kwargs)
        called_with.append(cmd[0])
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=json.dumps({"outcome": "pass", "reasoning": "ok"}), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}" if name in ("claude", "codex") else None)
    _capture_post(monkeypatch)

    # --judge-cli codex would win over the claude-code harness default, but
    # the gate's own judge_cli: claude is more specific still and wins over both.
    result = _invoke(
        {"hook_event_name": "Stop", "cwd": str(repo)}, harness="claude-code", extra=["--judge-cli", "codex"]
    )
    assert result.exit_code == 0, result.output
    assert called_with == ["/usr/bin/claude"]


def test_judge_cli_flag_overrides_the_harness_default_when_the_gate_has_none(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    _guardrail_path(repo).write_text(_gates_yaml(), encoding="utf-8")
    called_with: list[str] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "git":
            return _git_status_and_diff_run()(cmd, **kwargs)
        called_with.append(cmd[0])
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=json.dumps({"outcome": "pass", "reasoning": "ok"}), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}" if name in ("claude", "codex") else None)
    _capture_post(monkeypatch)

    result = _invoke(
        {"hook_event_name": "Stop", "cwd": str(repo)}, harness="claude-code", extra=["--judge-cli", "codex"]
    )
    assert result.exit_code == 0, result.output
    assert called_with == ["/usr/bin/codex"]


def test_judge_cli_falls_back_to_the_next_candidate_when_the_first_is_missing(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    _guardrail_path(repo).write_text(_gates_yaml(judge_cli="[claude, codex]"), encoding="utf-8")
    called_with: list[str] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "git":
            return _git_status_and_diff_run()(cmd, **kwargs)
        called_with.append(cmd[0])
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=json.dumps({"outcome": "pass", "reasoning": "ok"}), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    # Only codex is on PATH: claude is the preferred first candidate, but not available.
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    _capture_post(monkeypatch)

    result = _invoke({"hook_event_name": "Stop", "cwd": str(repo)}, harness="claude-code")
    assert result.exit_code == 0, result.output
    assert called_with == ["/usr/bin/codex"]


def test_reports_error_naming_every_candidate_tried_when_none_are_on_path(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    _guardrail_path(repo).write_text(_gates_yaml(judge_cli="[claude, codex]"), encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", _git_status_and_diff_run())
    monkeypatch.setattr(shutil, "which", lambda name: None)
    captured = _capture_post(monkeypatch)

    result = _invoke({"hook_event_name": "Stop", "cwd": str(repo)}, harness="claude-code")
    assert result.exit_code == 0, result.output
    assert captured["json"]["judge_results"] == [
        {
            "gate_id": "g",
            "outcome": "error",
            "reasoning": "none of the configured judge CLI(s) were found on PATH: claude, codex",
        }
    ]


def test_dry_run_message_names_the_resolved_harness_default(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    _guardrail_path(repo).write_text(_gates_yaml(), encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", _git_status_and_diff_run())
    captured = _capture_post(monkeypatch)

    result = _invoke({"hook_event_name": "Stop", "cwd": str(repo)}, harness="codex", extra=["--judge-dry-run"])
    assert result.exit_code == 0, result.output
    assert "real codex call skipped" in captured["json"]["judge_results"][0]["reasoning"]


def test_judge_cli_flag_rejects_an_unsupported_backend(repo: Path) -> None:
    _guardrail_path(repo).write_text(_gates_yaml(), encoding="utf-8")
    result = _invoke({"hook_event_name": "Stop", "cwd": str(repo)}, extra=["--judge-cli", "gemini"])
    assert result.exit_code != 0
    assert "claude" in result.output and "codex" in result.output


def test_judge_gates_run_concurrently_not_sequentially(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """Five judge gates, each faked to take ~0.3s, must finish in well under
    5 * 0.3s: `_hook_collect_judge_verdicts` runs them through a
    `ThreadPoolExecutor` (`_HOOK_GATE_MAX_WORKERS`), not one after another.
    Also pins that results keep declaration order despite running out of
    order (`ThreadPoolExecutor.map` guarantees this; completion order would
    not).
    """
    gate_count = 5
    per_gate_seconds = 0.3
    gates_yaml = "schema_version: '1.0'\npolicy:\n  id: test\ngates:\n" + "".join(
        f"  - id: g{i}\n    type: judge\n"
        "    runs: [stop.session]\n    enforcement: advisory\n    rubric: r{i}\n    message: m{i}\n"
        for i in range(gate_count)
    )

    monkeypatch.setattr(hook_cli, "_hook_collect_diff", lambda repo_root: "diff")

    def fake_run_judge(
        rubric: str,
        diff: str,
        transcript: str,
        *,
        judge_cli: tuple[str, ...],
        model: str | None,
        deadline: float,
        dry_run: bool = False,
    ) -> tuple[str, str]:
        time.sleep(per_gate_seconds)
        return "pass", f"checked {rubric}"

    monkeypatch.setattr(hook_cli, "_hook_run_judge", fake_run_judge)

    start = time.monotonic()
    results = hook_cli._hook_collect_judge_verdicts(
        parse_policy(gates_yaml, source="test.yml"),
        repo,
        None,
        [],
        judge_model=None,
    )
    elapsed = time.monotonic() - start

    assert [result["gate_id"] for result in results] == [f"g{i}" for i in range(gate_count)]
    assert all(result["outcome"] == "pass" for result in results)
    # Sequential would take at least gate_count * per_gate_seconds (1.5s);
    # concurrent, with headroom to _HOOK_GATE_MAX_WORKERS >= gate_count,
    # should take close to one gate's own time. Generous slack for CI jitter,
    # still nowhere near the sequential floor.
    assert elapsed < gate_count * per_gate_seconds * 0.6, (
        f"took {elapsed:.2f}s for {gate_count} gates at {per_gate_seconds}s each -- looks sequential"
    )


_JUDGE_AND_DETERMINISTIC_YAML = (
    "schema_version: '1.0'\n"
    "policy:\n  id: test\n"
    "gates:\n"
    "  - id: g\n"
    "    type: judge\n"
    "    runs: [stop.session]\n"
    "    enforcement: advisory\n"
    "    rubric: r\n"
    "    message: m\n"
    "  - id: ran-the-tests\n"
    "    type: command_if_changed\n"
    "    runs: [stop.session]\n"
    "    enforcement: advisory\n"
    "    when_changed: ['src/**']\n"
    "    require: ['make test']\n"
    "    message: You changed src but never ran the tests.\n"
)


def _advisory_failures(monkeypatch: pytest.MonkeyPatch, results: list[dict[str, Any]]) -> None:
    """Answer the check with advisory failures only, so nothing blocks and the turn ends at exit 0."""
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse({"blocked": False, "results": results}))
    monkeypatch.setattr(subprocess, "run", _git_status_and_diff_run())
    monkeypatch.setattr(shutil, "which", lambda name: None)


def test_a_failing_judge_gate_reaches_the_model_as_additional_context(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    """A judge gate is advisory by construction, so exit 2 is a channel it can never take.

    `additionalContext` is the one Stop field that puts a finding in front of
    the model without blocking the turn. Without it the model's own review of
    the turn reaches everybody except the agent that could act on it, which is
    the reason a rubric is written in the first place. The reasoning travels
    too, not just the gate's fixed `message`: which line the judge objected to
    is the whole of what makes the finding actionable.
    """
    _guardrail_path(repo).write_text(_gates_yaml(), encoding="utf-8")
    _advisory_failures(
        monkeypatch,
        [
            {
                "gate_id": "g",
                "enforcement": "advisory",
                "outcome": "fail",
                "message": "m",
                "detail": "The endpoint at src/module.py:42 calls db.query() directly.",
            }
        ],
    )

    result = _invoke({"hook_event_name": "Stop", "cwd": str(repo)}, harness="claude-code")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    context = payload["hookSpecificOutput"]["additionalContext"]
    assert payload["hookSpecificOutput"]["hookEventName"] == "Stop"
    assert "src/module.py:42" in context
    assert "m" in context
    # The person keeps their own channel: this adds a reader rather than
    # moving the finding from one to the other.
    assert "src/module.py:42" in payload["systemMessage"]


def test_a_deterministic_advisory_gate_stays_out_of_additional_context(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    """An author who wrote `advisory` on a gate that could have said `required` chose the quiet channel.

    Only a judge gate is denied `required` by the parser, so only a judge gate
    is owed a channel it did not choose to give up.
    """
    _guardrail_path(repo).write_text(_JUDGE_AND_DETERMINISTIC_YAML, encoding="utf-8")
    _advisory_failures(
        monkeypatch,
        [
            {
                "gate_id": "ran-the-tests",
                "enforcement": "advisory",
                "outcome": "fail",
                "message": "You changed src but never ran the tests.",
            }
        ],
    )

    result = _invoke({"hook_event_name": "Stop", "cwd": str(repo)}, harness="claude-code")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert "hookSpecificOutput" not in payload
    assert "never ran the tests" in payload["systemMessage"]


def test_additional_context_carries_the_judge_finding_alone(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """Both kinds failing on one Stop must not collapse into one channel.

    The model is handed what it is owed and nothing else, so an advisory
    finding its author aimed at a person does not become an instruction to the
    agent by riding along beside a judge's.
    """
    _guardrail_path(repo).write_text(_JUDGE_AND_DETERMINISTIC_YAML, encoding="utf-8")
    _advisory_failures(
        monkeypatch,
        [
            {"gate_id": "g", "enforcement": "advisory", "outcome": "fail", "message": "judge says no"},
            {
                "gate_id": "ran-the-tests",
                "enforcement": "advisory",
                "outcome": "fail",
                "message": "You changed src but never ran the tests.",
            },
        ],
    )

    result = _invoke({"hook_event_name": "Stop", "cwd": str(repo)}, harness="claude-code")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    context = payload["hookSpecificOutput"]["additionalContext"]
    assert "judge says no" in context
    assert "never ran the tests" not in context
    # The person still gets both, which is what systemMessage was always for.
    assert "judge says no" in payload["systemMessage"]
    assert "never ran the tests" in payload["systemMessage"]


def test_a_judge_gate_that_could_not_run_is_not_reported_to_the_model(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    """`error` means the model call produced no verdict, so no rubric was evaluated.

    Passing it on as a judge finding would invent one, and the agent can do
    nothing about a judge that failed to run in any case. The person still
    hears it on `systemMessage`, which is where a guardrail that could not run
    has always been reported.
    """
    _guardrail_path(repo).write_text(_gates_yaml(), encoding="utf-8")
    _advisory_failures(
        monkeypatch,
        [
            {
                "gate_id": "g",
                "enforcement": "advisory",
                "outcome": "error",
                "message": "m",
                "detail": "claude exited 1: prompt is too long",
            }
        ],
    )

    result = _invoke({"hook_event_name": "Stop", "cwd": str(repo)}, harness="claude-code")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert "hookSpecificOutput" not in payload
    assert "prompt is too long" in payload["systemMessage"]
