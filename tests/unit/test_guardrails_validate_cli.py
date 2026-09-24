"""Unit tests for `otari guardrails validate`, the offline check over a guardrail file.

Offline is the contract, so these run with nothing stubbed: no gateway, no
credential, no model CLI, no subprocess. The only thing the command touches
beyond the guardrail file itself is the verifier script a verifier gate
names, and that it stats rather than runs.
"""

from pathlib import Path

import pytest
from click.testing import CliRunner, Result

import otari_agent.hook as hook_cli

_HEADER = 'schema_version: "1.0"\npolicy:\n  id: demo/guardrails\ngates:\n'

_CLEAN = _HEADER + (
    "  - id: no-hand-edited-changelog\n"
    "    type: path\n"
    "    runs: [pre_tool_use.edit_target, stop.working_tree]\n"
    "    enforcement: required\n"
    '    forbidden: ["CHANGELOG.md"]\n'
    "    message: CHANGELOG.md is generated at release time.\n"
    "  - id: use-pnpm-not-npm\n"
    "    type: command\n"
    "    runs: [pre_tool_use.command]\n"
    "    enforcement: required\n"
    '    forbidden: ["npm install", "npm ci"]\n'
    "    message: web/ is pnpm, not npm.\n"
    "  - id: openapi-changed-needs-postman\n"
    "    type: command_if_changed\n"
    "    runs: [stop.session]\n"
    "    enforcement: required\n"
    '    when_changed: ["docs/public/openapi.json"]\n'
    '    require: ["make postman"]\n'
    "    message: Regenerate the Postman collection.\n"
    "  - id: no-narrative-comments\n"
    "    type: judge\n"
    "    runs: [stop.session]\n"
    "    enforcement: advisory\n"
    '    when_changed: ["src/**"]\n'
    "    rubric: Does this diff add a comment that restates the code?\n"
    "    message: Narration belongs in the commit message.\n"
)

_READS = _HEADER + (
    "  - id: no-secret-reads\n"
    "    type: path\n"
    "    runs: [pre_tool_use.read_target]\n"
    "    enforcement: required\n"
    '    forbidden: [".env", "**/.env"]\n'
    "    message: Secrets stay out of the transcript; grep for the one key you need.\n"
    "  - id: no-hand-edited-changelog\n"
    "    type: path\n"
    "    runs: [pre_tool_use.edit_target, stop.working_tree]\n"
    "    enforcement: required\n"
    '    forbidden: ["CHANGELOG.md"]\n'
    "    message: CHANGELOG.md is generated at release time.\n"
)

_FOOTGUNS = _HEADER + (
    "  - id: no-hand-edited-claude-md\n"
    "    type: path\n"
    "    runs: [pre_tool_use.edit_target]\n"
    "    enforcement: required\n"
    '    forbidden: ["**/CLAUDE.md"]\n'
    "    message: Edit the AGENTS.md beside it.\n"
)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write(repo: Path, policy: str) -> None:
    (repo / ".otari-guardrails.yml").write_text(policy, encoding="utf-8")


def _invoke(*args: str) -> Result:
    return CliRunner().invoke(hook_cli.guardrails, ["validate", *args])


def test_fails_outside_a_git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = _invoke()
    assert result.exit_code != 0
    assert "Not inside a Git repository" in result.output


def test_says_where_to_start_when_there_is_no_guardrail_file(repo: Path) -> None:
    result = _invoke()
    assert result.exit_code != 0
    assert "otari guardrails generate" in result.output


def test_a_clean_guardrail_reports_nothing_and_exits_zero(repo: Path) -> None:
    _write(repo, _CLEAN)
    result = _invoke()
    assert result.exit_code == 0
    assert "demo/guardrails, 4 gate(s), schema 1.0" in result.output
    assert "0 error(s), 0 warning(s)" in result.output


def test_a_malformed_guardrail_is_a_readable_error(repo: Path) -> None:
    """The failure a runtime 422 reports mid-session, reported while the author is editing."""
    _write(repo, _HEADER + "  - id: g\n    type: nonsense\n    enforcement: required\n    message: m\n")
    result = _invoke()
    assert result.exit_code != 0
    assert "unsupported type 'nonsense'" in result.output


def test_warnings_alone_exit_zero_and_strict_makes_them_non_zero(repo: Path) -> None:
    _write(repo, _FOOTGUNS)
    result = _invoke()
    assert result.exit_code == 0
    assert "can never match 'CLAUDE.md'" in result.output
    assert "runs at pre_tool_use.edit_target with no stop.working_tree" in result.output
    assert "0 error(s), 2 warning(s)" in result.output
    assert _invoke("--strict").exit_code == 1


def test_a_missing_verifier_script_is_an_error(repo: Path) -> None:
    _write(
        repo,
        _HEADER + "  - id: g\n    type: verifier\n    runs: [stop.verifier]\n"
        "    enforcement: required\n    verifier: verifiers/nope.sh\n    message: m\n",
    )
    result = _invoke()
    assert result.exit_code == 1
    assert "verifier 'verifiers/nope.sh' does not exist" in result.output


def test_a_verifier_script_without_its_executable_bit_is_an_error(repo: Path) -> None:
    """The real run learns this from a failed exec and reports `error`, which blocks a required gate."""
    script = repo / "check.sh"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(0o644)
    _write(
        repo,
        _HEADER + "  - id: g\n    type: verifier\n    runs: [stop.verifier]\n"
        "    enforcement: required\n    verifier: check.sh\n    message: m\n",
    )
    result = _invoke()
    assert result.exit_code == 1
    assert "is not executable" in result.output


def test_a_verifier_climbing_out_of_the_repo_is_an_error(repo: Path) -> None:
    _write(
        repo,
        _HEADER + "  - id: g\n    type: verifier\n    runs: [stop.verifier]\n"
        "    enforcement: required\n    verifier: ../../escape.sh\n    message: m\n",
    )
    result = _invoke()
    assert result.exit_code == 1
    assert "resolves outside the repo root" in result.output


def test_an_executable_verifier_is_clean_and_is_never_run(repo: Path) -> None:
    script = repo / "check.sh"
    script.write_text("#!/bin/sh\ntouch " + str(repo / "ran") + "\nexit 1\n", encoding="utf-8")
    script.chmod(0o755)
    _write(
        repo,
        _HEADER + "  - id: g\n    type: verifier\n    runs: [stop.verifier]\n"
        "    enforcement: required\n    verifier: check.sh\n    message: m\n",
    )
    result = _invoke()
    assert result.exit_code == 0
    assert not (repo / "ran").exists()


def test_a_dry_run_command_reports_the_gate_that_refuses_it(repo: Path) -> None:
    _write(repo, _CLEAN)
    result = _invoke("--command", "npm install lodash")
    assert result.exit_code == 0
    assert "PreToolUse, Bash: npm install lodash" in result.output
    assert "fires      use-pnpm-not-npm (required)" in result.output


def test_a_dry_run_command_that_only_mentions_the_tool_stays_quiet(repo: Path) -> None:
    """The narrowed phrase list in action: the footgun that made it necessary, checked."""
    _write(repo, _CLEAN)
    result = _invoke("--command", "grep -rn npm web/")
    assert "quiet      use-pnpm-not-npm (required)" in result.output


def test_a_dry_run_path_is_reported_at_both_moments_it_could_be_read(repo: Path) -> None:
    _write(repo, _CLEAN)
    result = _invoke("--path", "CHANGELOG.md")
    assert "PreToolUse, Edit/Write: CHANGELOG.md" in result.output
    assert "Stop, the finished turn: 1 changed path(s), 0 command(s)" in result.output
    assert result.output.count("fires      no-hand-edited-changelog (required)") == 2


def test_a_guardrail_with_no_read_gate_prints_no_read_moment(repo: Path) -> None:
    """The output of a guardrail that cannot see a read is exactly what it was before reads existed."""
    _write(repo, _CLEAN)
    assert "PreToolUse, Read:" not in _invoke("--path", "CHANGELOG.md").output


def test_a_read_gate_adds_a_read_moment_to_the_dry_run(repo: Path) -> None:
    """A path is dry-run at every moment its own guardrail can actually see it."""
    _write(repo, _READS)
    result = _invoke("--path", ".env")
    assert "PreToolUse, Read: .env" in result.output
    assert "fires      no-secret-reads (required)" in result.output


def test_the_read_moment_leaves_a_write_only_gate_alone(repo: Path) -> None:
    """Same path, same guardrail, different moment: only the gate that asked sees it."""
    _write(repo, _READS)
    result = _invoke("--path", ".env")
    # Only this block, not everything after it: the Stop block below reports
    # the same gate again, on evidence it really does run at.
    read_block = result.output.split("PreToolUse, Read: .env")[1].split("Stop,")[0]
    assert "no-hand-edited-changelog" not in read_block
    assert "1 gate(s) do not apply here." in read_block


def test_a_dry_run_firing_a_gate_does_not_change_the_exit_status(repo: Path) -> None:
    """A gate firing is the answer to the question asked, not a failure of the guardrail."""
    _write(repo, _CLEAN)
    assert _invoke("--command", "npm install", "--strict").exit_code == 0


def test_the_stop_block_names_which_judge_gates_would_cost_a_model_call(repo: Path) -> None:
    _write(repo, _CLEAN)
    applies = _invoke("--path", "src/gateway/cli.py")
    assert "would run  no-narrative-comments (judge, advisory, one model call)" in applies.output
    skipped = _invoke("--path", "README.md")
    assert "skipped    no-narrative-comments (judge, when_changed does not match)" in skipped.output


def test_a_correlation_gate_is_satisfied_by_the_command_in_the_same_dry_run(repo: Path) -> None:
    _write(repo, _CLEAN)
    without = _invoke("--path", "docs/public/openapi.json")
    assert "fires      openapi-changed-needs-postman (required)" in without.output
    with_command = _invoke("--path", "docs/public/openapi.json", "--command", "make postman")
    assert "quiet      openapi-changed-needs-postman (required)" in with_command.output


def test_guardrail_file_points_at_another_file(repo: Path) -> None:
    (repo / "other.yml").write_text(_FOOTGUNS, encoding="utf-8")
    result = _invoke("--guardrail-file", "other.yml")
    assert result.exit_code == 0
    assert "other.yml: demo/guardrails, 1 gate(s)" in result.output


@pytest.mark.parametrize("spelling", ["CHANGELOG.md", "./CHANGELOG.md", "{repo}/CHANGELOG.md"])
def test_a_dry_run_path_matches_however_it_is_spelled(repo: Path, spelling: str) -> None:
    """A gate's globs are repo-relative POSIX, and the hook resolves to that before matching.

    Taking the string literally would report `quiet` for the absolute or
    `./`-prefixed spelling a person naturally types, which is the silent false
    clean this command exists to remove.
    """
    _write(repo, _CLEAN)
    result = _invoke("--path", spelling.format(repo=repo))
    assert "PreToolUse, Edit/Write: CHANGELOG.md" in result.output
    assert result.output.count("fires      no-hand-edited-changelog (required)") == 2


def test_a_dry_run_path_outside_the_repo_is_refused(repo: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    outside = tmp_path_factory.mktemp("elsewhere") / "CHANGELOG.md"
    _write(repo, _CLEAN)
    result = _invoke("--path", str(outside))
    assert result.exit_code != 0
    assert "is outside" in result.output


def _judge_gates(count: int) -> str:
    return "".join(
        f"  - id: judge-{index}\n    type: judge\n    runs: [stop.session]\n"
        "    enforcement: advisory\n    rubric: r\n    message: m\n"
        for index in range(count)
    )


def test_the_dry_run_applies_the_same_judge_cap_the_hook_applies(repo: Path) -> None:
    """A preview promising six model calls where the hook makes five is wrong about its own point."""
    _write(repo, _HEADER + _judge_gates(hook_cli._HOOK_JUDGE_MAX_GATES_PER_RUN + 1))
    result = _invoke("--path", "README.md")
    assert result.output.count("would run") == hook_cli._HOOK_JUDGE_MAX_GATES_PER_RUN
    assert f"judge-{hook_cli._HOOK_JUDGE_MAX_GATES_PER_RUN} (judge, past the " in result.output
    assert "-gate cap for one Stop event)" in result.output


def test_the_dry_run_cap_counts_only_applicable_gates(repo: Path) -> None:
    """The hook caps after when_changed filtering, so a scoped-out gate does not use up a slot."""
    scoped_out = (
        "  - id: judge-scoped\n    type: judge\n    runs: [stop.session]\n"
        '    enforcement: advisory\n    when_changed: ["nothing/here/**"]\n    rubric: r\n    message: m\n'
    )
    _write(repo, _HEADER + scoped_out + _judge_gates(hook_cli._HOOK_JUDGE_MAX_GATES_PER_RUN))
    result = _invoke("--path", "README.md")
    assert "skipped    judge-scoped (judge, when_changed does not match)" in result.output
    assert result.output.count("would run") == hook_cli._HOOK_JUDGE_MAX_GATES_PER_RUN
    assert "past the" not in result.output


def test_a_separator_in_a_require_phrase_is_reported_as_an_error(repo: Path) -> None:
    """A required gate nothing can satisfy, caught before it blocks a session."""
    _write(
        repo,
        _HEADER + "  - id: g\n    type: command_if_changed\n    runs: [stop.session]\n"
        '    enforcement: required\n    when_changed: ["x.json"]\n'
        '    require: ["make a && make b"]\n    message: m\n',
    )
    result = _invoke()
    assert result.exit_code == 1
    assert "contains the shell separator(s) '&&'" in result.output


def test_a_symlinked_path_keeps_its_own_name_at_stop(repo: Path) -> None:
    """`hook` resolves an edit target and does not resolve a Stop path, so neither does the preview.

    `git status` reports the tracked name, link and all. Resolving it for the
    Stop block reports a gate forbidding the link's own name as quiet, where a
    real session fails it.
    """
    (repo / "AGENTS.md").write_text("real\n", encoding="utf-8")
    (repo / "CLAUDE.md").symlink_to("AGENTS.md")
    _write(
        repo,
        _HEADER + "  - id: no-hand-edited-claude-md\n    type: path\n"
        "    runs: [pre_tool_use.edit_target, stop.working_tree]\n"
        '    enforcement: required\n    forbidden: ["CLAUDE.md"]\n    message: m\n',
    )
    result = _invoke("--path", "CLAUDE.md")
    # The edit tool would have written through the link, so PreToolUse sees the target.
    assert "PreToolUse, Edit/Write: AGENTS.md" in result.output
    # Git reports the link itself, so the Stop block must fire.
    assert "Stop, the finished turn: 1 changed path(s)" in result.output
    stop_block = result.output.split("Stop, the finished turn:")[1]
    assert "fires      no-hand-edited-claude-md (required)" in stop_block


def test_an_ordinary_path_still_reads_the_same_at_both_moments(repo: Path) -> None:
    _write(repo, _CLEAN)
    result = _invoke("--path", "CHANGELOG.md")
    assert result.output.count("fires      no-hand-edited-changelog (required)") == 2


def test_an_absolute_path_through_a_repo_alias_keeps_the_symlink_name(
    repo: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A repo reached through a directory symlink must not drag file resolution along.

    The lexical attempt fails (the alias is not the real prefix), so without a
    middle spelling that resolves the directory alone, the fully resolved
    fallback follows the file's own link too and the Stop block goes quiet
    again. This is what /tmp -> /private/tmp does to every macOS temp path.
    """
    (repo / "AGENTS.md").write_text("real\n", encoding="utf-8")
    (repo / "CLAUDE.md").symlink_to("AGENTS.md")
    _write(
        repo,
        _HEADER + "  - id: no-hand-edited-claude-md\n    type: path\n"
        "    runs: [pre_tool_use.edit_target, stop.working_tree]\n"
        '    enforcement: required\n    forbidden: ["CLAUDE.md"]\n    message: m\n',
    )
    alias = tmp_path_factory.mktemp("aliases") / "repo"
    alias.symlink_to(repo, target_is_directory=True)

    result = _invoke("--path", str(alias / "CLAUDE.md"))
    assert "PreToolUse, Edit/Write: AGENTS.md" in result.output
    stop_block = result.output.split("Stop, the finished turn:")[1]
    assert "fires      no-hand-edited-claude-md (required)" in stop_block


def test_a_path_resolving_out_of_the_repo_is_still_checked_by_its_own_name(
    repo: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A symlink escaping the repo does not take the gate with it.

    This test previously asserted the opposite, on the reasoning that `hook`
    evaluated nothing for such a target so a preview promising a refusal would
    be lying. That was true and was the bug: a guardrail forbidding `.env` did
    nothing when `.env` was a link to a shared secrets file, which is the
    ordinary layout for the very file such a rule is written about. `hook` now
    submits the lexical spelling beside the resolved one, so the link's own
    name is enforced and the preview says so.
    """
    outside = tmp_path_factory.mktemp("elsewhere") / "outside.md"
    outside.write_text("x\n", encoding="utf-8")
    (repo / "escape.md").symlink_to(outside)
    _write(
        repo,
        _HEADER + "  - id: no-escape-md\n    type: path\n"
        "    runs: [pre_tool_use.edit_target, stop.working_tree]\n"
        '    enforcement: required\n    forbidden: ["escape.md"]\n    message: m\n',
    )
    result = _invoke("--path", "escape.md")
    pre_block, stop_block = result.output.split("Stop, the finished turn:")
    assert "fires      no-escape-md (required)" in pre_block
    assert "not checked" not in pre_block
    # Git reports the link itself, so Stop covered this even before the fix.
    assert "fires      no-escape-md (required)" in stop_block
