"""Unit tests for `otari gates generate`, the AGENTS.md/CLAUDE.md -> .otari-gates.yml proposer.

Mocks the CLI boundary (shutil.which, subprocess.run) so these run with no
`claude`/`codex` installed and no real model call; the schema every accepted
gate must still pass is the real one, `otari_agent.domain.policy.parse_policy`,
not a stub.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner

import otari_agent.hook as hook_cli
from otari_agent.domain.policy import parse_policy

_EXISTING_GATES_WITH_COMMENT = (
    'schema_version: "1.0"\n'
    "policy:\n"
    "  id: demo/gates\n"
    "  description: Rules this repo checks on its own working tree.\n"
    "\n"
    "gates:\n"
    "  # a hand-written comment that must survive\n"
    "  - id: no-force-push\n"
    "    type: command\n"
    "    runs: [pre_tool_use.command]\n"
    "    enforcement: advisory\n"
    '    forbidden: ["git push --force"]\n'
    "    message: >-\n"
    "      Force-pushing rewrites shared history.\n"
)

_VALID_PROPOSAL = {
    "id": "no-hand-edited-changelog",
    "type": "path",
    "runs": ["pre_tool_use.edit_target", "stop.working_tree"],
    "enforcement": "required",
    "forbidden": ["CHANGELOG.md"],
    "message": "CHANGELOG.md is generated; do not hand-edit it.",
}

_PROPOSAL_WITHOUT_RUNS = {
    "id": "no-hand-edited-changelog",
    "type": "path",
    "enforcement": "required",
    "forbidden": ["CHANGELOG.md"],
    "message": "CHANGELOG.md is generated; do not hand-edit it.",
}

_INVALID_PROPOSAL = {
    "id": "bad-judge",
    "type": "judge",
    "runs": ["stop.session"],
    "enforcement": "required",  # judge gates may only be advisory
    "rubric": "some rubric",
    "message": "bad",
}


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("# Rules\nCHANGELOG.md is generated; never hand-edit it.\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _stub_claude_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/fake/bin/claude" if name == "claude" else None)


def _stub_cli_output(monkeypatch: pytest.MonkeyPatch, stdout: str, *, returncode: int = 0) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(args=argv, returncode=returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _stub_getchar(monkeypatch: pytest.MonkeyPatch, keys: str) -> None:
    """Make `click.getchar()` return one character of `keys` per call, in order, then "".

    Not CliRunner's own `input=`: `_gates_generate_read_choice` calls
    `click.getchar()` directly, which (see that function's own docstring)
    opens `/dev/tty` itself whenever `sys.stdin` is not a real terminal,
    bypassing CliRunner's stdin substitution entirely. Whether that happens
    to succeed depends on whether *this* process has a controlling terminal
    at all, which is true on a developer's machine but false on a typical CI
    runner; a test that instead fed keystrokes through CliRunner's `input=`
    would pass here and fail there. Stubbing `click.getchar` itself is what
    keeps these tests independent of that.
    """
    iterator = iter(keys)

    def fake_getchar(echo: bool = False) -> str:
        return next(iterator, "")

    monkeypatch.setattr(click, "getchar", fake_getchar)


def _invoke(monkeypatch: pytest.MonkeyPatch, *args: str, keys: str = "") -> Any:
    _stub_getchar(monkeypatch, keys)
    return CliRunner().invoke(hook_cli.gates, ["generate", *args])


def test_fails_outside_a_git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = _invoke(monkeypatch)
    assert result.exit_code != 0
    assert "Not inside a Git repository" in result.output


def test_fails_when_no_agents_or_claude_md_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    result = _invoke(monkeypatch)
    assert result.exit_code != 0
    assert "No AGENTS.md or CLAUDE.md found" in result.output


def test_falls_back_to_claude_md_when_agents_md_is_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "CLAUDE.md").write_text("some rules", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _stub_claude_only(monkeypatch)
    _stub_cli_output(monkeypatch, "[]")

    result = _invoke(monkeypatch)
    assert result.exit_code == 0, result.output
    assert "CLAUDE.md" in result.output


def test_explicit_source_overrides_the_default_search(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    other = repo / "docs" / "style.md"
    other.parent.mkdir()
    other.write_text("custom rules doc", encoding="utf-8")
    _stub_claude_only(monkeypatch)
    calls: list[str] = []

    def fake_run(argv: list[str], input: str = "", **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(input)
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = _invoke(monkeypatch, "--source", str(other))
    assert result.exit_code == 0, result.output
    assert "custom rules doc" in calls[0]


def test_fails_when_no_cli_is_found_on_path(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    result = _invoke(monkeypatch)
    assert result.exit_code != 0
    assert "was found on PATH" in result.output


def test_oversize_source_is_rejected_before_any_cli_call(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (repo / "AGENTS.md").write_text("x" * (hook_cli._GATES_GENERATE_MAX_SOURCE_BYTES + 1), encoding="utf-8")

    def fail_if_called(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("subprocess.run should not be called for an oversize source")

    monkeypatch.setattr(subprocess, "run", fail_if_called)

    result = _invoke(monkeypatch)
    assert result.exit_code != 0
    assert "larger than" in result.output


def test_an_unparseable_existing_gates_file_is_rejected_before_any_cli_call(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (repo / ".otari-gates.yml").write_text("not: valid: yaml: at: all:\n  - [", encoding="utf-8")

    def fail_if_called(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("subprocess.run should not be called when the existing policy fails to parse")

    monkeypatch.setattr(subprocess, "run", fail_if_called)

    result = _invoke(monkeypatch)
    assert result.exit_code != 0
    assert "does not currently parse" in result.output


def test_accepting_a_valid_proposal_writes_it_and_it_parses(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_claude_only(monkeypatch)
    _stub_cli_output(monkeypatch, json.dumps([_VALID_PROPOSAL]))

    result = _invoke(monkeypatch, keys="yy")
    assert result.exit_code == 0, result.output
    assert "Added 1 gate(s)" in result.output

    gates_file = repo / ".otari-gates.yml"
    text = gates_file.read_text(encoding="utf-8")
    spec = parse_policy(text, source=str(gates_file))
    assert [gate.id for gate in spec.gates] == ["no-hand-edited-changelog"]


def test_declining_a_valid_proposal_writes_nothing(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_claude_only(monkeypatch)
    _stub_cli_output(monkeypatch, json.dumps([_VALID_PROPOSAL]))

    result = _invoke(monkeypatch, keys="yn")
    assert result.exit_code == 0, result.output
    assert "Added 0 gate(s)" in result.output
    assert not (repo / ".otari-gates.yml").exists()


def test_skips_a_proposal_whose_id_already_exists_without_prompting(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (repo / ".otari-gates.yml").write_text(_EXISTING_GATES_WITH_COMMENT, encoding="utf-8")
    duplicate = {**_VALID_PROPOSAL, "id": "no-force-push"}
    _stub_claude_only(monkeypatch)
    _stub_cli_output(monkeypatch, json.dumps([duplicate]))

    result = _invoke(monkeypatch)  # no confirmation input needed: nothing should prompt
    assert result.exit_code == 0, result.output
    assert "Skipping 'no-force-push': already in .otari-gates.yml." in result.output
    assert "Added 0 gate(s)" in result.output


def test_appending_preserves_existing_comments_and_gates(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (repo / ".otari-gates.yml").write_text(_EXISTING_GATES_WITH_COMMENT, encoding="utf-8")
    _stub_claude_only(monkeypatch)
    _stub_cli_output(monkeypatch, json.dumps([_VALID_PROPOSAL]))

    result = _invoke(monkeypatch, keys="yy")
    assert result.exit_code == 0, result.output

    text = (repo / ".otari-gates.yml").read_text(encoding="utf-8")
    assert "a hand-written comment that must survive" in text
    spec = parse_policy(text, source="check")
    assert {gate.id for gate in spec.gates} == {"no-force-push", "no-hand-edited-changelog"}


def test_appending_inserts_before_a_later_top_level_key_not_at_eof(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`gates:` need not be the last top-level key (unusual, but schema_version/policy/gates
    carry no order requirement): the new gate must land inside the `gates:` sequence, before
    whatever top-level key follows it, not appended after that key at end of file.
    """
    (repo / ".otari-gates.yml").write_text(
        'schema_version: "1.0"\n'
        "gates:\n"
        "  - id: existing\n"
        "    type: path\n"
        "    runs: [pre_tool_use.edit_target, stop.working_tree]\n"
        "    enforcement: required\n"
        '    forbidden: ["x"]\n'
        "    message: m\n"
        "\n"
        "policy:\n"
        "  id: test\n"
        "  description: unusual order, gates before policy\n",
        encoding="utf-8",
    )
    _stub_claude_only(monkeypatch)
    _stub_cli_output(monkeypatch, json.dumps([_VALID_PROPOSAL]))

    result = _invoke(monkeypatch, keys="yy")
    assert result.exit_code == 0, result.output

    text = (repo / ".otari-gates.yml").read_text(encoding="utf-8")
    assert text.rstrip().endswith("description: unusual order, gates before policy")
    spec = parse_policy(text, source="check")
    assert {gate.id for gate in spec.gates} == {"existing", "no-hand-edited-changelog"}


def test_no_existing_gates_file_creates_one_named_after_the_repo_directory(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_claude_only(monkeypatch)
    _stub_cli_output(monkeypatch, json.dumps([_VALID_PROPOSAL]))

    result = _invoke(monkeypatch, keys="yy")
    assert result.exit_code == 0, result.output
    text = (repo / ".otari-gates.yml").read_text(encoding="utf-8")
    assert parse_policy(text, source="check").policy_id == f"{repo.name}/gates"


def test_gates_file_flag_creates_missing_parent_directories(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_claude_only(monkeypatch)
    _stub_cli_output(monkeypatch, json.dumps([_VALID_PROPOSAL]))
    target = repo / "nested" / "dir" / ".otari-gates.yml"

    result = _invoke(monkeypatch, "--gates-file", str(target), keys="yy")
    assert result.exit_code == 0, result.output
    assert target.is_file()
    spec = parse_policy(target.read_text(encoding="utf-8"), source=str(target))
    assert spec.gates[0].id == "no-hand-edited-changelog"


def test_invalid_proposal_declining_edit_is_skipped(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_claude_only(monkeypatch)
    _stub_cli_output(monkeypatch, json.dumps([_INVALID_PROPOSAL]))

    result = _invoke(monkeypatch, keys="yn")  # decline "Edit it and try again?"
    assert result.exit_code == 0, result.output
    assert "does not pass validation" in result.output
    assert "Added 0 gate(s)" in result.output
    assert not (repo / ".otari-gates.yml").exists()


def test_editing_an_invalid_proposal_can_fix_and_accept_it(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_claude_only(monkeypatch)
    _stub_cli_output(monkeypatch, json.dumps([_INVALID_PROPOSAL]))
    fixed_yaml = (
        "id: bad-judge\ntype: judge\nruns: [stop.session]\nenforcement: advisory\nrubric: some rubric\nmessage: bad\n"
    )
    monkeypatch.setattr(click, "edit", lambda text: fixed_yaml)

    # "y" (edit it) then "y" (accept the now-valid gate)
    result = _invoke(monkeypatch, keys="yyy")
    assert result.exit_code == 0, result.output
    assert "Added 1 gate(s)" in result.output
    text = (repo / ".otari-gates.yml").read_text(encoding="utf-8")
    spec = parse_policy(text, source="check")
    assert spec.gates[0].id == "bad-judge"
    assert spec.gates[0].enforcement == "advisory"


def test_edit_choice_on_a_valid_proposal_lets_you_change_it_before_accepting(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_claude_only(monkeypatch)
    _stub_cli_output(monkeypatch, json.dumps([_VALID_PROPOSAL]))
    renamed_yaml = (
        "id: renamed-gate\ntype: path\n"
        "runs: [pre_tool_use.edit_target, stop.working_tree]\n"
        'enforcement: required\nforbidden:\n- CHANGELOG.md\nmessage: "m"\n'
    )
    monkeypatch.setattr(click, "edit", lambda text: renamed_yaml)

    # "e" (edit) then "y" (accept the edited gate)
    result = _invoke(monkeypatch, keys="yey")
    assert result.exit_code == 0, result.output
    text = (repo / ".otari-gates.yml").read_text(encoding="utf-8")
    spec = parse_policy(text, source="check")
    assert spec.gates[0].id == "renamed-gate"


def test_a_declined_edit_keeps_the_previous_candidate(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_claude_only(monkeypatch)
    _stub_cli_output(monkeypatch, json.dumps([_VALID_PROPOSAL]))
    monkeypatch.setattr(click, "edit", lambda text: None)  # editor closed with no save

    result = _invoke(monkeypatch, keys="yey")
    assert result.exit_code == 0, result.output
    assert "No changes made." in result.output
    text = (repo / ".otari-gates.yml").read_text(encoding="utf-8")
    spec = parse_policy(text, source="check")
    assert spec.gates[0].id == "no-hand-edited-changelog"


def test_quit_stops_processing_further_proposals(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    second = {**_VALID_PROPOSAL, "id": "second-gate"}
    _stub_claude_only(monkeypatch)
    _stub_cli_output(monkeypatch, json.dumps([_VALID_PROPOSAL, second]))

    result = _invoke(monkeypatch, keys="yq")
    assert result.exit_code == 0, result.output
    assert "Stopped early. Added 0 gate(s)" in result.output
    assert not (repo / ".otari-gates.yml").exists()


def test_declining_the_cli_confirmation_aborts_without_calling_the_cli(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_claude_only(monkeypatch)

    def fail_if_called(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("subprocess.run should not be called once the CLI confirmation is declined")

    monkeypatch.setattr(subprocess, "run", fail_if_called)

    result = _invoke(monkeypatch, keys="n")
    assert result.exit_code != 0
    assert "Aborted" in result.output


def test_confirming_the_cli_names_it_before_using_it(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_claude_only(monkeypatch)
    _stub_cli_output(monkeypatch, "[]")

    result = _invoke(monkeypatch, keys="y")
    assert result.exit_code == 0, result.output
    assert "Use claude (/fake/bin/claude)" in result.output


def test_each_proposal_is_shown_with_its_own_progress_header(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    second = {**_VALID_PROPOSAL, "id": "second-gate"}
    _stub_claude_only(monkeypatch)
    _stub_cli_output(monkeypatch, json.dumps([_VALID_PROPOSAL, second]))

    result = _invoke(monkeypatch, keys="yyy")  # confirm CLI, accept first, accept second
    assert result.exit_code == 0, result.output
    assert "Gate proposal 1 of 2" in result.output
    assert "Gate proposal 2 of 2" in result.output
    assert "Added 2 gate(s)" in result.output


def test_model_output_wrapped_in_a_markdown_fence_still_parses(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_claude_only(monkeypatch)
    fenced = f"```json\n{json.dumps([_VALID_PROPOSAL])}\n```"
    _stub_cli_output(monkeypatch, fenced)

    result = _invoke(monkeypatch, keys="yy")
    assert result.exit_code == 0, result.output
    assert "Added 1 gate(s)" in result.output


def test_model_output_wrapped_in_a_gates_key_object_is_unwrapped(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_claude_only(monkeypatch)
    _stub_cli_output(monkeypatch, json.dumps({"gates": [_VALID_PROPOSAL]}))

    result = _invoke(monkeypatch, keys="yy")
    assert result.exit_code == 0, result.output
    assert "Added 1 gate(s)" in result.output


def test_non_json_model_output_raises_a_click_exception(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_claude_only(monkeypatch)
    _stub_cli_output(monkeypatch, "not json at all")

    result = _invoke(monkeypatch)
    assert result.exit_code != 0
    assert "did not return valid JSON" in result.output


def test_empty_proposal_list_reports_nothing_to_add(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_claude_only(monkeypatch)
    _stub_cli_output(monkeypatch, "[]")

    result = _invoke(monkeypatch)
    assert result.exit_code == 0, result.output
    assert "No gate proposals came back." in result.output


def test_nonzero_cli_exit_raises_a_click_exception(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_claude_only(monkeypatch)
    _stub_cli_output(monkeypatch, "boom", returncode=1)

    result = _invoke(monkeypatch)
    assert result.exit_code != 0
    assert "claude -p exited 1" in result.output


def test_claude_p_call_shape_runs_isolated_with_no_tools(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_claude_only(monkeypatch)
    calls = _stub_cli_output(monkeypatch, "[]")

    result = _invoke(monkeypatch)
    assert result.exit_code == 0, result.output
    assert calls == [["/fake/bin/claude", "--tools", "", "--strict-mcp-config", "-p"]]


def test_claude_p_call_shape_includes_model_when_given(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_claude_only(monkeypatch)
    calls = _stub_cli_output(monkeypatch, "[]")

    result = _invoke(monkeypatch, "--model", "claude-opus-5")
    assert result.exit_code == 0, result.output
    assert calls == [["/fake/bin/claude", "--model", "claude-opus-5", "--tools", "", "--strict-mcp-config", "-p"]]


def test_codex_exec_call_shape(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/fake/bin/codex" if name == "codex" else None)
    calls = _stub_cli_output(monkeypatch, "[]")

    result = _invoke(monkeypatch, "--cli", "codex")
    assert result.exit_code == 0, result.output
    assert calls == [
        [
            "/fake/bin/codex",
            "exec",
            "-",
            "--sandbox",
            "read-only",
            "--ask-for-approval",
            "never",
            "--skip-git-repo-check",
            "--ephemeral",
            "--color",
            "never",
        ]
    ]


def test_cli_option_overrides_the_default_preference_order(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: {"claude": "/fake/bin/claude", "codex": "/fake/bin/codex"}.get(name),
    )
    calls = _stub_cli_output(monkeypatch, "[]")

    result = _invoke(monkeypatch, "--cli", "codex,claude")
    assert result.exit_code == 0, result.output
    assert calls[0][0] == "/fake/bin/codex"


def test_runs_from_the_isolated_judge_workdir_not_the_repo(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_claude_only(monkeypatch)
    captured_cwd: list[object] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured_cwd.append(kwargs.get("cwd"))
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = _invoke(monkeypatch)
    assert result.exit_code == 0, result.output
    assert captured_cwd == [hook_cli._hook_judge_workdir()]


def test_describe_gate_does_not_dim_a_wrapped_values_own_continuation_lines() -> None:
    """A long `rubric`/`message` is one value PyYAML wraps across several physical lines;
    only its first line carries "key:" and should be colored, and a continuation line of
    that same value must read the same as the rest of it, not dimmed like a list item.
    """
    gate = {
        "id": "g",
        "type": "judge",
        "runs": ["stop.session"],
        "enforcement": "advisory",
        "rubric": "one two three four five six seven eight nine ten " * 6,
        "message": "m",
        "forbidden": ["CHANGELOG.md"],
    }
    rendered = hook_cli._gates_generate_describe_gate(gate)
    lines = rendered.split("\n")

    rubric_lines = [i for i, line in enumerate(lines) if "rubric" in line or "one two three" in line]
    assert len(rubric_lines) > 1, "the rubric value should wrap across more than one line"
    first_rubric_line, *continuation_lines = rubric_lines
    assert click.style("rubric", fg="cyan", bold=True) in lines[first_rubric_line]
    for idx in continuation_lines:
        assert "\x1b[2m" not in lines[idx], f"continuation line was dimmed like a list item: {lines[idx]!r}"

    forbidden_item_line = next(line for line in lines if "CHANGELOG.md" in line)
    assert "\x1b[2m" in forbidden_item_line


def test_appending_matches_the_files_own_column_zero_sequence_style(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `gates:` sequence written in column 0 (PyYAML's own dump style) is as valid as
    this repo's indented one, and the appended item has to land at the same column: a
    two-space item spliced in front of column-0 ones is invalid YAML, which `otari hook`
    then fails *open* on, silently disabling every gate in the policy.
    """
    (repo / ".otari-gates.yml").write_text(
        'schema_version: "1.0"\n'
        "policy:\n"
        "  id: demo/gates\n"
        "gates:\n"
        "- id: existing\n"
        "  type: path\n"
        "  runs: [pre_tool_use.edit_target, stop.working_tree]\n"
        "  enforcement: required\n"
        "  forbidden:\n"
        "  - x\n"
        "  message: m\n",
        encoding="utf-8",
    )
    _stub_claude_only(monkeypatch)
    _stub_cli_output(monkeypatch, json.dumps([_VALID_PROPOSAL]))

    result = _invoke(monkeypatch, keys="yy")
    assert result.exit_code == 0, result.output

    text = (repo / ".otari-gates.yml").read_text(encoding="utf-8")
    spec = parse_policy(text, source="check")
    assert {gate.id for gate in spec.gates} == {"existing", "no-hand-edited-changelog"}
    assert "\n- id: no-hand-edited-changelog\n" in text, text


def test_a_splice_that_would_not_parse_is_refused_with_the_file_untouched(tmp_path: Path) -> None:
    """The splice is a text heuristic where the loader is a parser, so the write is gated
    on `parse_policy` accepting the result rather than on the heuristic being right.
    """
    gates_file = tmp_path / ".otari-gates.yml"
    original = 'schema_version: "1.0"\npolicy:\n  id: t\ngates:\n  - id: e\n'
    gates_file.write_text(original, encoding="utf-8")

    with pytest.raises(click.ClickException) as excinfo:
        hook_cli._gates_generate_write_checked(gates_file, "gates: [oops\n")

    assert "no longer parses" in str(excinfo.value)
    assert gates_file.read_text(encoding="utf-8") == original


def test_scaffolded_policy_id_survives_an_unusual_directory_name(tmp_path: Path) -> None:
    """A directory name is not guaranteed to be a bare YAML scalar; one containing ": "
    would otherwise scaffold a header that does not parse at all.
    """
    gates_file = tmp_path / ".otari-gates.yml"
    hook_cli._gates_generate_append(gates_file, "weird: name", dict(_VALID_PROPOSAL))

    spec = parse_policy(gates_file.read_text(encoding="utf-8"), source="check")
    assert spec.policy_id == "weird: name/gates"


def test_a_source_file_that_is_not_utf8_is_a_clean_error(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary_source = repo / "not-text.md"
    binary_source.write_bytes(b"\xff\xfe\x00binary\x00")
    _stub_claude_only(monkeypatch)

    result = _invoke(monkeypatch, "--source", str(binary_source))
    assert result.exit_code != 0
    assert "as UTF-8 text" in result.output
    assert not isinstance(result.exception, UnicodeDecodeError)


def test_no_stdin_at_all_declines_rather_than_taking_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no controlling terminal *and* nothing on stdin, `input()` raises EOFError.
    That is not someone pressing Enter, so it must not resolve to a "y" default and make
    a model call nobody asked for.
    """

    def no_tty(echo: bool = False) -> str:
        raise OSError(6, "Device not configured")

    def no_stdin() -> str:
        raise EOFError

    monkeypatch.setattr(click, "getchar", no_tty)
    monkeypatch.setattr("builtins.input", no_stdin)

    assert hook_cli._gates_generate_read_choice("Use claude? [Y/n]: ", "yn", "y") == "n"
    assert hook_cli._gates_generate_read_choice("Add this gate? ", "yneq", "n") == "n"


def test_a_proposal_missing_runs_is_refused_before_anything_is_written(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The likeliest real failure: a model that learned the schema before `runs` existed.

    The prompt asks for the field, but a model is free to ignore it, so the
    parser is the thing that has to catch it. A refusal naming the legal values
    is the difference between an author fixing one line and a policy file that
    silently stops enforcing every gate in it.
    """
    _stub_claude_only(monkeypatch)
    _stub_cli_output(monkeypatch, json.dumps([_PROPOSAL_WITHOUT_RUNS]))

    result = _invoke(monkeypatch, keys="yn")

    assert result.exit_code == 0, result.output
    assert "does not pass validation" in result.output
    assert "needs a non-empty 'runs'" in result.output
    assert "pre_tool_use.edit_target, stop.working_tree" in result.output
    assert "Added 0 gate(s)" in result.output
    assert not (repo / ".otari-gates.yml").exists()
