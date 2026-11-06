"""Unit tests for the diff `otari hook` hands a judge gate.

Against a real repository, unlike tests/unit/test_hook_cli.py, which mocks
the Git boundary: what is under test here is which files Git reports and how
their content is rendered, so a mocked `subprocess.run` would only assert
that the fixture matches itself.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

import otari_agent.hook as hook_cli
from otari_agent.domain.policy import parse_policy
from otari_agent.domain.types import JudgeVerdict, PolicySpec

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs a real git binary")


def _repo(root: Path) -> Path:
    """A repository with one committed file, ready for a working-tree change."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    (root / "kept.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=root,
        check=True,
    )
    return root


def test_an_untracked_file_reaches_the_diff(tmp_path: Path) -> None:
    """The gap this closes: a new file's content, not only its path."""
    repo = _repo(tmp_path)
    (repo / "added.py").write_text("SECRET = 'leaked'\n", encoding="utf-8")

    diff = hook_cli._hook_collect_diff(repo)

    assert diff is not None
    assert "+++ b/added.py" in diff
    assert "+SECRET = 'leaked'" in diff


def test_a_tracked_change_still_reaches_the_diff(tmp_path: Path) -> None:
    """An edit's own hunk, which carries the surrounding code a new file has none of."""
    repo = _repo(tmp_path)
    (repo / "kept.py").write_text("value = 2\n", encoding="utf-8")
    (repo / "added.py").write_text("other = 3\n", encoding="utf-8")

    diff = hook_cli._hook_collect_diff(repo)

    assert diff is not None
    assert "+value = 2" in diff
    assert "+other = 3" in diff


def test_an_ignored_file_stays_out_of_the_diff(tmp_path: Path) -> None:
    """Without --exclude-standard this is most of a real tree (.venv, node_modules)."""
    repo = _repo(tmp_path)
    (repo / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    (repo / "ignored").mkdir()
    (repo / "ignored" / "vendored.py").write_text("noise = 1\n", encoding="utf-8")

    diff = hook_cli._hook_collect_diff(repo)

    assert diff is not None
    assert "vendored.py" not in diff
    assert "noise = 1" not in diff


def test_an_untracked_binary_is_named_but_not_quoted(tmp_path: Path) -> None:
    """A judge gate has nothing to read in the bytes, and they would spend the budget."""
    repo = _repo(tmp_path)
    (repo / "blob.bin").write_bytes(b"\x00\x01\x02payload")

    diff = hook_cli._hook_collect_diff(repo)

    assert diff is not None
    assert "b/blob.bin" in diff
    assert "Binary file, content not shown." in diff
    assert "payload" not in diff


def test_one_large_untracked_file_cannot_crowd_out_the_others(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-file cap is what keeps a generated artifact from taking the whole budget."""
    monkeypatch.setattr(hook_cli, "_HOOK_JUDGE_MAX_UNTRACKED_FILE_CHARS", 200)
    repo = _repo(tmp_path)
    (repo / "generated.py").write_text("x = 1\n" * 500, encoding="utf-8")
    (repo / "small.py").write_text("real_change = True\n", encoding="utf-8")

    diff = hook_cli._hook_collect_diff(repo)

    assert diff is not None
    assert "... (file truncated)" in diff
    assert "+real_change = True" in diff


def test_the_diff_budget_stops_at_a_whole_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fragment would read to a judge as the end of the change."""
    monkeypatch.setattr(hook_cli, "_HOOK_JUDGE_MAX_DIFF_CHARS", 400)
    repo = _repo(tmp_path)
    for index in range(20):
        (repo / f"file_{index:02d}.py").write_text(f"value = {index}\n" * 5, encoding="utf-8")

    diff = hook_cli._hook_collect_diff(repo)

    assert diff is not None
    assert len(diff) <= hook_cli._HOOK_JUDGE_MAX_DIFF_CHARS
    assert "... (diff truncated)" not in diff


def test_a_collection_failure_is_still_none(tmp_path: Path) -> None:
    """Fail-open sentinel: outside a repository there is no diff, not an empty one."""
    assert hook_cli._hook_collect_diff(tmp_path) is None


def test_a_symlink_is_rendered_without_reading_its_target(tmp_path: Path) -> None:
    """The ignore rules filter a link's path, not where it points."""
    nested = tmp_path / "repo"
    nested.mkdir()
    repo = _repo(nested)
    secret = tmp_path / "outside.env"
    secret.write_text("AWS_SECRET_ACCESS_KEY=leaked\n", encoding="utf-8")
    (repo / "link.env").symlink_to(secret)

    diff = hook_cli._hook_collect_diff(repo)

    assert diff is not None
    assert "b/link.env" in diff
    assert "target not read" in diff
    assert "leaked" not in diff


def test_a_symlink_to_an_ignored_file_is_not_read_either(tmp_path: Path) -> None:
    """Ignoring the target does nothing: the link itself is what Git reports."""
    repo = _repo(tmp_path)
    (repo / ".gitignore").write_text("hidden.env\n", encoding="utf-8")
    (repo / "hidden.env").write_text("TOKEN=leaked\n", encoding="utf-8")
    (repo / "link.env").symlink_to(repo / "hidden.env")

    diff = hook_cli._hook_collect_diff(repo)

    assert diff is not None
    assert "target not read" in diff
    assert "leaked" not in diff


def test_listing_untracked_files_failing_is_a_collection_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not an empty diff: a change made only of new files would look like no change."""
    repo = _repo(tmp_path)
    (repo / "added.py").write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(hook_cli, "_hook_untracked_paths", lambda repo_root: None)

    assert hook_cli._hook_collect_diff(repo) is None


def test_an_unreadable_file_is_named_rather_than_dropped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence would shorten the change a judge rules on without saying so.

    The refusal is injected rather than made with `chmod(0o000)`, which a
    privileged runner reads straight through and Windows does not honor at
    all. What is under test is that an `OSError` becomes a named placeholder,
    not the platform's own permission semantics.
    """
    repo = _repo(tmp_path)
    (repo / "locked.py").write_text("value = 1\n", encoding="utf-8")
    (repo / "open.py").write_text("readable = True\n", encoding="utf-8")
    real_open = os.open

    def refuse_one(path: object, *args: int, **kwargs: object) -> int:
        if str(path).endswith("locked.py"):
            raise PermissionError(13, "Permission denied")
        return real_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", refuse_one)

    diff = hook_cli._hook_collect_diff(repo)

    assert diff is not None
    assert "b/locked.py" in diff
    assert "Unreadable, content not shown." in diff
    assert "+readable = True" in diff, "one refused file must not drop the others"


def _commit(repo: Path) -> str:
    """Commit the whole working tree and return the new commit's SHA."""
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "change"],
        cwd=repo,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_a_range_ending_at_a_commit_reads_only_the_commits(tmp_path: Path) -> None:
    """A range with a head reads Git objects, so nothing in the working tree reaches it."""
    repo = _repo(tmp_path)
    (repo / "kept.py").write_text("value = 2\n", encoding="utf-8")
    (repo / "committed.py").write_text("committed = True\n", encoding="utf-8")
    head = _commit(repo)
    (repo / "kept.py").write_text("value = 3\n", encoding="utf-8")
    (repo / "untracked.py").write_text("untracked = True\n", encoding="utf-8")

    diff = hook_cli._hook_collect_diff(repo, hook_cli._DiffRange(f"{head}~1", head))

    assert diff is not None
    assert "+value = 2" in diff
    assert "+committed = True" in diff
    assert "value = 3" not in diff
    assert "untracked.py" not in diff


def test_a_range_ending_at_the_working_tree_starts_at_its_base(tmp_path: Path) -> None:
    """Committed and uncommitted changes since the base both reach the diff."""
    repo = _repo(tmp_path)
    (repo / "committed.py").write_text("committed = True\n", encoding="utf-8")
    head = _commit(repo)
    (repo / "untracked.py").write_text("untracked = True\n", encoding="utf-8")

    diff = hook_cli._hook_collect_diff(repo, hook_cli._DiffRange(f"{head}~1"))

    assert diff is not None
    assert "+committed = True" in diff
    assert "+untracked = True" in diff


def test_stop_range_uses_the_upstream_merge_base_for_committed_and_working_tree_changes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(["git", "update-ref", "refs/remotes/origin/feature", base], cwd=repo, check=True)
    subprocess.run(["git", "switch", "-q", "-c", "feature"], cwd=repo, check=True)
    subprocess.run(["git", "config", "remote.origin.url", "https://example.invalid/repo.git"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "branch.feature.remote", "origin"], cwd=repo, check=True)
    subprocess.run(["git", "config", "branch.feature.merge", "refs/heads/feature"], cwd=repo, check=True)
    (repo / "committed.py").write_text("committed = True\n", encoding="utf-8")
    _commit(repo)
    (repo / "uncommitted.py").write_text("uncommitted = True\n", encoding="utf-8")

    diff_range = hook_cli._hook_collect_stop_diff_range(repo)
    assert diff_range is not None
    assert diff_range.base == base
    assert diff_range.description is not None
    assert "upstream origin/feature merge base" in diff_range.description

    paths = hook_cli._hook_collect_changed_paths(repo, diff_range)
    assert paths is not None
    assert paths == ["committed.py", "uncommitted.py"]
    diff = hook_cli._hook_collect_diff(repo, diff_range)
    assert diff is not None
    assert "+committed = True" in diff
    assert "+uncommitted = True" in diff


def test_stop_range_excludes_default_branch_changes_after_rebase(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    subprocess.run(["git", "switch", "-q", "-c", "feature"], cwd=repo, check=True)
    (repo / "feature.py").write_text("feature = True\n", encoding="utf-8")
    old_feature = _commit(repo)
    subprocess.run(["git", "update-ref", "refs/remotes/origin/feature", old_feature], cwd=repo, check=True)
    subprocess.run(["git", "config", "remote.origin.url", "https://example.invalid/repo.git"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "branch.feature.remote", "origin"], cwd=repo, check=True)
    subprocess.run(["git", "config", "branch.feature.merge", "refs/heads/feature"], cwd=repo, check=True)
    subprocess.run(["git", "switch", "-q", "main"], cwd=repo, check=True)
    (repo / "CHANGELOG.md").write_text("release\n", encoding="utf-8")
    main = _commit(repo)
    subprocess.run(["git", "update-ref", "refs/remotes/origin/main", main], cwd=repo, check=True)
    subprocess.run(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main"], cwd=repo, check=True
    )
    subprocess.run(["git", "switch", "-q", "feature"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", "rebase", "main"], cwd=repo, check=True
    )

    diff_range = hook_cli._hook_collect_stop_diff_range(repo)
    assert diff_range is not None
    assert diff_range.base == main
    assert hook_cli._hook_collect_changed_paths(repo, diff_range) == ["feature.py"]
    diff = hook_cli._hook_collect_diff(repo, diff_range)
    assert diff is not None
    assert "feature = True" in diff
    assert "CHANGELOG.md" not in diff


def test_stop_range_on_default_branch_does_not_include_its_commits(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "committed.py").write_text("committed = True\n", encoding="utf-8")
    head = _commit(repo)

    diff_range = hook_cli._hook_collect_stop_diff_range(repo)
    assert diff_range is not None
    assert diff_range.base == head
    assert hook_cli._hook_collect_changed_paths(repo, diff_range) == []
    assert hook_cli._hook_collect_diff(repo, diff_range) == ""


def test_stop_range_uses_default_branch_when_pushed_upstream_is_at_head(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(["git", "update-ref", "refs/remotes/origin/main", base], cwd=repo, check=True)
    subprocess.run(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "remote.origin.url", "https://example.invalid/repo.git"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*"], cwd=repo, check=True
    )
    subprocess.run(["git", "switch", "-q", "-c", "feature"], cwd=repo, check=True)
    subprocess.run(["git", "config", "branch.feature.remote", "origin"], cwd=repo, check=True)
    subprocess.run(["git", "config", "branch.feature.merge", "refs/heads/feature"], cwd=repo, check=True)
    (repo / "committed.py").write_text("committed = True\n", encoding="utf-8")
    head = _commit(repo)
    subprocess.run(["git", "update-ref", "refs/remotes/origin/feature", head], cwd=repo, check=True)
    (repo / "uncommitted.py").write_text("uncommitted = True\n", encoding="utf-8")

    diff_range = hook_cli._hook_collect_stop_diff_range(repo)

    assert diff_range is not None
    assert diff_range.base == base
    assert diff_range.description is not None
    assert "default branch refs/remotes/origin/main merge base" in diff_range.description
    assert hook_cli._hook_collect_changed_paths(repo, diff_range) == ["committed.py", "uncommitted.py"]
    diff = hook_cli._hook_collect_diff(repo, diff_range)
    assert diff is not None
    assert "+committed = True" in diff
    assert "+uncommitted = True" in diff


def test_stop_range_falls_back_to_the_remote_default_branch_without_an_upstream(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(["git", "update-ref", "refs/remotes/origin/main", base], cwd=repo, check=True)
    subprocess.run(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main"], cwd=repo, check=True
    )
    subprocess.run(["git", "switch", "-q", "-c", "feature"], cwd=repo, check=True)
    (repo / "committed.py").write_text("committed = True\n", encoding="utf-8")
    _commit(repo)

    diff_range = hook_cli._hook_collect_stop_diff_range(repo)
    assert diff_range is not None
    assert diff_range.base == base
    assert diff_range.description is not None
    assert "default branch refs/remotes/origin/main merge base" in diff_range.description


def test_stop_range_falls_back_to_the_default_branch_when_upstream_has_no_common_history(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    unrelated_upstream = subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=t",
            "commit-tree",
            tree,
            "-m",
            "unrelated upstream",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "update-ref", "refs/remotes/origin/main", base], cwd=repo, check=True)
    subprocess.run(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main"], cwd=repo, check=True
    )
    subprocess.run(["git", "update-ref", "refs/remotes/origin/feature", unrelated_upstream], cwd=repo, check=True)
    subprocess.run(["git", "config", "remote.origin.url", "https://example.invalid/repo.git"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*"], cwd=repo, check=True
    )
    subprocess.run(["git", "switch", "-q", "-c", "feature"], cwd=repo, check=True)
    subprocess.run(["git", "config", "branch.feature.remote", "origin"], cwd=repo, check=True)
    subprocess.run(["git", "config", "branch.feature.merge", "refs/heads/feature"], cwd=repo, check=True)
    (repo / "committed.py").write_text("committed = True\n", encoding="utf-8")
    _commit(repo)

    diff_range = hook_cli._hook_collect_stop_diff_range(repo)

    assert diff_range is not None
    assert diff_range.base == base
    assert diff_range.description is not None
    assert "default branch refs/remotes/origin/main merge base" in diff_range.description


def test_stop_range_falls_back_to_head_when_no_upstream_or_default_branch_exists(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    subprocess.run(["git", "switch", "-q", "-c", "topic"], cwd=repo, check=True)
    subprocess.run(["git", "branch", "-D", "main"], cwd=repo, check=True, capture_output=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    (repo / "untracked.py").write_text("untracked = True\n", encoding="utf-8")

    diff_range = hook_cli._hook_collect_stop_diff_range(repo)

    assert diff_range is not None
    assert diff_range.base == head
    assert diff_range.description == "HEAD (no upstream or default branch was available)"
    assert hook_cli._hook_collect_changed_paths(repo, diff_range) == ["untracked.py"]
    diff = hook_cli._hook_collect_diff(repo, diff_range)
    assert diff is not None
    assert "+untracked = True" in diff


def test_stop_judge_receives_committed_changes_and_reports_its_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(["git", "update-ref", "refs/remotes/origin/main", base], cwd=repo, check=True)
    subprocess.run(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main"], cwd=repo, check=True
    )
    subprocess.run(["git", "switch", "-q", "-c", "feature"], cwd=repo, check=True)
    (repo / "committed.py").write_text("# TODO: review this\n", encoding="utf-8")
    _commit(repo)
    diff_range = hook_cli._hook_collect_stop_diff_range(repo)
    assert diff_range is not None
    paths = hook_cli._hook_collect_changed_paths(repo, diff_range)
    assert paths == ["committed.py"]

    spec = parse_policy(
        "schema_version: '1.0'\n"
        "policy:\n  id: test\n"
        "gates:\n"
        "  - id: review\n"
        "    type: judge\n"
        "    runs: [stop.session]\n"
        "    enforcement: advisory\n"
        "    rubric: Find TODO comments.\n",
        source="test",
    )
    seen: list[str] = []
    monkeypatch.setattr(hook_cli, "_hook_log_judge_call", lambda *args, **kwargs: None)

    def fake_run_judge(rubric: str, diff: str, transcript: str, **kwargs: object) -> tuple[str, str]:
        seen.append(diff)
        return "fail", "TODO found"

    monkeypatch.setattr(hook_cli, "_hook_run_judge", fake_run_judge)
    verdicts = hook_cli._hook_collect_judge_verdicts(
        spec,
        hook_cli._JudgedChange(repo, paths or [], diff_range=diff_range),
        hook_cli._JudgeSettings(max_gates=5),
    )
    assert verdicts[0].outcome == "fail"
    assert "Diff base: default branch refs/remotes/origin/main merge base" in seen[0]
    assert "+# TODO: review this" in seen[0]
    assert "Diff base: default branch refs/remotes/origin/main merge base" in verdicts[0].reasoning


def test_stop_hook_blocks_a_committed_path_and_sends_the_same_range_to_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    repo = _repo(tmp_path)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(["git", "update-ref", "refs/remotes/origin/main", base], cwd=repo, check=True)
    subprocess.run(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main"], cwd=repo, check=True
    )
    subprocess.run(["git", "switch", "-q", "-c", "feature"], cwd=repo, check=True)
    (repo / "committed.py").write_text("# TODO: review this\n", encoding="utf-8")
    _commit(repo)
    (repo / ".otari").mkdir()
    (repo / ".otari/guardrails.yml").write_text(
        "schema_version: '1.0'\n"
        "policy:\n  id: test\n"
        "gates:\n"
        "  - id: protected-path\n"
        "    type: path\n"
        "    runs: [stop.working_tree]\n"
        "    enforcement: required\n"
        "    forbidden: ['committed.py']\n"
        "    message: committed path must be blocked\n"
        "  - id: review\n"
        "    type: judge\n"
        "    runs: [stop.session]\n"
        "    enforcement: advisory\n"
        "    rubric: Find TODO comments.\n"
        "    message: TODO found\n",
        encoding="utf-8",
    )

    judged_changes: list[hook_cli._JudgedChange] = []

    def fake_collect_judges(
        spec: PolicySpec,
        change: hook_cli._JudgedChange,
        settings: hook_cli._JudgeSettings,
        runs: dict[str, hook_cli._GateRun] | None = None,
    ) -> list[JudgeVerdict]:
        judged_changes.append(change)
        return [JudgeVerdict(gate_id="review", outcome="pass", reasoning="clear")]

    monkeypatch.setattr(hook_cli, "_hook_collect_judge_verdicts", fake_collect_judges)
    result = CliRunner().invoke(
        hook_cli.hook,
        input=json.dumps({"hook_event_name": "Stop", "cwd": str(repo)}),
    )

    assert result.exit_code == 2, result.output
    assert "committed path must be blocked" in result.output
    assert len(judged_changes) == 1
    assert "committed.py" in judged_changes[0].changed_paths
    assert judged_changes[0].diff_range.base == base
