"""Unit tests for the diff `otari hook` hands a judge gate.

Against a real repository, unlike tests/unit/test_hook_cli.py, which mocks
the Git boundary: what is under test here is which files Git reports and how
their content is rendered, so a mocked `subprocess.run` would only assert
that the fixture matches itself.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

import otari_agent.hook as hook_cli

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs a real git binary")


def _repo(root: Path) -> Path:
    """A repository with one committed file, ready for a working-tree change."""
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
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


def test_an_unreadable_file_is_named_rather_than_dropped(tmp_path: Path) -> None:
    """Silence would shorten the change a judge rules on without saying so."""
    repo = _repo(tmp_path)
    unreadable = repo / "locked.py"
    unreadable.write_text("value = 1\n", encoding="utf-8")
    unreadable.chmod(0o000)
    try:
        diff = hook_cli._hook_collect_diff(repo)
    finally:
        unreadable.chmod(0o600)

    assert diff is not None
    assert "b/locked.py" in diff
    assert "Unreadable, content not shown." in diff
