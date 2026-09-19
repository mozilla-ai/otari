#!/usr/bin/env python3
"""check_passed verifier: fails when a docblock in web/src sits directly above
a declaration it does not document (AGENTS.md, "Repository Conventions").

Run with cwd at the repo root. Exit 0 is pass, 1 is fail, anything else is
error, per docs/agent-gates.md's check_passed exit-code contract. Python's
own `re`, not a shelled-out `grep`, is what makes the match itself correct:
the pattern spans a newline, which a real `grep -P` invocation only handles
portably with `--null-data`, and BSD grep has no `-P` at all either way.

Scoped to files that actually changed (working tree vs HEAD, the same
"changed" `otari hook` itself submits as evidence), not a scan of the whole
web/src tree: web/src carries pre-existing violations this one script did
not introduce, and a whole-tree scan would fail this gate on every future
change to web/src regardless of what that change touched. Touching a file
that already has one still flags it (the same "if you touch it, fix it"
rule linters scoped to a diff already follow), which is a much narrower
surface than every unrelated edit in the directory.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

TARGET_PREFIX = "web/src/"
PATTERN = re.compile(r"\*/\n[ \t]*/\*\*")


def changed_paths_under_target() -> list[str] | None:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None

    paths = []
    for line in result.stdout.splitlines():
        entry = line[3:]  # two status chars + a space precede the path
        if " -> " in entry:  # a rename: "old -> new"; only the new path still exists
            entry = entry.split(" -> ", 1)[1]
        paths.append(entry)
    return [p for p in paths if p.startswith(TARGET_PREFIX)]


def main() -> int:
    paths = changed_paths_under_target()
    if paths is None:
        print("could not determine changed files via `git status`")
        return 2

    offenders = []
    for rel in paths:
        path = Path(rel)
        if not path.is_file():
            continue  # deleted, or a directory entry
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            # Exit 1 is reserved for a real docblock mismatch. A file that
            # cannot be read (permissions, a broken symlink, a race against
            # the working tree) is an execution failure, and reporting it as
            # a mismatch would blame the diff for the verifier's own problem.
            print(f"could not read {rel}: {exc}")
            return 2
        match = PATTERN.search(text)
        if match:
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{rel}:{line}")

    if offenders:
        print("\n".join(offenders))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
