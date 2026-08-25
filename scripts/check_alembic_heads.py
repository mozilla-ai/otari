#!/usr/bin/env python3
"""Check the Alembic revision graph has exactly one head.

Two migrations that name the same `down_revision` are both heads, and alembic
refuses to resolve `head` when there is more than one. Nothing reports that at
the point it is introduced: the branch that adds the second child is green until
main lands the first, and then every test that builds a schema fails at once
with the same MultipleHeads error. That reads like a broken branch rather than
what it is, a rebase away from being correct.

This makes it a one line answer instead, and names the fix: chain the newer
revision onto the older rather than beside it.

Usage:
    uv run python scripts/check_alembic_heads.py

Exit codes:
    0 - Exactly one head
    1 - Zero heads, or more than one
"""

import sys
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

REPO_ROOT = Path(__file__).resolve().parent.parent


def find_heads(repo_root: Path = REPO_ROOT) -> list[str]:
    """Every revision in the graph that nothing else claims as a parent."""
    script = ScriptDirectory.from_config(Config(str(repo_root / "alembic.ini")))
    return list(script.get_heads())


def describe(revision: str, repo_root: Path = REPO_ROOT) -> str:
    """A head's revision id, its file, and the parent it claims."""
    script = ScriptDirectory.from_config(Config(str(repo_root / "alembic.ini")))
    rev = script.get_revision(revision)
    parent = rev.down_revision or "(base)"
    return f"  {revision}  down_revision={parent}\n    {Path(rev.path).name}"


def main() -> int:
    heads = find_heads()

    if len(heads) == 1:
        print(f"Single head: {heads[0]}")
        return 0

    if not heads:
        print("No Alembic head found. The revision graph is empty or every revision is claimed as a parent.")
        return 1

    print(f"{len(heads)} Alembic heads. `alembic upgrade head` cannot resolve a target:\n")
    for head in sorted(heads):
        print(describe(head))
    print(
        "\nTwo revisions claim the same parent. Re-point the newer one's `down_revision`\n"
        "at the older head so they chain instead of branching, then re-run the migration\n"
        "round-trip: the ordering changed, so earlier round-trip evidence no longer covers it."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
