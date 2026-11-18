#!/usr/bin/env python3
"""Check that a change gives no baseline of the architecture check back room it had given up.

Every ``*_BASELINE`` in the architecture check records debt the repository has
already taken on, and each one only shrinks. The check itself proves the code
agrees with the number in front of it, so a change that moves a builder and the
number together passes. This reads the two revisions instead, and fails when the
later one records more room than the earlier one.

A baseline that is new passes, because that is how a rule arrives with the debt
it found. A baseline that is gone passes only once it is empty, because a rule
is retired after its debt is paid rather than before.

NOTE: This takes two files and reads neither repository. The caller decides which
revision is which, so the same command serves a pull request and a local branch.
Either file may be given as ``-``, which reads it from standard input.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Sequence
from pathlib import Path

BASELINE_SUFFIX = "_BASELINE"

_ONLY_FALLS = "a baseline records room already given up, so it may only fall"
_ONLY_LOSES = "a baseline records room already given up, so it may only lose entries"


class _NotWrittenOut:
    """A baseline whose value is not written out, so it cannot be read without running the check."""


NOT_WRITTEN_OUT = _NotWrittenOut()


def baselines(source: str) -> dict[str, object]:
    """Return the value each baseline of one revision of the check is written out as, by name.

    A baseline that is not written out in full maps to ``NOT_WRITTEN_OUT``.
    """
    found: dict[str, object] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.AnnAssign):
            names = [node.target.id] if isinstance(node.target, ast.Name) else []
        elif isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        else:
            continue
        if node.value is None:
            continue
        for name in names:
            if not name.endswith(BASELINE_SUFFIX):
                continue
            try:
                found[name] = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                found[name] = NOT_WRITTEN_OUT
    return found


def check_baselines(base: str, head: str) -> list[str]:
    """Return each way one revision of the check gives a baseline back room an earlier one had given up."""
    try:
        before = baselines(base)
        after = baselines(head)
    except SyntaxError:
        return ["the check does not parse, so no baseline can be read from it"]
    findings: list[str] = []
    for name, recorded in sorted(before.items()):
        if name in after:
            findings.extend(_compared(name, recorded, after[name]))
        else:
            findings.extend(_retired(name, recorded))
    return findings


def _compared(label: str, before: object, after: object) -> list[str]:
    """Return each way one baseline records more room than it did, naming the part that moved."""
    if isinstance(after, _NotWrittenOut):
        return [f"{label} is no longer written out in full; a baseline is read without running the check"]
    if isinstance(before, _NotWrittenOut):
        return []
    if isinstance(before, int) and isinstance(after, int):
        return [f"{label} rises from {before} to {after}; {_ONLY_FALLS}"] if after > before else []
    if isinstance(before, dict) and isinstance(after, dict):
        findings = [f"{label} gains the entry {key!r}; {_ONLY_LOSES}" for key in after if key not in before]
        for key, recorded in before.items():
            if key in after:
                findings.extend(_compared(f"{label}[{key!r}]", recorded, after[key]))
        return findings
    if isinstance(before, tuple | list) and isinstance(after, tuple | list):
        return [f"{label} gains {entry!r}; {_ONLY_LOSES}" for entry in after if entry not in before]
    return [f"{label} changes from {_shape(before)} to {_shape(after)}; a reader cannot tell what room it gives back"]


def _retired(name: str, recorded: object) -> list[str]:
    """Return a finding when a baseline is gone before the room it records was given back."""
    if isinstance(recorded, _NotWrittenOut):
        return []
    if isinstance(recorded, int):
        if recorded == 0:
            return []
        return [f"{name} is gone while it still stood at {recorded}; take it to zero before the rule that reads it"]
    held = len(recorded)  # type: ignore[arg-type]
    if held == 0:
        return []
    entries = "entry" if held == 1 else "entries"
    return [f"{name} is gone while it still held {held} {entries}; empty it before the rule that reads it goes"]


def _shape(value: object) -> str:
    """Return the kind of room one baseline records, in the words its findings use."""
    if isinstance(value, int):
        return "a count"
    if isinstance(value, dict):
        return "a map of entries"
    return "a list of entries"


def _read(path: Path) -> str:
    """Return one revision of the check, reading standard input when the path is a single dash."""
    if str(path) == "-":
        return sys.stdin.read()
    return path.read_text(encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Compare the baselines of two revisions of the architecture check."""
    parser = argparse.ArgumentParser(description="Check that no baseline of the architecture check gains room.")
    parser.add_argument("--base", required=True, type=Path, help="the check as the revision branched from has it")
    parser.add_argument("--head", required=True, type=Path, help="the check as the change has it")
    arguments = parser.parse_args(argv)

    sources = []
    for path in (arguments.base, arguments.head):
        try:
            sources.append(_read(path))
        except OSError as error:
            print(f"❌ Cannot read {path}: {error}")
            return 1

    findings = check_baselines(*sources)
    if findings:
        print("\n❌ Baselines that give back room:\n")
        for finding in findings:
            print(f"  {finding}")
        print(f"\nTotal baselines that give back room: {len(findings)}")
        print("\n💡 A baseline records debt already taken on, not a budget the next change may spend")
        return 1

    print("✅ No baseline gives back room")
    return 0


if __name__ == "__main__":
    sys.exit(main())
