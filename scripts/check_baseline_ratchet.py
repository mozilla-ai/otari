#!/usr/bin/env python3
"""Check that a change gives no baseline of the architecture check back room it had given up.

Every baseline in the architecture check records debt the repository has already
taken on, and each one only shrinks. The check itself proves the code agrees with
the number in front of it, so a change that adds a builder and raises the number
together passes. This reads the two revisions instead, and fails when the later
one records more room than the earlier one.

A baseline that is new passes, because that is how a rule arrives with the debt
it found. A baseline that is gone passes only once it is empty, because a rule is
retired after its debt is paid rather than before. A baseline whose room cannot
be read fails either way, because a comparison that proves nothing is not a pass.

NOTE: This takes two files and reads neither repository. The caller decides which
revision is which, so the same command serves a pull request and a local branch.
Either file may be given as ``-``, which reads it from standard input.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

BASELINE_SUFFIX = "_BASELINE"
BASELINE_KEY = "baseline"

_ONLY_FALLS = "a baseline records room already given up, so it may only fall"
_ONLY_LOSES = "a baseline records room already given up, so it may only lose entries"


class Unreadable:
    """A baseline whose recorded room cannot be read from the source alone."""

    def __init__(self, why: str) -> None:
        self.why = why


_NOT_A_LITERAL = "is not written out in full"
_WRITTEN_AGAIN = "is written at module scope more than once, so the last value decides"
_UNDECLARED = "is written at module scope without a declaration this check can read"
_UNSUPPORTED = "records a kind of value this check does not read"


def _module_scope_statements(body: Sequence[ast.stmt]) -> Iterator[ast.stmt]:
    """Yield each statement that runs at module scope, descending through control flow but not into a new scope."""
    for node in body:
        yield node
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ExceptHandler | ast.match_case):
                yield from _module_scope_statements(child.body)
            elif isinstance(child, ast.stmt):
                yield from _module_scope_statements([child])


def _names_written(statement: ast.stmt) -> Iterator[str]:
    """Yield the name at the root of each target one module-scope statement writes.

    A write through a subscript or an attribute counts against the name it starts from,
    so assigning to one field of a constant counts as writing that constant.

    NOTE: This reads the targets of the statements that declare a constant. A name
    written by anything more exotic is not a form the convention produces.
    """
    targets: list[ast.expr] = []
    if isinstance(statement, ast.Assign):
        targets.extend(statement.targets)
    elif isinstance(statement, ast.AugAssign):
        targets.append(statement.target)
    elif isinstance(statement, ast.AnnAssign):
        # A bare annotation declares a type and writes nothing.
        if statement.value is not None:
            targets.append(statement.target)
    elif isinstance(statement, ast.For | ast.AsyncFor):
        targets.append(statement.target)
    elif isinstance(statement, ast.With | ast.AsyncWith):
        targets.extend(item.optional_vars for item in statement.items if item.optional_vars is not None)
    for target in targets:
        yield from _roots(target)


def _roots(target: ast.expr) -> Iterator[str]:
    """Yield the name each part of one assignment target ultimately writes through."""
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, ast.Subscript | ast.Attribute):
        yield from _roots(target.value)
    elif isinstance(target, ast.Tuple | ast.List):
        for element in target.elts:
            yield from _roots(element)
    elif isinstance(target, ast.Starred):
        yield from _roots(target.value)


def _declarations(body: Sequence[ast.stmt]) -> Iterator[tuple[str, ast.expr]]:
    """Yield the name and the written-out value of each constant one module declares at its top level."""
    for node in body:
        if isinstance(node, ast.AnnAssign):
            names = [node.target.id] if isinstance(node.target, ast.Name) else []
        elif isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        else:
            continue
        if node.value is None:
            continue
        for name in names:
            yield name, node.value


def _nested_baselines(label: str, node: ast.expr) -> Iterator[tuple[str, ast.expr]]:
    """Yield the written-out value of each baseline a constant records under a ``baseline`` key.

    This reads the mapping as it is written rather than as a whole value, so one entry
    that is not written out in full costs only that entry.
    """
    if not isinstance(node, ast.Dict):
        return
    for key, value in zip(node.keys, node.values, strict=True):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            continue
        if key.value == BASELINE_KEY:
            yield f"{label}[{BASELINE_KEY!r}]", value
        else:
            yield from _nested_baselines(f"{label}[{key.value!r}]", value)


def _supported(value: object) -> bool:
    """Whether one baseline records room in a form this check knows how to compare.

    NOTE: A count is an ``int`` exactly. A ``bool`` is one to Python, but never a count,
    and comparing one against a count would read ``True`` as a ceiling of one.
    """
    if type(value) is int:
        return True
    if isinstance(value, dict):
        return all(type(key) is str and _supported(item) for key, item in value.items())
    if isinstance(value, tuple | list):
        return all(_entry(entry) for entry in value)
    return False


def _entry(entry: object) -> bool:
    """Whether one entry of a baseline names something this check can compare."""
    if isinstance(entry, tuple):
        return all(type(part) is str for part in entry)
    return type(entry) is str


def _read_value(node: ast.expr) -> object:
    """Return the room one declaration records, or an ``Unreadable`` saying why it cannot be read."""
    try:
        value = ast.literal_eval(node)
    except (TypeError, ValueError):
        return Unreadable(_NOT_A_LITERAL)
    return value if _supported(value) else Unreadable(_UNSUPPORTED)


def baselines(source: str) -> dict[str, object]:
    """Return the room each baseline of one revision of the check records, by the name that reaches it.

    A baseline is a constant whose name ends in the baseline suffix, or a ``baseline``
    key inside a constant that records one per entry. One whose room cannot be read maps
    to an ``Unreadable``.
    """
    tree = ast.parse(source)
    written: dict[str, int] = {}
    for statement in _module_scope_statements(tree.body):
        for name in _names_written(statement):
            written[name] = written.get(name, 0) + 1

    declared: dict[str, tuple[str, ast.expr]] = {}
    for name, node in _declarations(tree.body):
        if name.endswith(BASELINE_SUFFIX):
            declared[name] = (name, node)
            continue
        for label, value in _nested_baselines(name, node):
            declared[label] = (name, value)

    found: dict[str, object] = {}
    for label, (owner, node) in declared.items():
        found[label] = Unreadable(_WRITTEN_AGAIN) if written.get(owner, 0) > 1 else _read_value(node)
    for name in written:
        if name.endswith(BASELINE_SUFFIX) and name not in found:
            found[name] = Unreadable(_UNDECLARED)
    return found


def check_baselines(base: str, head: str) -> list[str]:
    """Return each way one revision of the check gives a baseline back room an earlier one had given up."""
    if not base.strip():
        return ["the earlier revision is empty, so there is nothing to measure against"]
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
    findings.extend(_arrived(name, after[name]) for name in sorted(after) if name not in before)
    return [finding for finding in findings if finding]


def _arrived(name: str, recorded: object) -> str:
    """Return a finding when a baseline arrives recording room that cannot be read."""
    if isinstance(recorded, Unreadable):
        return f"{name} arrives and {recorded.why}; a baseline is read without running the check"
    return ""


def _compared(label: str, before: object, after: object) -> list[str]:
    """Return each way one baseline records more room than it did, naming the part that moved."""
    if isinstance(after, Unreadable):
        return [f"{label} {after.why}; a baseline is read without running the check"]
    if isinstance(before, Unreadable):
        return [f"{label} cannot be measured, because at the branch point it {before.why}"]
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
    if isinstance(recorded, Unreadable):
        return [f"{name} is gone, and whether it was empty cannot be read, because it {recorded.why}"]
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
