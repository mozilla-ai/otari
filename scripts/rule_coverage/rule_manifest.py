"""Rule-coverage manifest: parsing and drift detection.

A skill under ``.github/skills/`` is the long form of a standard, written for
whoever is editing the code. Its counterpart under ``.github/instructions/`` is
the short form CodeRabbit loads as review guidance, and it is a deliberate
compression: one numbered rung carries what several skill sections say, phrased
as what is a finding rather than as how to write code. Nothing has ever compared
the two, so a rule could be stated in one and absent from the other, and the only
thing that noticed was a person asking.

A manifest classifies every skill heading as ``[covered]`` by a named rung or
``[excluded]`` with a reason, and the checks below assert the map is total in
both directions: a heading nothing classifies fails, an entry naming a heading
that no longer exists fails, and a rung that no entry names (or that the
instructions no longer number) fails too. Adding a section to a topic guide then
fails the repository's own tests until somebody decides whether a reviewer needs
it, which is a sentence of work at the moment the decision is cheap.

The comparison is structural on purpose. The two layers say the same thing in
deliberately different words, so anything comparing wording would fight that
difference forever.

Format: ``[covered]`` / ``[excluded]`` sections, one ``<file> :: <heading>`` per
line. Inside ``[covered]`` a ``rung <n>`` line opens the run of headings that
rung carries. An entry takes an optional ``# reason`` trailer, required under
``[excluded]``. Blank lines and ``#`` lines are ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = Path(__file__).resolve().parent

FRONTEND_MANIFEST = MANIFEST_DIR / "frontend-standards.txt"
FRONTEND_SKILL_DIR = REPO_ROOT / ".github" / "skills" / "frontend-standards"
FRONTEND_INSTRUCTIONS = (
    REPO_ROOT / ".github" / "instructions" / "frontend-standards.instructions.md"
)

# `<file> :: <heading>`. Two colons rather than one because a heading may end in
# one, and rather than `#` because `#` already means "reason trailer" here.
SEPARATOR = " :: "

# The unit is an `##`/`###` heading. An `#` is the document's own title, which
# names the file rather than stating a rule.
_HEADING = re.compile(r"^#{2,3}\s+(\S.*?)\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
# A rung is any top-level numbered item. Reading the number rather than the
# bold that usually follows it means a rung written without one is still seen,
# instead of silently dropping out of the "every rung is grounded" direction.
_RUNG = re.compile(r"^(\d+)\.\s+(\S.*?)\s*$")
_SUBJECT = re.compile(r"^\*\*(.+?)\*\*")
_RUNG_MARKER = re.compile(r"^rung\s+(\d+)$")


@dataclass(frozen=True)
class Manifest:
    """Every classified heading: ``covered`` maps to a rung, ``excluded`` to a reason."""

    covered: dict[str, int]
    excluded: dict[str, str]

    @property
    def entries(self) -> set[str]:
        return set(self.covered) | set(self.excluded)


def parse_manifest(text: str) -> Manifest:
    """Parse manifest text, raising ``ValueError`` on anything malformed.

    A malformed line is an editing slip in the commit under test rather than a
    condition to tolerate, so it fails where it is read instead of silently
    classifying nothing.
    """
    covered: dict[str, int] = {}
    excluded: dict[str, str] = {}
    section: str | None = None
    rung: int | None = None

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line in {"[covered]", "[excluded]"}:
            section, rung = line[1:-1], None
            continue
        marker = _RUNG_MARKER.match(line)
        if marker:
            if section != "covered":
                raise ValueError(f"line {number}: a `rung` line belongs under [covered]")
            rung = int(marker.group(1))
            continue

        entry, _, reason = (part.strip() for part in line.partition("#"))
        if section is None:
            raise ValueError(f"line {number}: entry before any section header: {entry!r}")
        if SEPARATOR not in entry:
            raise ValueError(f"line {number}: entry is not `<file>{SEPARATOR}<heading>`: {entry!r}")
        if entry in covered or entry in excluded:
            raise ValueError(f"line {number}: duplicate entry: {entry!r}")

        if section == "covered":
            if rung is None:
                raise ValueError(f"line {number}: no `rung <n>` line above {entry!r}")
            covered[entry] = rung
        else:
            if not reason:
                raise ValueError(f"line {number}: [excluded] entry needs a `# reason`: {entry!r}")
            excluded[entry] = reason

    return Manifest(covered=covered, excluded=excluded)


def skill_headings(skill_dir: Path) -> list[str]:
    """Every `##`/`###` heading in a skill directory, as `<file> :: <heading>`.

    Headings inside a fenced code block are skipped: a `#` there is a comment or
    a CSS at-rule, not a section of the guide.
    """
    headings: list[str] = []
    for path in sorted(skill_dir.glob("*.md")):
        in_fence = False
        for raw in path.read_text(encoding="utf-8").splitlines():
            if _FENCE.match(raw):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            match = _HEADING.match(raw)
            if match:
                headings.append(f"{path.name}{SEPARATOR}{match.group(1)}")
    return headings


def parse_rungs(text: str) -> dict[int, str]:
    """The numbered rungs an instructions file states, as ``number -> subject``.

    A repeated number raises rather than overwriting. Two rungs numbered alike
    would drop one of them out of the comparison entirely while the numbering
    stayed a contiguous sequence, so the manifest would look complete with a
    rule missing from it.
    """
    rungs: dict[int, str] = {}
    in_fence = False
    for number, raw in enumerate(text.splitlines(), start=1):
        if _FENCE.match(raw):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _RUNG.match(raw)
        if not match:
            continue
        rung = int(match.group(1))
        if rung in rungs:
            raise ValueError(f"line {number}: rung {rung} is numbered twice")
        subject = _SUBJECT.match(match.group(2))
        rungs[rung] = subject.group(1) if subject else match.group(2)
    return rungs


def instruction_rungs(path: Path) -> dict[int, str]:
    """``parse_rungs`` over an instructions file on disk."""
    return parse_rungs(path.read_text(encoding="utf-8"))


def misnumbered_rungs(rungs: dict[int, str]) -> list[int]:
    """The numbers missing from ``1..n``, sorted.

    A gap means a rung was dropped or renumbered, and a stray top-level numbered
    list elsewhere in the file shows up the same way. Either one makes the rung
    keys in the manifest mean something other than what they meant when it was
    written, so both are worth failing on.
    """
    return [] if not rungs else sorted(set(range(1, max(rungs) + 1)) - set(rungs))


def unkeyable_headings(headings: list[str]) -> list[str]:
    """Headings the manifest cannot key on, sorted.

    Two sections of one file sharing a name collide, and the entry format reads
    `#` as the start of a reason trailer and `::` as the file separator, so a
    heading containing either cannot be written down unambiguously. Renaming the
    heading is the fix in all three cases.
    """
    unkeyable = {h for h in headings if "#" in h or h.count(SEPARATOR) != 1}
    return sorted(unkeyable | set(_duplicates(headings)))


def _duplicates(headings: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for heading in headings:
        if heading in seen:
            duplicates.add(heading)
        seen.add(heading)
    return sorted(duplicates)


def unaccounted(headings: list[str], manifest: Manifest) -> list[str]:
    """Skill headings the manifest classifies neither way, sorted."""
    return sorted(set(headings) - manifest.entries)


def stale(headings: list[str], manifest: Manifest) -> list[str]:
    """Manifest entries naming a heading the skill no longer has, sorted."""
    return sorted(manifest.entries - set(headings))


def unknown_rungs(manifest: Manifest, rungs: dict[int, str]) -> list[int]:
    """Rungs the manifest names that the instructions do not state, sorted."""
    return sorted(set(manifest.covered.values()) - set(rungs))


def uncited_rungs(manifest: Manifest, rungs: dict[int, str]) -> list[int]:
    """Rungs the instructions state that no covered heading maps to, sorted."""
    return sorted(set(rungs) - set(manifest.covered.values()))
