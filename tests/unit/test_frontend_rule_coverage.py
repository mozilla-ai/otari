"""Drift gate for the frontend-standards skill against its CodeRabbit instructions.

The skill is the long form, written for whoever is editing the dashboard; the
instructions file is the short form the review bot loads. Nothing compared them
until this, and four rules had drifted apart in a single day's work: two stated
in the skill that the bot had never been able to enforce, one that an amendment
reached in both layers only because the person doing it went looking, and one
followed consistently and written down in neither.

A prose comparison would fight the compression on purpose in the pair (one rung
carries several sections, phrased as what is a finding rather than as how to
write code), so the check is structural: every heading is classified, and every
classification still names something that exists. All three files are read from
disk, so the result depends only on the commit under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "rule_coverage"))

import rule_manifest as rc  # noqa: E402

MANIFEST = rc.FRONTEND_MANIFEST.relative_to(rc.REPO_ROOT)


@pytest.fixture(scope="module")
def manifest() -> rc.Manifest:
    return rc.parse_manifest(rc.FRONTEND_MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def headings() -> list[str]:
    return rc.skill_headings(rc.FRONTEND_SKILL_DIR)


@pytest.fixture(scope="module")
def rungs() -> dict[int, str]:
    return rc.instruction_rungs(rc.FRONTEND_INSTRUCTIONS)


def test_skill_headings_are_usable_as_keys(headings: list[str]) -> None:
    """The heading text is the manifest's key, so it has to be unique and writable."""
    assert headings, "no headings found; did the skill directory move?"
    unkeyable = rc.unkeyable_headings(headings)
    assert not unkeyable, (
        f"these skill headings cannot be manifest keys: {unkeyable}. Either two sections of one "
        f"file share the heading, or it contains `#` (which opens a reason trailer) or "
        f"`{rc.SEPARATOR.strip()}` (the file separator). Rename the heading."
    )


def test_a_repeated_rung_number_is_rejected() -> None:
    """Two rungs numbered alike would drop one out of the comparison silently.

    The numbering stays a contiguous sequence either way, so nothing else here
    would notice; the manifest would simply look complete with a rule missing
    from it.
    """
    with pytest.raises(ValueError, match="rung 11 is numbered twice"):
        rc.parse_rungs("11. **One rule.** Body.\n11. **Another.** Body.\n")


def test_rungs_are_numbered_without_gaps(rungs: dict[int, str]) -> None:
    """The manifest keys on a rung's number, so the numbering has to stay a sequence."""
    missing = rc.misnumbered_rungs(rungs)
    assert not missing, (
        f"{rc.FRONTEND_INSTRUCTIONS.relative_to(rc.REPO_ROOT)} numbers its rungs "
        f"{sorted(rungs)}, skipping {missing}. A gap means a rung was dropped or renumbered "
        f"under the entries in {MANIFEST} that name it, so re-point them and close the gap."
    )


def test_every_skill_heading_is_classified(manifest: rc.Manifest, headings: list[str]) -> None:
    missing = rc.unaccounted(headings, manifest)
    assert not missing, (
        f"the frontend-standards skill has {len(missing)} heading(s) absent from {MANIFEST}: "
        f"{missing}. Add each under [covered] beneath the rung of "
        ".github/instructions/frontend-standards.instructions.md that carries it, or under "
        "[excluded] with a reason if a reviewer needs nothing from it. [excluded] is the "
        "common answer and is fine; the value is that the answer is recorded rather than "
        "assumed."
    )


def test_manifest_has_no_stale_entries(manifest: rc.Manifest, headings: list[str]) -> None:
    """An entry naming a heading that is gone is a leftover, not a deferral.

    The manifest and the skill are in the same commit, so a mismatch is always an
    editing slip and always fixable in the PR that caused it.
    """
    leftovers = rc.stale(headings, manifest)
    assert not leftovers, (
        f"{MANIFEST} names {len(leftovers)} heading(s) the skill no longer has: {leftovers}. "
        "Renaming a heading reads here as a delete plus an add, so update the entry to match "
        "or drop it."
    )


def test_manifest_names_only_rungs_that_exist(manifest: rc.Manifest, rungs: dict[int, str]) -> None:
    unknown = rc.unknown_rungs(manifest, rungs)
    assert not unknown, (
        f"{MANIFEST} maps headings to rung(s) {unknown}, which "
        ".github/instructions/frontend-standards.instructions.md does not state. Renumbering "
        "the rungs moves every entry under them, so re-point those headings or restore the rung."
    )


def test_every_rung_carries_a_skill_heading(manifest: rc.Manifest, rungs: dict[int, str]) -> None:
    """The gap in the other direction: a rule the bot enforces that the skill never states.

    That is the shape nobody goes looking for, because the review comment reads
    as authoritative on its own.
    """
    uncited = rc.uncited_rungs(manifest, rungs)
    assert not uncited, (
        "rung(s) "
        + str([f"{number}: {rungs[number]}" for number in uncited])
        + f" in .github/instructions/frontend-standards.instructions.md are named by no entry "
        f"in {MANIFEST}. Either the skill states the rule somewhere and the heading belongs "
        "under that rung, or the bot is enforcing something the skill never told anyone."
    )
