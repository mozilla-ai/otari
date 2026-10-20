#!/usr/bin/env python3
"""Move deploy/railway to a new release's image tag.

Rewrites ``mzdotai/otari:<pinned>`` to the new version in template.json,
listing.md and README.md, plus the README's example versions: the upgrade
example names the next minor release and the auto-update example the new
version's minor line (``0.28.0`` to ``0.28.1``). Every file is rewritten as
text, so template.json keeps its formatting, and none is written unless every
pattern was found exactly as often as expected, so a reworded README fails the
run instead of half-bumping.

A version not above the pinned one changes nothing: a patch to an older minor
line must not move new deploys backward. ``--pending-template`` names the
template.json of an open bump PR, whose pin counts the same way, so an older
release cannot overwrite a newer one still waiting to merge.

Standard library only, so the release workflow runs it with no environment.
Prints ``changed=`` and ``previous=`` lines for ``$GITHUB_OUTPUT`` on stdout
and diagnostics on stderr. Exits 0 when bumped or already current, 1 on error.

Usage:
    python scripts/bump_railway_template.py 0.29.0
    python scripts/bump_railway_template.py 0.29.0 --pending-template pr-template.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RAILWAY_DIR = Path("deploy") / "railway"
IMAGE = "mzdotai/otari"
VERSION_RE = re.compile(r"v?(\d+)\.(\d+)\.(\d+)")
SEMVER = r"\d+\.\d+\.\d+"
UPGRADE_EXAMPLE_RE = re.compile(rf"(for example\s+`){SEMVER}(`)")
PATCH_EXAMPLE_RE = re.compile(rf"\(`{SEMVER}` to `{SEMVER}`\)")

Version = tuple[int, int, int]


class BumpError(Exception):
    """The template could not be bumped."""


def parse_version(text: str) -> Version:
    """Parse ``X.Y.Z`` or ``vX.Y.Z``; a prerelease or anything else is refused."""
    match = VERSION_RE.fullmatch(text.strip())
    if not match:
        raise BumpError(f"{text!r} is not a release version (X.Y.Z)")
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def _format(version: Version) -> str:
    return ".".join(str(part) for part in version)


def pinned_version(template: dict[str, Any]) -> str:
    """Return the version the snapshot's one Otari image is pinned to."""
    prefix = f"{IMAGE}:"
    images: set[str] = {
        service["source"]["image"]
        for service in template["services"].values()
        if ((service.get("source") or {}).get("image") or "").startswith(prefix)
    }
    if len(images) != 1:
        raise BumpError(f"template.json should pin one {IMAGE} image, found {sorted(images) or 'none'}")
    return images.pop().removeprefix(prefix)


def _sub(pattern: re.Pattern[str], repl: str, text: str, where: str) -> str:
    new_text, count = pattern.subn(repl, text)
    if not count:
        raise BumpError(f"{where}: found no match for {pattern.pattern!r}")
    return new_text


def _sub_once(pattern: re.Pattern[str], repl: str, text: str, where: str) -> str:
    """Replace the one match, refusing text that holds none or several.

    The example patterns match a shape of prose rather than the pinned version,
    so a second one would be rewritten without anybody having asked.
    """
    new_text, count = pattern.subn(repl, text)
    if not count:
        raise BumpError(f"{where}: found no match for {pattern.pattern!r}")
    if count > 1:
        raise BumpError(f"{where}: found {count} matches for {pattern.pattern!r}, expected one")
    return new_text


def bump_texts(texts: dict[str, str], old: str, new: Version) -> dict[str, str]:
    """Return each file's text moved from ``old`` to ``new``, keyed like ``texts``."""
    image_re = re.compile(rf"{re.escape(f'{IMAGE}:{old}')}(?![\w.-])")
    new_image = f"{IMAGE}:{_format(new)}"
    bumped = {name: _sub(image_re, new_image, text, name) for name, text in texts.items()}

    major, minor, _ = new
    readme = bumped["README.md"]
    readme = _sub_once(UPGRADE_EXAMPLE_RE, rf"\g<1>{major}.{minor + 1}.0\g<2>", readme, "README.md")
    readme = _sub_once(PATCH_EXAMPLE_RE, f"(`{major}.{minor}.0` to `{major}.{minor}.1`)", readme, "README.md")
    bumped["README.md"] = readme
    return bumped


def bump(root: Path, version: str, pending: str | None = None) -> str | None:
    """Bump the files under ``root``. Return the previous version, or None when already current.

    ``pending`` is the pin of an open bump PR; a version not above it changes nothing either.
    """
    directory = root / RAILWAY_DIR
    paths = {name: directory / name for name in ("template.json", "listing.md", "README.md")}
    texts = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}

    new = parse_version(version)
    old = pinned_version(json.loads(texts["template.json"]))
    if new <= parse_version(old) or (pending is not None and new <= parse_version(pending)):
        return None

    bumped = bump_texts(texts, old, new)
    for name, path in paths.items():
        path.write_text(bumped[name], encoding="utf-8")
    return old


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="the release to pin, X.Y.Z or vX.Y.Z")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="the repository root")
    parser.add_argument("--pending-template", type=Path, help="template.json of an open bump PR")
    args = parser.parse_args(argv)

    try:
        pending = None
        if args.pending_template:
            pending = pinned_version(json.loads(args.pending_template.read_text(encoding="utf-8")))
        previous = bump(args.root, args.version, pending)
    except BumpError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if previous is None:
        where = "main or the open bump PR" if pending else "The Railway template"
        print(f"{where} already pins {args.version} or later; nothing to do.", file=sys.stderr)
        print("changed=false")
    else:
        print(f"Moved the Railway template from {previous} to {args.version.removeprefix('v')}.", file=sys.stderr)
        print("changed=true")
        print(f"previous={previous}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
