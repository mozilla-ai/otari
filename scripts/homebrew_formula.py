#!/usr/bin/env python3
"""Render the Homebrew formula for the otari-agent CLI from uv.lock.

Homebrew builds every Python resource from its sdist, so each ``resource``
stanza needs the sdist URL and sha256 that uv.lock already records for every
package. The closure is walked from the otari-agent entry in the lock, so the
formula and the lock cannot disagree, and a dependency that ships no sdist
fails here rather than at ``brew install``. Standard library only, like
oss_edition_smoke.py, so the release workflow runs it before any environment
exists.

Usage:
    uv run python scripts/homebrew_formula.py --version 0.9.0 \
        --sdist dist/otari_agent-0.9.0.tar.gz --output dist/otari.rb
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = REPO_ROOT / "uv.lock"
TEMPLATE_PATH = REPO_ROOT / "packaging" / "homebrew" / "otari.rb.tmpl"
PACKAGE = "otari-agent"
RELEASE_URL = "https://github.com/mozilla-ai/otari/releases/download/v{version}/{filename}"

Package = dict[str, Any]


class FormulaError(Exception):
    """The lock or the template cannot produce a formula."""


def normalize(name: str) -> str:
    """PEP 503 name normalization, so lock entries and dependency edges compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def closure(lock: dict[str, Any], root: str = PACKAGE) -> dict[str, Package]:
    """Every package the root depends on, transitively, by normalized name; the root itself excluded."""
    by_name: dict[str, list[Package]] = {}
    for package in lock["package"]:
        by_name.setdefault(normalize(package["name"]), []).append(package)

    resolved: dict[str, Package] = {}
    pending = [normalize(root)]
    while pending:
        name = pending.pop()
        if name in resolved:
            continue
        entries = by_name.get(name)
        if not entries:
            raise FormulaError(f"{name} is not in {LOCK_PATH.name}")
        if len(entries) > 1:
            versions = ", ".join(sorted(entry["version"] for entry in entries))
            raise FormulaError(f"{name} resolves to more than one version in the lock ({versions}); a formula pins one")
        resolved[name] = entries[0]
        pending.extend(normalize(dependency["name"]) for dependency in entries[0].get("dependencies", []))

    del resolved[normalize(root)]
    return dict(sorted(resolved.items()))


def resource_stanzas(packages: dict[str, Package]) -> str:
    """One Homebrew ``resource`` block per package, from the lock's sdist entry."""
    stanzas = []
    for name, package in packages.items():
        sdist = package.get("sdist")
        if not sdist:
            raise FormulaError(
                f"{name}=={package['version']} has no sdist in the lock. Homebrew builds every resource "
                "from source, so a wheel-only dependency cannot ship in the formula."
            )
        digest = str(sdist["hash"]).removeprefix("sha256:")
        stanzas.append(f'  resource "{name}" do\n    url "{sdist["url"]}"\n    sha256 "{digest}"\n  end')
    return "\n\n".join(stanzas)


def render(template: str, **fields: str) -> str:
    """Fill every ``{{name}}`` placeholder; a placeholder left over is an error, not a blank."""
    rendered = template
    for key, value in fields.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    leftover = re.findall(r"\{\{(\w+)}}", rendered)
    if leftover:
        raise FormulaError(f"template placeholder(s) not filled: {', '.join(sorted(set(leftover)))}")
    return rendered


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_formula(
    *, version: str, sdist: Path, lock_path: Path = LOCK_PATH, template_path: Path = TEMPLATE_PATH
) -> str:
    """The formula text for one release: the sdist asset the release carries, plus the lock's closure."""
    with lock_path.open("rb") as handle:
        lock = tomllib.load(handle)
    return render(
        template_path.read_text(encoding="utf-8"),
        version=version,
        sdist_url=RELEASE_URL.format(version=version, filename=sdist.name),
        sdist_sha256=sha256_of(sdist),
        resources=resource_stanzas(closure(lock)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="The release version without the leading v, e.g. 0.9.0")
    parser.add_argument("--sdist", required=True, type=Path, help="The built otari_agent-<version>.tar.gz to hash")
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--template", type=Path, default=TEMPLATE_PATH)
    parser.add_argument("--output", type=Path, default=None, help="Where to write the formula (default: stdout)")
    args = parser.parse_args(argv)

    try:
        formula = build_formula(
            version=args.version, sdist=args.sdist, lock_path=args.lock, template_path=args.template
        )
    except (FormulaError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.output is None:
        sys.stdout.write(formula)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(formula, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
