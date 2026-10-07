"""Build the GitBook site for the documentation into site/.

The site holds the pages that docs/SUMMARY.md lists, so the menu decides what is published.
A relative link to any other file in the repository becomes a GitHub link at the published ref.
The pages keep their paths under docs/, so links between published pages need no rewriting.

Standard library only, so the publish workflow runs it without installing the project.

Usage:
    python scripts/prepare_gitbook_site.py [--ref REF] [--out DIR]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
REPO_URL = "https://github.com/mozilla-ai/otari"
SUMMARY = "SUMMARY.md"

# Pages under docs/ that are written for contributors to this repository, not for users of Otari.
UNPUBLISHED_PAGES = frozenset({"domains.md"})

# Repository-root files copied to the root of the site, keyed by their name in the repository.
BRANCH_FILES = {
    ".gitbook-branch-readme.md": "README.md",
    ".gitbook.yaml": ".gitbook.yaml",
}

_SUMMARY_ENTRY = re.compile(r"^\s*[*-]\s+\[[^\]]*\]\(([^)\s]+)\)")
# NOTE: A code span and a link are one alternation, so a code span inside a link label stays part of the link.
_CODE_OR_LINK = re.compile(r"(?P<code>`+[^`]*`+)|(?P<label>\[[^\]]*\])\((?P<target>[^)\s]+)(?P<title>\s+\"[^\"]*\")?\)")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_NOT_A_PATH = ("http://", "https://", "mailto:", "#")


def summary_pages(summary: str) -> list[str]:
    """Return the page paths a GitBook SUMMARY.md lists, in menu order."""
    return [match.group(1) for line in summary.splitlines() if (match := _SUMMARY_ENTRY.match(line))]


def unclassified_pages(docs_dir: Path, published: frozenset[str]) -> list[str]:
    """Return the Markdown pages under docs_dir that are neither published nor marked unpublished."""
    pages = (path.relative_to(docs_dir).as_posix() for path in docs_dir.rglob("*.md"))
    return sorted(page for page in pages if page != SUMMARY and page not in published | UNPUBLISHED_PAGES)


@dataclass(frozen=True)
class Site:
    """The pages of one GitBook site build and the ref its outbound links point at."""

    docs_dir: Path
    published: frozenset[str]
    ref: str

    @classmethod
    def from_summary(cls, docs_dir: Path, ref: str) -> Site:
        """Read the published pages from the SUMMARY.md in docs_dir."""
        pages = summary_pages((docs_dir / SUMMARY).read_text(encoding="utf-8"))
        return cls(docs_dir=docs_dir, published=frozenset(pages), ref=ref)

    def rewrite_links(self, text: str, page: str) -> str:
        """Return text with each link to an unpublished file replaced by its GitHub URL.

        Links inside fenced code blocks and inline code spans are left alone.
        """
        lines: list[str] = []
        fence = ""
        for line in text.split("\n"):
            if marker := _FENCE.match(line):
                if not fence:
                    fence = marker.group(1)
                elif marker.group(1).startswith(fence) and not line.strip().strip(fence[0]):
                    fence = ""
                lines.append(line)
                continue
            if fence:
                lines.append(line)
                continue
            lines.append(_CODE_OR_LINK.sub(lambda match: self._rewrite(match, page), line))
        return "\n".join(lines)

    def _rewrite(self, match: re.Match[str], page: str) -> str:
        label, target, title = match.group("label"), match.group("target"), match.group("title") or ""
        if match.group("code") or target.startswith(_NOT_A_PATH):
            return match.group(0)
        path, _, anchor = target.partition("#")
        repo_root = self.docs_dir.parent.resolve()
        resolved = (self.docs_dir / page).parent.joinpath(path).resolve()
        if not resolved.is_relative_to(repo_root):
            raise ValueError(f"{page}: link {target!r} leaves the repository")
        if not resolved.exists():
            raise ValueError(f"{page}: link {target!r} names no file")
        if resolved.is_relative_to(self.docs_dir.resolve()):
            if resolved.relative_to(self.docs_dir.resolve()).as_posix() in self.published:
                return match.group(0)
        kind = "tree" if resolved.is_dir() else "blob"
        url = f"{REPO_URL}/{kind}/{self.ref}/{PurePosixPath(resolved.relative_to(repo_root))}"
        if anchor:
            url = f"{url}#{anchor}"
        return f"{label}({url}{title})"

    def build(self, out_dir: Path) -> None:
        """Write the site to out_dir, replacing anything already there."""
        if out_dir.exists():
            shutil.rmtree(out_dir)
        for page in sorted(self.published | {SUMMARY}):
            source = self.docs_dir / page
            if not source.is_file():
                raise ValueError(f"{SUMMARY} lists {page!r}, which is not a file under docs/")
            dest = out_dir / page
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(self.rewrite_links(source.read_text(encoding="utf-8"), page), encoding="utf-8")
        for name, dest_name in BRANCH_FILES.items():
            shutil.copyfile(self.docs_dir.parent / name, out_dir / dest_name)


def main() -> int:
    """Build the site and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default="main", help="Git ref that links to unpublished files point at.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "site", help="Directory to write the site to.")
    args = parser.parse_args()

    site = Site.from_summary(DOCS_DIR, args.ref)
    if unclassified := unclassified_pages(DOCS_DIR, site.published):
        print(
            f"Pages neither listed in docs/{SUMMARY} nor marked unpublished: {', '.join(unclassified)}",
            file=sys.stderr,
        )
        return 1
    try:
        site.build(args.out)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Built {len(site.published)} pages into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
