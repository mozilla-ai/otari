"""Build the GitBook site for the documentation into site/.

The site holds the pages that docs/SUMMARY.md lists, and every page under docs/ must be on that menu.
A relative link to any other file in the repository becomes a GitHub link at the published ref.
The pages keep their paths under docs/, so links between published pages need no rewriting.
The site also carries the OpenAPI specification and a generated endpoint page per tag under the API reference.
GitBook renders each operation on those pages from the specification registered in it under SPEC_NAME.

Standard library only, so the publish workflow runs it without installing the project.

Usage:
    python scripts/prepare_gitbook_site.py [--ref REF] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
REPO = "mozilla-ai/otari"
REPO_URL = f"https://github.com/{REPO}"
SUMMARY = "SUMMARY.md"
SPEC = "public/openapi.json"
SPEC_NAME = "otari-openapi-spec"
SPEC_SITE_PATH = "api/openapi.json"
SPEC_URL = f"https://raw.githubusercontent.com/{REPO}/gitbook-docs/{SPEC_SITE_PATH}"
ENDPOINTS_DIR = "api-endpoints"
ENDPOINTS_PARENT = "api-reference.md"

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
# Words of a page name that a title spells other than in sentence case.
_ACRONYMS = {"mcp": "MCP", "otel": "OTel"}
# Reads come before writes, which is the order an endpoint page lists them in.
_HTTP_METHODS = ("get", "head", "options", "post", "put", "patch", "delete", "trace")
_RELEASE_TAG = re.compile(r"^v(\d+\.\d+\.\d+)$")
_NOT_A_SLUG = re.compile(r"[^a-z0-9]+")


def summary_pages(summary: str) -> list[str]:
    """Return the page paths a GitBook SUMMARY.md lists, in menu order."""
    return [match.group(1) for line in summary.splitlines() if (match := _SUMMARY_ENTRY.match(line))]


def pages_off_the_menu(docs_dir: Path, published: frozenset[str]) -> list[str]:
    """Return the Markdown pages under docs_dir that the menu does not list."""
    pages = (path.relative_to(docs_dir).as_posix() for path in docs_dir.rglob("*.md"))
    return sorted(page for page in pages if page != SUMMARY and page not in published)


def operations_by_page(spec: dict[str, Any]) -> dict[str, list[tuple[str, str]]]:
    """Return the (path, method) pairs of each endpoint page, keyed by page name and sorted.

    An operation goes on the page named for its first tag, or on "other" when it has none.
    """
    by_page: dict[str, list[tuple[str, str]]] = {}
    for path, item in spec.get("paths", {}).items():
        for method, operation in item.items():
            if method not in _HTTP_METHODS:
                continue
            tags = operation.get("tags") or ["other"]
            page = _NOT_A_SLUG.sub("-", tags[0].lower()).strip("-") or "other"
            by_page.setdefault(page, []).append((path, method))
    return {page: sorted(by_page[page], key=_path_then_method) for page in sorted(by_page)}


def _path_then_method(operation: tuple[str, str]) -> tuple[str, int]:
    path, method = operation
    return path, _HTTP_METHODS.index(method)


def page_title(page: str) -> str:
    """Return the title of an endpoint page in sentence case, with acronyms in their usual form."""
    words = [_ACRONYMS.get(word, word) for word in page.split("-")]
    return " ".join([words[0][:1].upper() + words[0][1:], *words[1:]])


def endpoint_page(page: str, operations: list[tuple[str, str]]) -> str:
    """Return the Markdown page that renders operations from the registered specification."""
    blocks = (
        f'{{% openapi-operation spec="{SPEC_NAME}" path="{path}" method="{method}" %}}\n'
        f"[OpenAPI {SPEC_NAME}]({SPEC_URL})\n"
        "{% endopenapi-operation %}\n"
        for path, method in operations
    )
    return f"# {page_title(page)}\n\n" + "\n".join(blocks)


def nest_endpoint_pages(summary: str, pages: list[str]) -> str:
    """Return summary with an entry per endpoint page nested under the API reference entry, after its own children."""
    lines = summary.split("\n")
    for index, line in enumerate(lines):
        if (match := _SUMMARY_ENTRY.match(line)) and match.group(1) == ENDPOINTS_PARENT:
            depth = len(line) - len(line.lstrip())
            end = index + 1
            while end < len(lines) and lines[end].strip() and len(lines[end]) - len(lines[end].lstrip()) > depth:
                end += 1
            indent = line[:depth] + "  "
            entries = [f"{indent}* [{page_title(page)}]({ENDPOINTS_DIR}/{page}.md)" for page in pages]
            return "\n".join([*lines[:end], *entries, *lines[end:]])
    raise ValueError(f"{SUMMARY} lists no {ENDPOINTS_PARENT!r} to nest the endpoint pages under")


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
        self._build_endpoint_reference(out_dir)

    def _build_endpoint_reference(self, out_dir: Path) -> None:
        spec = json.loads((self.docs_dir / SPEC).read_text(encoding="utf-8"))
        if release := _RELEASE_TAG.match(self.ref):
            spec["info"]["version"] = release.group(1)
        by_page = operations_by_page(spec)

        spec_dest = out_dir / SPEC_SITE_PATH
        spec_dest.parent.mkdir(parents=True, exist_ok=True)
        spec_dest.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if (out_dir / ENDPOINTS_DIR).exists():
            raise ValueError(f"docs/{ENDPOINTS_DIR}/ is reserved for the generated endpoint pages")
        (out_dir / ENDPOINTS_DIR).mkdir()
        for page, operations in by_page.items():
            (out_dir / ENDPOINTS_DIR / f"{page}.md").write_text(endpoint_page(page, operations), encoding="utf-8")
        summary = out_dir / SUMMARY
        summary.write_text(nest_endpoint_pages(summary.read_text(encoding="utf-8"), list(by_page)), encoding="utf-8")


def main() -> int:
    """Build the site and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default="main", help="Git ref that links to unpublished files point at.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "site", help="Directory to write the site to.")
    args = parser.parse_args()

    site = Site.from_summary(DOCS_DIR, args.ref)
    if off_menu := pages_off_the_menu(DOCS_DIR, site.published):
        print(
            f"Pages under docs/ that docs/{SUMMARY} does not list: {', '.join(off_menu)}",
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
