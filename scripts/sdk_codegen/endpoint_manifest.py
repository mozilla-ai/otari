"""Endpoint-coverage manifest: parsing and drift detection.

``sdk-endpoints.txt`` (next to this module) records which gateway OpenAPI
endpoints the SDKs surface. It is the canonical copy: the codegen workflow
pushes it into each SDK repo alongside the generated core, so the copies there
are generated artifacts rather than four hand-kept files.

Keeping it here means drift is caught in the repo that causes it. A gateway PR
that adds an endpoint fails ``tests/unit/test_sdk_endpoint_coverage.py`` until
the endpoint is classified, instead of silently reddening four downstream repos
whose CI happens not to run that week.

Format: ``[covered]`` / ``[excluded]`` sections, one ``METHOD /path`` per line
with an optional ``# reason`` trailer. Blank lines and ``#`` lines are ignored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path(__file__).resolve().parent / "sdk-endpoints.txt"

# Verbs that denote an API operation; OpenAPI path items also carry keys like
# "parameters" and "summary", which are not endpoints.
HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete"})

# Liveness/readiness routes are gateway plumbing, not an SDK surface, so they
# are filtered out rather than needing an [excluded] entry apiece.
_META_PREFIX = "/health"


def parse_manifest(text: str) -> tuple[set[str], set[str]]:
    """Return the ``(covered, excluded)`` endpoint sets from manifest text.

    Entries are normalized to ``METHOD /path`` with an uppercase method, so
    comparison against a spec is case-insensitive on the verb.
    """
    covered: set[str] = set()
    excluded: set[str] = set()
    section: set[str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == "[covered]":
            section = covered
            continue
        if line == "[excluded]":
            section = excluded
            continue
        entry = line.split("#", 1)[0].strip()
        if not entry or section is None:
            continue
        method, path = entry.split(None, 1)
        section.add(f"{method.upper()} {path.strip()}")
    return covered, excluded


def spec_endpoints(spec: dict[str, Any]) -> set[str]:
    """Extract ``METHOD /path`` pairs from an OpenAPI doc, dropping meta routes."""
    endpoints: set[str] = set()
    for path, operations in spec.get("paths", {}).items():
        if path == _META_PREFIX or path.startswith(f"{_META_PREFIX}/"):
            continue
        for method in operations:
            if method.lower() in HTTP_METHODS:
                endpoints.add(f"{method.upper()} {path}")
    return endpoints


def load_manifest(path: Path | None = None) -> tuple[set[str], set[str]]:
    """Read and parse the manifest, defaulting to the canonical copy."""
    return parse_manifest((path or MANIFEST_PATH).read_text(encoding="utf-8"))


def load_spec(path: Path) -> dict[str, Any]:
    """Read an OpenAPI document from ``path``."""
    spec: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return spec


def unaccounted(endpoints: set[str], covered: set[str], excluded: set[str]) -> list[str]:
    """Spec endpoints listed in neither section, sorted."""
    return sorted(endpoints - (covered | excluded))


def stale(endpoints: set[str], covered: set[str], excluded: set[str]) -> list[str]:
    """Manifest entries no longer present in the spec, sorted."""
    return sorted((covered | excluded) - endpoints)
