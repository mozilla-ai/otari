"""Check that every API path the docs spell out is a route the gateway serves.

A reader copies a path out of the docs, so one that names no route is a 404 they
debug themselves. ``docs/api-reference.md`` once named ``/api/v1/key``, a router
prefix with no route of its own, as where a key reads its spend (#3030). Each
backticked ``/api/v1/...`` path under ``docs/`` must therefore be a path in
``docs/public/openapi.json``, end in ``*`` to name a family of routes, or be one
of the exceptions below, which only shrink.
"""

import json
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DOCS_DIR = _REPO / "docs"
_SPEC = _DOCS_DIR / "public" / "openapi.json"

_API_PATH = re.compile(r"`(?:(?:GET|POST|PUT|PATCH|DELETE) )?(/api/v1/[^`\s?]*)(?:\?[^`\s]*)?`")
_PARAM = re.compile(r"\{[^}]+\}")

# Paths the docs name on purpose although the spec has no route for them.
_NOT_IN_SPEC = {
    "/api/v1/openapi.json": "The spec itself, which the spec does not list.",
    "/api/v1/docs": "The interactive API docs page, outside the spec.",
    "/api/v1/admin": "Named as a path that answers 404.",
    "/api/v1/provider-credentials": "Named as a retired page.",
}


def _normalize(path: str) -> str:
    return _PARAM.sub("{}", path).rstrip("/")


def _documented_paths() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for doc in sorted(_DOCS_DIR.rglob("*.md")):
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for path in _API_PATH.findall(line):
                found.setdefault(_normalize(path), []).append(f"{doc.relative_to(_DOCS_DIR)}:{lineno}")
    return found


def test_documented_api_paths_are_served() -> None:
    served = {_normalize(path) for path in json.loads(_SPEC.read_text(encoding="utf-8"))["paths"]}
    unknown = {
        path: where
        for path, where in _documented_paths().items()
        if path not in served and path not in _NOT_IN_SPEC and not path.endswith("*")
    }

    assert not unknown, f"Docs name API paths the gateway does not serve: {unknown}"


def test_exceptions_are_still_named() -> None:
    stale = _NOT_IN_SPEC.keys() - _documented_paths().keys()

    assert not stale, f"{sorted(stale)} no longer appear in docs/; remove them from _NOT_IN_SPEC."
