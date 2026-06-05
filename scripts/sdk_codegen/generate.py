#!/usr/bin/env python3
"""Generate language SDK control-plane clients from the otari OpenAPI spec.

This produces typed clients for the gateway's management / control-plane endpoints
(API keys, users, budgets, pricing, usage) using OpenAPI Generator.

Intentionally excluded:
  - inference / proxy (chat, responses, embeddings, messages, moderations,
    rerank, audio, images, models, health): the published SDKs wrap the official
    OpenAI SDK, the spec leaves them loosely typed, and OpenAPI Generator cannot
    generate their streaming (SSE) responses.
  - batches: responses are untyped in the spec (the route uses
    ``response_model=None``), so generation would regress them, and every SDK
    already implements batches with hand-written typed models.

So only the typed, bespoke management surface is generated; everything else
stays hand-written.

Usage:
    python scripts/sdk_codegen/generate.py --language python --out-dir dist
    python scripts/sdk_codegen/generate.py --language all

The OpenAPI Generator CLI must be on PATH (or set OPENAPI_GENERATOR_CLI). It
requires a JRE. Install with:
    npm install -g @openapitools/openapi-generator-cli
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SPEC = REPO_ROOT / "docs" / "public" / "openapi.json"
DEFAULT_OUT_DIR = REPO_ROOT / "dist" / "sdk-codegen"

# Operations carrying one of these tags form the control plane we generate: the
# management endpoints whose responses are fully typed in the spec. See the
# module docstring for what is excluded and why (notably batches, whose
# responses are untyped in the spec and already hand-written in every SDK).
CONTROL_PLANE_TAGS: frozenset[str] = frozenset(
    {"keys", "users", "budgets", "pricing", "usage"}
)


@dataclass(frozen=True)
class LanguageTarget:
    """How to generate a given language and where the result is published."""

    generator: str
    additional_properties: str
    sdk_repo: str
    target_path: str


# target_path is where the generated client project is dropped inside the SDK
# repo by the codegen workflow. Wiring the generated client into each SDK's
# public surface is a per-repo follow-up.
TARGETS: dict[str, LanguageTarget] = {
    "python": LanguageTarget(
        generator="python",
        additional_properties="packageName=otari_control_plane",
        sdk_repo="mozilla-ai/otari-sdk-python",
        target_path="src/otari/_generated",
    ),
    "typescript": LanguageTarget(
        generator="typescript-fetch",
        additional_properties="npmName=otari-control-plane,supportsES6=true",
        sdk_repo="mozilla-ai/otari-sdk-ts",
        target_path="src/generated",
    ),
    "go": LanguageTarget(
        generator="go",
        additional_properties="packageName=generated,withGoMod=false",
        sdk_repo="mozilla-ai/otari-sdk-go",
        target_path="otari/generated",
    ),
    "rust": LanguageTarget(
        generator="rust",
        additional_properties="packageName=otari-control-plane,library=reqwest",
        sdk_repo="mozilla-ai/otari-sdk-rust",
        target_path="src/generated",
    ),
}


def _collect_schema_refs(node: Any, acc: set[str]) -> None:
    """Recursively collect every ``#/components/schemas/NAME`` reference."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
            acc.add(ref.rsplit("/", 1)[-1])
        for value in node.values():
            _collect_schema_refs(value, acc)
    elif isinstance(node, list):
        for item in node:
            _collect_schema_refs(item, acc)


def filter_spec(spec: dict[str, Any], tags: frozenset[str]) -> dict[str, Any]:
    """Return a copy of ``spec`` with only control-plane operations and schemas.

    Keeps operations whose ``tags`` intersect ``tags`` and the transitive closure
    of component schemas they reference, so the result is a self-contained,
    valid OpenAPI document.
    """
    paths: dict[str, Any] = spec.get("paths", {})
    kept_paths: dict[str, Any] = {}
    for path, methods in paths.items():
        kept_methods: dict[str, Any] = {}
        for method, operation in methods.items():
            if not isinstance(operation, dict):
                continue
            op_tags = operation.get("tags", [])
            if isinstance(op_tags, list) and tags.intersection(op_tags):
                kept_methods[method] = operation
        if kept_methods:
            kept_paths[path] = kept_methods

    all_schemas: dict[str, Any] = spec.get("components", {}).get("schemas", {})
    needed: set[str] = set()
    _collect_schema_refs(kept_paths, needed)
    queue: list[str] = list(needed)
    while queue:
        name = queue.pop()
        schema = all_schemas.get(name)
        if schema is None:
            continue
        found: set[str] = set()
        _collect_schema_refs(schema, found)
        for ref_name in found:
            if ref_name not in needed:
                needed.add(ref_name)
                queue.append(ref_name)

    kept_schemas = {name: all_schemas[name] for name in sorted(needed) if name in all_schemas}

    filtered: dict[str, Any] = {key: value for key, value in spec.items() if key not in {"paths", "components"}}
    filtered["paths"] = kept_paths
    components = {key: value for key, value in spec.get("components", {}).items() if key != "schemas"}
    components["schemas"] = kept_schemas
    filtered["components"] = components
    return filtered


def _generator_cli() -> str:
    return os.environ.get("OPENAPI_GENERATOR_CLI", "openapi-generator-cli")


def generate_language(language: str, spec_path: Path, out_dir: Path) -> Path:
    """Run OpenAPI Generator for ``language`` and return the output directory."""
    target = TARGETS[language]
    dest = out_dir / language
    cmd = [
        _generator_cli(),
        "generate",
        "-i",
        str(spec_path),
        "-g",
        target.generator,
        "-o",
        str(dest),
        "--additional-properties",
        target.additional_properties,
    ]
    subprocess.run(cmd, check=True)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate control-plane SDK clients.")
    parser.add_argument("--language", choices=[*TARGETS, "all"], default="all")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--write-filtered-spec",
        type=Path,
        default=None,
        help="Also write the control-plane-only spec to this path (for inspection).",
    )
    args = parser.parse_args()

    spec: dict[str, Any] = json.loads(Path(args.spec).read_text())
    filtered = filter_spec(spec, CONTROL_PLANE_TAGS)
    languages = list(TARGETS) if args.language == "all" else [str(args.language)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.write_filtered_spec is not None:
        Path(args.write_filtered_spec).write_text(json.dumps(filtered, indent=2))

    with tempfile.TemporaryDirectory() as tmp:
        filtered_path = Path(tmp) / "control-plane.openapi.json"
        filtered_path.write_text(json.dumps(filtered, indent=2))
        for language in languages:
            dest = generate_language(language, filtered_path, out_dir)
            print(f"generated {language} -> {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
