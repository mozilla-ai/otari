#!/usr/bin/env python3
"""Generate language SDK clients from the otari OpenAPI spec via OpenAPI Generator.

Two modes:

- ``control-plane`` (default): typed clients for the management endpoints only
  (keys, users, budgets, pricing, usage). The inference surface stays a
  hand-written wrapper around the official OpenAI SDK; batches are hand-written
  (their responses are untyped in the spec).

- ``full`` (Option C): enriches the spec's inference surface with the real typed
  completion schemas (from ``any-llm``), then generates a typed core covering
  *every* endpoint. The SDK hand-writes a thin shell over this core for the
  three things generation can't do: streaming (SSE), typed error mapping, and
  ergonomic method names / auth modes.

Usage:
    python scripts/sdk_codegen/generate.py --language python --out-dir dist
    python scripts/sdk_codegen/generate.py --language all --mode full

The OpenAPI Generator CLI must be on PATH (or set OPENAPI_GENERATOR_CLI). It
requires a JRE. Install with:
    npm install -g @openapitools/openapi-generator-cli
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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


# target_path is where the generated client is dropped inside the SDK repo by
# the codegen workflow (1:1 copy of the normalized per-language output). These
# match the layout each SDK's hand-written wiring imports from.
TARGETS: dict[str, LanguageTarget] = {
    "python": LanguageTarget(
        generator="python",
        # Dotted package so internal imports are ``otari._control_plane.*`` and
        # the client is importable as a subpackage of the SDK's ``otari`` package.
        additional_properties="packageName=otari._control_plane",
        sdk_repo="mozilla-ai/otari-sdk-python",
        target_path="src/otari/_control_plane",
    ),
    "typescript": LanguageTarget(
        generator="typescript-fetch",
        # No npmName: that makes typescript-fetch emit a full npm project (sources
        # under src/). Without it the output is flat (runtime.ts, apis/, models/),
        # which is what the SDK imports from under src/_control_plane.
        additional_properties="supportsES6=true",
        sdk_repo="mozilla-ai/otari-sdk-ts",
        target_path="src/_control_plane",
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
        target_path="control-plane",
    ),
}

# Option C: the FULL client (every endpoint, typed core). The SDK hand-writes a
# thin shell over this for streaming, error mapping, and ergonomic method names.
FULL_TARGETS: dict[str, LanguageTarget] = {
    "python": LanguageTarget(
        generator="python",
        additional_properties="packageName=otari._client",
        sdk_repo="mozilla-ai/otari-sdk-python",
        target_path="src/otari/_client",
    ),
    "typescript": LanguageTarget(
        generator="typescript-fetch",
        additional_properties="supportsES6=true",
        sdk_repo="mozilla-ai/otari-sdk-ts",
        target_path="src/_client",
    ),
    "go": LanguageTarget(
        generator="go",
        additional_properties="packageName=client,withGoMod=false",
        sdk_repo="mozilla-ai/otari-sdk-go",
        target_path="otari/client",
    ),
    "rust": LanguageTarget(
        generator="rust",
        additional_properties="packageName=otari-client,library=reqwest",
        sdk_repo="mozilla-ai/otari-sdk-rust",
        target_path="client",
    ),
}


# Helper that this generator version's typescript-fetch models import from
# ``runtime`` but the runtime template omits. Injected post-generation so the
# generated TypeScript compiles without a hand-edit.
_TS_MAPVALUES = """
// Added by sdk_codegen post-processing: the generated models import `mapValues`
// from runtime, but this generator version omits it.
export function mapValues(data: any, fn: (item: any) => any) {
  return Object.keys(data).reduce(
    (acc, key) => ({ ...acc, [key]: fn(data[key]) }),
    {} as Record<string, any>,
  );
}
"""


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


def enrich_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Type the inference surface so the FULL client generates typed methods.

    The gateway spec leaves chat loosely typed (``messages`` are untyped dicts,
    the chat ``200`` response has no schema). For Option C (a generated core that
    covers every endpoint) we inject the real, typed completion schemas from
    ``any-llm`` (which the gateway already depends on) so the generated chat
    method returns a typed ``ChatCompletion`` and accepts typed messages.

    Streaming still cannot be generated (OpenAPI Generator emits no SSE); the SDK
    shell hand-writes the stream iterator over the generated client.
    """
    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
    from openai.types.chat import ChatCompletionMessageParam
    from pydantic import TypeAdapter

    schemas: dict[str, Any] = spec.setdefault("components", {}).setdefault("schemas", {})

    def absorb(js: dict[str, Any], prefix: str) -> dict[str, Any]:
        for name, body in js.pop("$defs", {}).items():
            schemas[f"{prefix}_{name}"] = body
        return js

    schemas["ChatCompletion"] = absorb(
        ChatCompletion.model_json_schema(ref_template="#/components/schemas/CC_{model}"), "CC"
    )
    schemas["ChatCompletionChunk"] = absorb(
        ChatCompletionChunk.model_json_schema(ref_template="#/components/schemas/CCK_{model}"), "CCK"
    )
    msgs = absorb(
        TypeAdapter(list[ChatCompletionMessageParam]).json_schema(ref_template="#/components/schemas/MSG_{model}"),
        "MSG",
    )
    schemas["ChatMessageInput"] = msgs.get("items", {})

    chat = spec["paths"]["/v1/chat/completions"]["post"]
    chat["responses"]["200"] = {
        "description": "Chat completion",
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ChatCompletion"}}},
    }
    schemas["ChatCompletionRequest"]["properties"]["messages"] = {
        "type": "array",
        "minItems": 1,
        "items": {"$ref": "#/components/schemas/ChatMessageInput"},
    }
    return spec


def _generator_cli() -> str:
    return os.environ.get("OPENAPI_GENERATOR_CLI", "openapi-generator-cli")


def postprocess(language: str, dest: Path) -> None:
    """Apply language-specific fix-ups so the raw generator output is usable.

    Keeps each SDK repo's CI green without per-repo hand-edits to generated code:
    - rust: exempt the crate from rustfmt (``cargo fmt --all -- --check`` reaches
      it; ``disable_all_formatting`` is the only stable-channel option).
    - typescript: inject the ``mapValues`` helper the models import but this
      generator version omits.
    - go: drop the generated ``test/`` dir, whose unfilled
      ``GIT_USER_ID/GIT_REPO_ID`` import placeholders break ``go vet``/``go test``.
    """
    if language == "rust":
        (dest / "rustfmt.toml").write_text("disable_all_formatting = true\n")
    elif language == "typescript":
        runtime = dest / "runtime.ts"
        if runtime.exists() and "export function mapValues" not in runtime.read_text():
            with runtime.open("a") as handle:
                handle.write(_TS_MAPVALUES)
    elif language == "go":
        test_dir = dest / "test"
        if test_dir.is_dir():
            shutil.rmtree(test_dir)


def normalize(language: str, dest: Path, python_package: str) -> None:
    """Reduce the generator output to exactly what gets copied into the SDK repo.

    The python generator emits a full project (an ``otari/<package>/`` package
    plus ``setup.py`` etc.); collapse ``dest`` to the package directory so the
    workflow's ``dest -> target_path`` copy lands ``otari.<package>`` directly.
    Other languages already emit a flat payload.
    """
    if language == "python":
        package = dest / "otari" / python_package
        if package.is_dir():
            staged = dest.parent / f"{dest.name}__pkg"
            shutil.move(str(package), str(staged))
            shutil.rmtree(dest)
            shutil.move(str(staged), str(dest))


def generate_language(language: str, spec_path: Path, out_dir: Path, target: LanguageTarget) -> Path:
    """Run OpenAPI Generator for ``language`` and return the output directory."""
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
    postprocess(language, dest)
    normalize(language, dest, target.target_path.rsplit("/", 1)[-1])
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SDK clients from the otari OpenAPI spec.")
    parser.add_argument("--language", choices=[*TARGETS, "all"], default="all")
    parser.add_argument(
        "--mode",
        choices=["control-plane", "full"],
        default="control-plane",
        help="control-plane: typed management endpoints only. full: every endpoint "
        "(typed inference core) with the spec enriched (Option C).",
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--write-spec",
        type=Path,
        default=None,
        help="Also write the (filtered or enriched) spec to this path, for inspection.",
    )
    args = parser.parse_args()

    spec: dict[str, Any] = json.loads(Path(args.spec).read_text())
    if args.mode == "full":
        prepared = enrich_spec(spec)
        targets = FULL_TARGETS
        spec_name = "otari-full.openapi.json"
    else:
        prepared = filter_spec(spec, CONTROL_PLANE_TAGS)
        targets = TARGETS
        spec_name = "control-plane.openapi.json"

    languages = list(targets) if args.language == "all" else [str(args.language)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.write_spec is not None:
        Path(args.write_spec).write_text(json.dumps(prepared, indent=2))

    with tempfile.TemporaryDirectory() as tmp:
        spec_path = Path(tmp) / spec_name
        spec_path.write_text(json.dumps(prepared, indent=2))
        for language in languages:
            dest = generate_language(language, spec_path, out_dir, targets[language])
            print(f"generated {language} ({args.mode}) -> {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
