#!/usr/bin/env python3
"""Generate language SDK clients from the otari OpenAPI spec via OpenAPI Generator.

Two modes:

- ``control-plane`` (default): typed clients for the management endpoints only
  (keys, users, budgets, pricing, usage). The inference surface stays a
  hand-written wrapper around the official OpenAI SDK; batches are hand-written
  (their responses are untyped in the spec).

- ``full``: enriches the spec's inference surface with the real typed
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
import re
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SPEC = REPO_ROOT / "docs" / "public" / "openapi.json"
DEFAULT_OUT_DIR = REPO_ROOT / "dist" / "sdk-codegen"

# Placeholder stamped into the spec's ``info.version`` until a real gateway
# release supplies one (see gateway.version.__version__). Mirrors that default so
# a marker is always written even outside a release build.
DEFAULT_SPEC_VERSION = "0.0.0-dev"

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
    # When true, the generated output is reduced from a standalone crate to a
    # single module tree (lib.rs -> mod.rs, crate scaffolding dropped) so the SDK
    # publishes as one crate with no path dependency. Rust full mode only; see
    # _rust_inline_module.
    inline_module: bool = False


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

# The FULL client (every endpoint, typed core). The SDK hand-writes a
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
        # Inlined as a private module of the SDK crate (not a separate crate), so
        # the crate publishes to crates.io as a single unit with no path
        # dependency. The SDK's src/lib.rs declares `mod _client` and re-exports
        # `apis` / `models` at the crate root for the generated code's
        # `crate::models` / `crate::apis` paths.
        target_path="src/_client",
        inline_module=True,
    ),
}


# Header prepended to the inlined Rust core's mod.rs (replacing the generator's
# own crate-root attributes). Exempts the generated code from the SDK crate's
# strict lints, the way it had its own relaxed lints as a separate crate.
_RUST_MODULE_HEADER = """\
//! OpenAPI-generated typed core. This module is produced by the otari codegen
//! pipeline (scripts/sdk_codegen) and is not hand-edited; the lint allowances
//! below exempt it from the SDK crate's strict lints, which it escaped when it
//! lived in its own crate. Do not add hand-written code here.
#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(clippy::all)]
#![allow(clippy::pedantic)]
#![allow(clippy::nursery)]

"""


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
    the chat ``200`` response has no schema). For the full client (a generated core that
    covers every endpoint) we inject the real, typed completion schemas from
    ``any-llm`` (which the gateway already depends on) so the generated chat
    method returns a typed ``ChatCompletion`` and accepts typed messages.

    Streaming still cannot be generated (OpenAPI Generator emits no SSE); the SDK
    shell hand-writes the stream iterator over the generated client.
    """
    from any_llm.types.completion import (
        ChatCompletion,
        ChatCompletionChunk,
        CreateEmbeddingResponse,
    )
    from any_llm.types.image import ImagesResponse
    from any_llm.types.messages import MessageResponse
    from any_llm.types.rerank import RerankResponse
    from openai.types.chat import ChatCompletionMessageParam
    from pydantic import TypeAdapter

    schemas: dict[str, Any] = spec.setdefault("components", {}).setdefault("schemas", {})

    def absorb(js: dict[str, Any], prefix: str) -> dict[str, Any]:
        # Pydantic emits nested models under ``$defs``; lift them into the spec's
        # shared schema map under a unique prefix so distinct models can't collide
        # on a common nested name (e.g. two responses both referencing Usage).
        for name, body in js.pop("$defs", {}).items():
            schemas[f"{prefix}_{name}"] = body
        return js

    # These are response shapes, so generate the JSON Schema in ``serialization``
    # mode: it reflects the bytes the gateway actually puts on the wire, not the
    # (looser) validation shape. The any-llm ``Reasoning`` type is the motivating
    # case: it accepts ``{"content": str}`` on input but a ``model_serializer``
    # emits a plain string, so ``message.reasoning`` is a string in responses.
    # Validation-mode schema typed it as an object, and the generated client then
    # rejected the string the server sends. See mozilla-ai/otari#143.
    schemas["ChatCompletion"] = absorb(
        ChatCompletion.model_json_schema(mode="serialization", ref_template="#/components/schemas/CC_{model}"),
        "CC",
    )
    schemas["ChatCompletionChunk"] = absorb(
        ChatCompletionChunk.model_json_schema(mode="serialization", ref_template="#/components/schemas/CCK_{model}"),
        "CCK",
    )
    msgs = absorb(
        TypeAdapter(list[ChatCompletionMessageParam]).json_schema(ref_template="#/components/schemas/MSG_{model}"),
        "MSG",
    )
    schemas["ChatMessageInput"] = msgs.get("items", {})

    # otari-owned inference endpoints the gateway leaves ``response_model=None``.
    # Their real response shapes live in any-llm (which the gateway depends on);
    # inject them so the generated core returns typed models instead of ``object``.
    # Audio stays opaque here. Speech returns binary audio, which has no JSON
    # response schema. Transcription's response is JSON, but each SDK hand-writes
    # it over a raw multipart upload (a request-side concern the generated core
    # can't carry) and its shape varies by response_format, so there is no single
    # response model to inject. Image generation is plain JSON, so it is typed.
    # Response shapes too: serialization mode keeps the schema aligned with the
    # serialized wire format (see the ChatCompletion note above).
    schemas["MessageResponse"] = absorb(
        MessageResponse.model_json_schema(mode="serialization", ref_template="#/components/schemas/MR_{model}"),
        "MR",
    )
    schemas["RerankResponse"] = absorb(
        RerankResponse.model_json_schema(mode="serialization", ref_template="#/components/schemas/RR_{model}"),
        "RR",
    )
    schemas["CreateEmbeddingResponse"] = absorb(
        CreateEmbeddingResponse.model_json_schema(
            mode="serialization", ref_template="#/components/schemas/EMB_{model}"
        ),
        "EMB",
    )
    schemas["ImagesResponse"] = absorb(
        ImagesResponse.model_json_schema(mode="serialization", ref_template="#/components/schemas/IMG_{model}"),
        "IMG",
    )

    def set_json_200(path: str, schema_name: str, description: str) -> None:
        op = spec["paths"][path]["post"]
        op["responses"]["200"] = {
            "description": description,
            "content": {"application/json": {"schema": {"$ref": f"#/components/schemas/{schema_name}"}}},
        }

    set_json_200("/v1/chat/completions", "ChatCompletion", "Chat completion")
    set_json_200("/v1/messages", "MessageResponse", "Anthropic-style message")
    set_json_200("/v1/rerank", "RerankResponse", "Rerank result")
    set_json_200("/v1/embeddings", "CreateEmbeddingResponse", "Embeddings")
    set_json_200("/v1/images/generations", "ImagesResponse", "Generated images")

    schemas["ChatCompletionRequest"]["properties"]["messages"] = {
        "type": "array",
        "minItems": 1,
        "items": {"$ref": "#/components/schemas/ChatMessageInput"},
    }
    return spec


# Shared schema name for an arbitrary JSON object (``{"type": "object"}`` with
# free-form values). Injected by ``sanitize_freeform_object_arrays`` so union
# members that are arrays of such objects reference a named type instead of an
# inline one.
_FREE_FORM_OBJECT = "FreeFormObject"


def _is_free_form_object(node: Any) -> bool:
    """True for a bare ``{"type": "object"}`` schema that accepts arbitrary values.

    Matches an object with no declared ``properties`` whose only other key may be
    ``additionalProperties``, and only when that is omitted or literally ``true``.
    A closed object (``additionalProperties: false``) or a typed map
    (``additionalProperties: {schema}``) is not free-form and must not be
    collapsed to the shared open ``FreeFormObject``.
    """
    if not (
        isinstance(node, dict)
        and node.get("type") == "object"
        and "properties" not in node
        and set(node) <= {"type", "additionalProperties"}
    ):
        return False
    return node.get("additionalProperties", True) is True


def _is_empty_schema(node: Any) -> bool:
    """True for an always-matching schema used as a direct union member.

    An empty ``{}`` (or one carrying only annotation keys such as ``title`` /
    ``description``) accepts any JSON value. As a direct ``anyOf`` / ``oneOf``
    member it gives the generator no name to bind to, so the statically typed
    SDKs emit a dangling placeholder type for it (Rust
    ``models::AnyOfLessThanGreaterThan``, Go ``AnyOf`` / ``NullableAnyOf``,
    TypeScript an empty ``| null`` union with bare ``FromJSON`` / ``ToJSON``
    calls). any-llm produces these for loosely typed passthrough fields such as
    ``ResponsesRequest.text`` (``anyOf: [{}, {"type": "null"}]``). Point the
    member at the shared ``FreeFormObject`` so it is a named, generatable type.
    """
    return isinstance(node, dict) and not (set(node) - {"title", "description"})


def sanitize_freeform_object_arrays(spec: dict[str, Any]) -> dict[str, Any]:
    """Give free-form-object union members a named, generatable type.

    Two inline free-form shapes break the statically typed generators when they
    appear in an ``anyOf`` / ``oneOf`` union, so both are pointed at a shared
    named ``FreeFormObject`` schema:

    - **array-of-free-form-object members.** any-llm's Anthropic params type
      fields like ``system`` as ``str | list[dict] | None``, whose array variant
      has inline free-form-object ``items``
      (``{"type": "object", "additionalProperties": true}``). When such a union
      has two or more non-null variants the Go generator synthesizes a wrapper
      struct and names the array field from that inline item type, emitting an
      invalid identifier (``ArrayOf*mapmapOfStringAny`` — the ``*`` comes from
      the pointer-typed item). That fails ``gofmt`` and breaks the Go SDK build.
    - **bare empty-schema members.** A direct ``{}`` member (see
      :func:`_is_empty_schema`) has no name for the generator to bind to, so the
      static SDKs emit a dangling placeholder type that never gets generated and
      fails to compile (e.g. ``ResponsesRequest.text``).

    The generator-internal naming is inconsistent, so rather than depend on it,
    rewrite every such member to reference the shared schema; the member name is
    then deterministic across all SDK languages.
    """
    schemas = spec.get("components", {}).get("schemas")
    if not isinstance(schemas, dict):
        return spec
    used = False

    def walk(node: Any) -> None:
        nonlocal used
        if isinstance(node, dict):
            for key in ("anyOf", "oneOf"):
                members = node.get(key)
                if not isinstance(members, list):
                    continue
                for index, member in enumerate(members):
                    if (
                        isinstance(member, dict)
                        and member.get("type") == "array"
                        and _is_free_form_object(member.get("items"))
                    ):
                        member["items"] = {"$ref": f"#/components/schemas/{_FREE_FORM_OBJECT}"}
                        used = True
                    elif _is_empty_schema(member):
                        members[index] = {"$ref": f"#/components/schemas/{_FREE_FORM_OBJECT}"}
                        used = True
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(spec)
    if used:
        desired = {"type": "object", "additionalProperties": True}
        existing = schemas.get(_FREE_FORM_OBJECT)
        if existing is not None and existing != desired:
            raise ValueError(
                f"schema {_FREE_FORM_OBJECT!r} already exists with a different shape; "
                "cannot reuse it for free-form array-item sanitization"
            )
        schemas[_FREE_FORM_OBJECT] = desired
    return spec


def _generator_cli() -> str:
    return os.environ.get("OPENAPI_GENERATOR_CLI", "openapi-generator-cli")


def _patch_rust_cargo_toml(dest: Path) -> None:
    """Correct the generated Rust ``Cargo.toml`` so the crate builds as the SDK needs.

    The rust generator emits a truncated ``version`` (``0.0.0-de`` from "dev") and a
    ``reqwest`` dependency pinned to ``^0.13`` with a feature set that does not build
    against the SDK's pinned ``reqwest 0.12``; it also names the wrong rustls feature
    (``reqwest/rustls`` rather than ``reqwest/rustls-tls``). The Rust SDK port fixed
    these by hand; do them here so regenerations are clean. Replacements are tolerant
    of formatting so they survive generator-version churn.
    """
    cargo = dest / "Cargo.toml"
    if not cargo.exists():
        return
    text = cargo.read_text()

    # version under [package]: any "0.0.0-*" placeholder -> 0.1.0.
    text = re.sub(
        r'^(version\s*=\s*)"0\.0\.0[^"]*"',
        r'\1"0.1.0"',
        text,
        count=1,
        flags=re.MULTILINE,
    )

    # reqwest dependency: pin to 0.12 and keep only the features the SDK builds with.
    text = re.sub(
        r'^reqwest\s*=\s*\{[^}]*\}',
        'reqwest = { version = "0.12", default-features = false, '
        'features = ["json", "multipart"] }',
        text,
        count=1,
        flags=re.MULTILINE,
    )

    # rustls feature must map onto reqwest's rustls-tls (not rustls).
    text = text.replace('"reqwest/rustls"', '"reqwest/rustls-tls"')

    cargo.write_text(text)


def postprocess(language: str, dest: Path) -> None:
    """Apply language-specific fix-ups so the raw generator output is usable.

    Keeps each SDK repo's CI green without per-repo hand-edits to generated code:
    - rust: exempt the crate from rustfmt (``cargo fmt --all -- --check`` reaches
      it; ``disable_all_formatting`` is the only stable-channel option), and patch
      the emitted ``Cargo.toml`` (placeholder version + a ``reqwest`` pin/feature
      set that does not build against the SDK's pinned ``reqwest 0.12``).
    - typescript: inject the ``mapValues`` helper the models import but this
      generator version omits.
    - go: drop the generated ``test/`` dir, whose unfilled
      ``GIT_USER_ID/GIT_REPO_ID`` import placeholders break ``go vet``/``go test``,
      then ``gofmt -w`` the payload (the generator emits un-gofmt'd Go, which fails
      the SDK repo's ``gofmt -l`` check).
    """
    if language == "rust":
        (dest / "rustfmt.toml").write_text("disable_all_formatting = true\n")
        _patch_rust_cargo_toml(dest)
    elif language == "typescript":
        runtime = dest / "runtime.ts"
        if runtime.exists() and "export function mapValues" not in runtime.read_text():
            with runtime.open("a") as handle:
                handle.write(_TS_MAPVALUES)
    elif language == "go":
        test_dir = dest / "test"
        if test_dir.is_dir():
            shutil.rmtree(test_dir)
        gofmt = shutil.which("gofmt")
        if gofmt is None:
            warnings.warn(
                "gofmt not found on PATH; skipping gofmt of generated Go "
                "(the SDK repo's gofmt -l check may then fail).",
                stacklevel=2,
            )
        else:
            subprocess.run([gofmt, "-w", str(dest)], check=True)


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


def _go_package_name(target: LanguageTarget) -> str:
    """Read ``packageName=`` out of a go target's additional properties.

    The marker file must declare the same ``package`` as the rest of the generated
    Go core (``generated`` for control-plane, ``client`` for full); fall back to the
    target path's last segment if the property is absent.
    """
    for prop in target.additional_properties.split(","):
        key, _, value = prop.partition("=")
        if key.strip() == "packageName" and value.strip():
            return value.strip()
    return target.target_path.rsplit("/", 1)[-1]


def write_spec_version(language: str, dest: Path, version: str, target: LanguageTarget) -> None:
    """Stamp ``version`` into the generated core as a language-native marker file.

    Compatibility between an SDK and the gateway is expressed through the spec
    version, not matching package numbers, so each generated core carries the
    gateway/spec version it was generated from. The SDK's hand-written shell reads
    this marker to surface it (``__spec_version__`` / ``SPEC_VERSION`` / equivalent).

    Each file is self-contained so no SDK-side wiring beyond an import is needed:
    Python and TypeScript are imported by path; Go shares the core's package and so
    the const is in scope automatically; for Rust the marker is declared as a
    submodule of the inlined core's ``mod.rs`` (created by ``_rust_inline_module``).
    Runs last in ``generate_language`` so ``dest`` is already in its final layout.
    """
    if language == "python":
        (dest / "_spec_version.py").write_text(f'__spec_version__ = "{version}"\n')
    elif language == "typescript":
        (dest / "specVersion.ts").write_text(f'export const SPEC_VERSION = "{version}";\n')
    elif language == "go":
        package = _go_package_name(target)
        (dest / "spec_version.go").write_text(f'package {package}\n\nconst SpecVersion = "{version}"\n')
    elif language == "rust":
        # The crate root is ``dest/mod.rs`` for the inlined full core, or
        # ``dest/src/lib.rs`` for the standalone crate the rust generator emits in
        # control-plane mode. Place the marker beside whichever root exists and
        # declare the module there, so it is compiled and importable in both
        # layouts (the bare ``dest/mod.rs`` fallback covers a partial payload).
        root = dest / "mod.rs"
        if not root.exists() and (dest / "src" / "lib.rs").exists():
            root = dest / "src" / "lib.rs"
        (root.parent / "spec_version.rs").write_text(f'pub const SPEC_VERSION: &str = "{version}";\n')
        declaration = "pub mod spec_version;"
        if root.exists():
            text = root.read_text()
            if declaration not in text:
                root.write_text(f"{text.rstrip()}\n{declaration}\n")
        else:
            root.write_text(f"{declaration}\n")


def _rustfmt_tree(dest: Path) -> None:
    """Format the inlined Rust module in place with rustfmt.

    The inlined core becomes part of the SDK crate, so it is reached by the SDK
    repo's ``cargo fmt --all -- --check``; the old separate crate was exempted
    via ``rustfmt.toml``. rustfmt recurses into child ``mod`` declarations by
    default, so formatting the module root (``mod.rs``) formats the whole tree.
    Missing rustfmt warns and skips rather than failing, mirroring the go path.
    """
    rustfmt = shutil.which("rustfmt")
    if rustfmt is None:
        warnings.warn(
            "rustfmt not found on PATH; skipping format of the generated Rust "
            "core (the SDK repo's `cargo fmt --all -- --check` may then fail).",
            stacklevel=2,
        )
        return
    root = dest / "mod.rs"
    if root.exists():
        subprocess.run([rustfmt, "--edition", "2021", str(root)], check=True)


def _rust_inline_module(dest: Path) -> None:
    """Reduce the generated Rust crate at ``dest`` to a single module tree.

    The crate split (`otari` shell + generated `otari-client` core wired as a
    path dependency) made the SDK unpublishable: ``cargo publish`` rejects a path
    dependency with no version. Inlining the core as a private module of the SDK
    crate fixes that (see mozilla-ai/otari-sdk-rust#37). This rewrites the
    generator's crate output so it drops cleanly into the SDK's ``src/_client``:

    - ``src/lib.rs`` -> ``mod.rs`` with the crate-root attributes replaced by the
      SDK lint-exemption header (the module declarations are preserved);
    - the rest of ``src/`` (``apis/``, ``models/``) is hoisted to ``dest``;
    - crate scaffolding (``Cargo.toml``, docs, generator metadata, etc.) is
      dropped, since the core's dependencies live in the SDK crate's Cargo.toml;
    - every ``.rs`` file is formatted (the old separate crate was rustfmt-exempt).
    """
    src = dest / "src"
    lib = src / "lib.rs"
    if lib.exists():
        lines = lib.read_text().splitlines()
        start = next(
            (i for i, line in enumerate(lines) if line.startswith(("pub mod ", "mod "))),
            len(lines),
        )
        body = "\n".join(lines[start:]).strip()
        lib.unlink()
    else:
        body = "pub mod apis;\npub mod models;"

    for child in sorted(src.iterdir()):
        # Clear any same-named entry first so re-running into a non-empty
        # out-dir overwrites rather than nesting (e.g. apis/ into apis/apis/).
        target = dest / child.name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        shutil.move(str(child), str(target))
    src.rmdir()
    (dest / "mod.rs").write_text(_RUST_MODULE_HEADER + body + "\n")

    for name in (
        "Cargo.toml",
        "README.md",
        ".travis.yml",
        "git_push.sh",
        ".gitignore",
        "rustfmt.toml",
        ".openapi-generator-ignore",
    ):
        (dest / name).unlink(missing_ok=True)
    for directory in ("docs", ".openapi-generator"):
        shutil.rmtree(dest / directory, ignore_errors=True)

    _rustfmt_tree(dest)


def generate_language(
    language: str, spec_path: Path, out_dir: Path, target: LanguageTarget, spec_version: str
) -> Path:
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
    if target.inline_module:
        _rust_inline_module(dest)
    else:
        postprocess(language, dest)
        normalize(language, dest, target.target_path.rsplit("/", 1)[-1])
    write_spec_version(language, dest, spec_version, target)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SDK clients from the otari OpenAPI spec.")
    parser.add_argument("--language", choices=[*TARGETS, "all"], default="all")
    parser.add_argument(
        "--mode",
        choices=["control-plane", "full"],
        default="control-plane",
        help="control-plane: typed management endpoints only. full: every endpoint "
        "(typed inference core) with the spec enriched.",
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--spec-version",
        default=None,
        help="Gateway/spec version stamped into each generated core (and into the "
        "spec's info.version). Defaults to the spec's own info.version. The codegen "
        "workflow passes the gateway release tag here.",
    )
    parser.add_argument(
        "--write-spec",
        type=Path,
        default=None,
        help="Also write the (filtered or enriched) spec to this path, for inspection.",
    )
    args = parser.parse_args()

    spec: dict[str, Any] = json.loads(Path(args.spec).read_text())
    spec_version = args.spec_version or spec.get("info", {}).get("version", DEFAULT_SPEC_VERSION)
    if args.spec_version:
        spec.setdefault("info", {})["version"] = args.spec_version
    if args.mode == "full":
        prepared = enrich_spec(spec)
        targets = FULL_TARGETS
        spec_name = "otari-full.openapi.json"
    else:
        prepared = filter_spec(spec, CONTROL_PLANE_TAGS)
        targets = TARGETS
        spec_name = "control-plane.openapi.json"
    prepared = sanitize_freeform_object_arrays(prepared)

    languages = list(targets) if args.language == "all" else [str(args.language)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.write_spec is not None:
        Path(args.write_spec).write_text(json.dumps(prepared, indent=2))

    with tempfile.TemporaryDirectory() as tmp:
        spec_path = Path(tmp) / spec_name
        spec_path.write_text(json.dumps(prepared, indent=2))
        for language in languages:
            dest = generate_language(language, spec_path, out_dir, targets[language], spec_version)
            print(f"generated {language} ({args.mode}) at spec version {spec_version} -> {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
