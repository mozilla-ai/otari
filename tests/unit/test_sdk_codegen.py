"""Unit tests for the control-plane SDK codegen spec filter."""

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_GENERATE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "sdk_codegen" / "generate.py"


def _load_generate() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sdk_codegen_generate", _GENERATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generate = _load_generate()


def _sample_spec() -> dict[str, Any]:
    return {
        "openapi": "3.1.0",
        "info": {"title": "t", "version": "0"},
        "paths": {
            "/v1/keys": {
                "post": {
                    "tags": ["keys"],
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/KeyInfo"}}
                            }
                        }
                    },
                }
            },
            "/v1/chat/completions": {
                "post": {
                    "tags": ["chat"],
                    "requestBody": {
                        "content": {
                            "application/json": {"schema": {"$ref": "#/components/schemas/ChatRequest"}}
                        }
                    },
                }
            },
        },
        "components": {
            "schemas": {
                "KeyInfo": {
                    "type": "object",
                    "properties": {"meta": {"$ref": "#/components/schemas/KeyMeta"}},
                },
                "KeyMeta": {"type": "object"},
                "ChatRequest": {"type": "object"},
                "Orphan": {"type": "object"},
            }
        },
    }


def test_filter_keeps_control_plane_and_drops_inference() -> None:
    result = generate.filter_spec(_sample_spec(), frozenset({"keys"}))
    assert set(result["paths"]) == {"/v1/keys"}


def test_filter_prunes_schemas_to_reachable_closure() -> None:
    result = generate.filter_spec(_sample_spec(), frozenset({"keys"}))
    schemas = set(result["components"]["schemas"])
    # KeyInfo is referenced by the kept op; KeyMeta is reachable from KeyInfo.
    assert schemas == {"KeyInfo", "KeyMeta"}
    # Inference-only and unreferenced schemas are dropped.
    assert "ChatRequest" not in schemas
    assert "Orphan" not in schemas


def test_postprocess_rust_writes_rustfmt_exemption(tmp_path: Path) -> None:
    generate.postprocess("rust", tmp_path)
    cfg = tmp_path / "rustfmt.toml"
    assert cfg.exists()
    assert "disable_all_formatting = true" in cfg.read_text()


def test_postprocess_typescript_injects_mapvalues_idempotently(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime.ts"
    runtime.write_text("export const BASE_PATH = '';\n")
    generate.postprocess("typescript", tmp_path)
    once = runtime.read_text()
    assert "export function mapValues" in once
    # Running again must not append a second copy.
    generate.postprocess("typescript", tmp_path)
    assert runtime.read_text().count("export function mapValues") == 1


def test_postprocess_go_drops_broken_generated_tests(tmp_path: Path) -> None:
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "api_keys_test.go").write_text('import "github.com/GIT_USER_ID/GIT_REPO_ID"\n')
    generate.postprocess("go", tmp_path)
    assert not test_dir.exists()


def test_postprocess_go_gofmts_generated_payload(tmp_path: Path) -> None:
    if shutil.which("gofmt") is None:
        pytest.skip("gofmt not on PATH")
    # OpenAPI Generator emits un-gofmt'd Go; postprocess must run gofmt -w so the
    # SDK repo's `gofmt -l` check stays empty. Use an intentionally mis-formatted
    # (but valid) source file and assert it is reformatted in place.
    unformatted = "package generated\nfunc  Foo( )  int {return  1}\n"
    src = tmp_path / "thing.go"
    src.write_text(unformatted)
    generate.postprocess("go", tmp_path)
    formatted = src.read_text()
    assert formatted != unformatted
    assert "func Foo() int {" in formatted
    assert "return 1" in formatted


def test_postprocess_go_skips_gracefully_without_gofmt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # When gofmt is absent, postprocess must warn and skip rather than crash.
    monkeypatch.setattr(generate.shutil, "which", lambda _name: None)
    src = tmp_path / "thing.go"
    original = "package generated\nfunc  Foo( ) {}\n"
    src.write_text(original)
    with pytest.warns(UserWarning, match="gofmt"):
        generate.postprocess("go", tmp_path)
    # File is left untouched (no formatter ran).
    assert src.read_text() == original


def test_postprocess_rust_patches_cargo_toml(tmp_path: Path) -> None:
    # Stub mirrors what the rust generator currently emits (truncated dev version,
    # reqwest ^0.13 with an over-broad feature set, and the wrong rustls feature).
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text(
        "[package]\n"
        'name = "otari-client"\n'
        'version = "0.0.0-de"\n'
        'edition = "2021"\n'
        "\n"
        "[dependencies]\n"
        'serde = { version = "^1.0", features = ["derive"] }\n'
        'reqwest = { version = "^0.13", default-features = false, '
        'features = ["json", "multipart", "query", "form"] }\n'
        "\n"
        "[features]\n"
        'default = ["native-tls"]\n'
        'native-tls = ["reqwest/native-tls"]\n'
        'rustls = ["reqwest/rustls"]\n'
    )
    generate.postprocess("rust", tmp_path)
    text = cargo.read_text()
    # Placeholder version -> real release version.
    assert 'version = "0.1.0"' in text
    assert "0.0.0" not in text
    # reqwest pinned to 0.12 with only the features the SDK builds against.
    assert 'reqwest = { version = "0.12", default-features = false, features = ["json", "multipart"] }' in text
    assert "0.13" not in text
    assert '"query"' not in text
    assert '"form"' not in text
    # rustls feature mapped onto reqwest's rustls-tls.
    assert 'rustls = ["reqwest/rustls-tls"]' in text
    assert '"reqwest/rustls"' not in text
    # Unrelated deps and the rustfmt exemption are left intact.
    assert 'serde = { version = "^1.0", features = ["derive"] }' in text
    assert (tmp_path / "rustfmt.toml").exists()


def test_postprocess_rust_cargo_toml_missing_is_noop(tmp_path: Path) -> None:
    # No Cargo.toml (e.g. partial output) must not crash; rustfmt.toml still written.
    generate.postprocess("rust", tmp_path)
    assert (tmp_path / "rustfmt.toml").exists()
    assert not (tmp_path / "Cargo.toml").exists()


def test_normalize_python_collapses_to_package(tmp_path: Path) -> None:
    dest = tmp_path / "python"
    pkg = dest / "otari" / "_control_plane"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("x = 1\n")
    (dest / "setup.py").write_text("# project scaffolding\n")
    generate.normalize("python", dest, "_control_plane")
    # dest is now the package itself: its __init__.py is present, scaffolding gone.
    assert (dest / "__init__.py").read_text() == "x = 1\n"
    assert not (dest / "setup.py").exists()
    assert not (dest / "otari").exists()


def _full_spec_stub() -> dict[str, Any]:
    return {
        "openapi": "3.1.0",
        "info": {"title": "t", "version": "0"},
        "paths": {
            "/v1/chat/completions": {"post": {"responses": {}}},
            "/v1/messages": {"post": {"responses": {}}},
            "/v1/rerank": {"post": {"responses": {}}},
            "/v1/embeddings": {"post": {"responses": {}}},
            "/v1/images/generations": {"post": {"responses": {}}},
        },
        "components": {"schemas": {"ChatCompletionRequest": {"type": "object", "properties": {}}}},
    }


def test_enrich_types_otari_owned_inference_endpoints() -> None:
    spec = generate.enrich_spec(_full_spec_stub())
    schemas = spec["components"]["schemas"]
    for name in (
        "ChatCompletion",
        "ChatCompletionChunk",
        "MessageResponse",
        "RerankResponse",
        "CreateEmbeddingResponse",
        "ImagesResponse",
        "ChatMessageInput",
    ):
        assert name in schemas, f"{name} schema missing after enrich"

    def ref(path: str) -> str:
        schema = spec["paths"][path]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        return str(schema["$ref"])

    assert ref("/v1/chat/completions").endswith("/ChatCompletion")
    # /messages is Anthropic-shaped and has no OpenAI-SDK equivalent; typing it
    # from any-llm's MessageResponse is the whole point of generating from the
    # otari spec rather than wrapping the OpenAI SDK.
    assert ref("/v1/messages").endswith("/MessageResponse")
    assert ref("/v1/rerank").endswith("/RerankResponse")
    assert ref("/v1/embeddings").endswith("/CreateEmbeddingResponse")
    # Image generation is plain JSON, so it is typed from any-llm's
    # ImagesResponse. Speech (binary response) and transcription (multipart
    # request) stay opaque in the codegen enrichment.
    assert ref("/v1/images/generations").endswith("/ImagesResponse")

    messages_field = schemas["ChatCompletionRequest"]["properties"]["messages"]
    assert messages_field["items"]["$ref"].endswith("/ChatMessageInput")


def test_enrich_types_reasoning_as_string_matching_wire_format() -> None:
    # The gateway serializes ``message.reasoning`` as a plain string (any-llm's
    # ``Reasoning`` model has a ``model_serializer`` that emits a string), but the
    # validation-mode JSON Schema typed it as an object, so the generated client
    # rejected the string the server returns. Generating the response schemas in
    # serialization mode keeps the schema aligned with the wire format.
    # Regression test for mozilla-ai/otari#143.
    spec = generate.enrich_spec(_full_spec_stub())
    schemas = spec["components"]["schemas"]
    for prefix in ("CC", "CCK"):
        reasoning = schemas[f"{prefix}_Reasoning"]
        assert reasoning == {"type": "string"}, f"{prefix}_Reasoning must be a plain string, got {reasoning}"


def _spec_with_freeform_union() -> dict[str, Any]:
    # Mirrors how any-llm types Anthropic's ``system`` (``str | list[dict] | None``):
    # an anyOf whose array variant has inline free-form-object items, alongside a
    # single-variant ``tools`` (``list[dict] | None``) array.
    free_form = {"additionalProperties": True, "type": "object"}
    return {
        "openapi": "3.1.0",
        "info": {"title": "t", "version": "0"},
        "components": {
            "schemas": {
                "MessagesRequest": {
                    "type": "object",
                    "properties": {
                        "system": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "array", "items": dict(free_form)},
                                {"type": "null"},
                            ]
                        },
                        "tools": {
                            "anyOf": [
                                {"type": "array", "items": dict(free_form)},
                                {"type": "null"},
                            ]
                        },
                    },
                }
            }
        },
    }


def test_sanitize_freeform_object_arrays_names_union_array_items() -> None:
    spec = generate.sanitize_freeform_object_arrays(_spec_with_freeform_union())
    schemas = spec["components"]["schemas"]
    assert schemas[generate._FREE_FORM_OBJECT] == {"type": "object", "additionalProperties": True}
    props = schemas["MessagesRequest"]["properties"]
    for field in ("system", "tools"):
        array_variant = next(v for v in props[field]["anyOf"] if v.get("type") == "array")
        assert array_variant["items"] == {
            "$ref": f"#/components/schemas/{generate._FREE_FORM_OBJECT}"
        }, f"{field} array items should reference the shared free-form object schema"


def test_sanitize_freeform_object_collapses_empty_union_member() -> None:
    # any-llm types ResponsesRequest.text as anyOf: [{}, {"type": "null"}]. The
    # empty {} member has no name for the static generators to bind to, so point
    # it at the shared FreeFormObject (the null member is left alone).
    spec = {
        "components": {
            "schemas": {
                "ResponsesRequest": {
                    "type": "object",
                    "properties": {
                        "text": {"anyOf": [{}, {"type": "null"}], "title": "Text"},
                    },
                }
            }
        }
    }
    out = generate.sanitize_freeform_object_arrays(spec)
    schemas = out["components"]["schemas"]
    assert schemas[generate._FREE_FORM_OBJECT] == {"type": "object", "additionalProperties": True}
    members = schemas["ResponsesRequest"]["properties"]["text"]["anyOf"]
    assert {"$ref": f"#/components/schemas/{generate._FREE_FORM_OBJECT}"} in members
    assert {"type": "null"} in members


def test_sanitize_freeform_object_leaves_typed_union_members() -> None:
    # A union of concrete members has nothing to collapse: no FreeFormObject is
    # introduced and the members are untouched.
    spec = {
        "components": {
            "schemas": {
                "Typed": {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]}
            }
        }
    }
    out = generate.sanitize_freeform_object_arrays(spec)
    assert generate._FREE_FORM_OBJECT not in out["components"]["schemas"]
    assert out["components"]["schemas"]["Typed"]["anyOf"] == [
        {"type": "string"},
        {"type": "integer"},
        {"type": "null"},
    ]


def test_sanitize_freeform_object_arrays_leaves_closed_object_arrays() -> None:
    # A closed object (additionalProperties: false) is not free-form: collapsing it
    # to the open FreeFormObject would change the contract, so it must be left alone.
    spec = {
        "components": {
            "schemas": {
                "Closed": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "object", "additionalProperties": False}},
                    ]
                }
            }
        }
    }
    out = generate.sanitize_freeform_object_arrays(spec)
    assert generate._FREE_FORM_OBJECT not in out["components"]["schemas"]
    array_variant = next(v for v in out["components"]["schemas"]["Closed"]["anyOf"] if v.get("type") == "array")
    assert array_variant["items"] == {"type": "object", "additionalProperties": False}


def test_sanitize_freeform_object_arrays_leaves_typed_map_arrays() -> None:
    # A typed map (additionalProperties is a schema) constrains its values, so it
    # is not free-form and must not be collapsed to the open FreeFormObject.
    typed_map = {"type": "object", "additionalProperties": {"type": "string"}}
    spec = {
        "components": {
            "schemas": {
                "TypedMap": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "array", "items": dict(typed_map)},
                    ]
                }
            }
        }
    }
    out = generate.sanitize_freeform_object_arrays(spec)
    assert generate._FREE_FORM_OBJECT not in out["components"]["schemas"]
    array_variant = next(v for v in out["components"]["schemas"]["TypedMap"]["anyOf"] if v.get("type") == "array")
    assert array_variant["items"] == typed_map


def test_sanitize_freeform_object_arrays_raises_on_conflicting_existing_schema() -> None:
    # A pre-existing FreeFormObject with different semantics must not be silently
    # overwritten; the sanitizer cannot safely reuse the name.
    spec = _spec_with_freeform_union()
    spec["components"]["schemas"][generate._FREE_FORM_OBJECT] = {
        "type": "object",
        "properties": {"id": {"type": "string"}},
    }
    with pytest.raises(ValueError, match="different shape"):
        generate.sanitize_freeform_object_arrays(spec)


def test_sanitize_freeform_object_arrays_is_noop_without_freeform_unions() -> None:
    spec = {
        "components": {
            "schemas": {
                "Plain": {
                    "type": "object",
                    "properties": {"items": {"type": "array", "items": {"type": "string"}}},
                }
            }
        }
    }
    out = generate.sanitize_freeform_object_arrays(spec)
    assert generate._FREE_FORM_OBJECT not in out["components"]["schemas"]


def test_control_plane_tags_are_typed_management_only() -> None:
    assert generate.CONTROL_PLANE_TAGS == frozenset(
        {"keys", "users", "budgets", "pricing", "usage"}
    )
    # Excluded on purpose: proxy/inference surfaces and batches, all of which are
    # untyped in the spec (so generation would regress them).
    for excluded in ("chat", "responses", "embeddings", "batches"):
        assert excluded not in generate.CONTROL_PLANE_TAGS


def _fake_rust_crate(dest: Path) -> None:
    """Build a minimal stand-in for the rust generator's crate output."""
    src = dest / "src"
    (src / "apis").mkdir(parents=True)
    (src / "models").mkdir(parents=True)
    (src / "lib.rs").write_text(
        "#![allow(unused_imports)]\n"
        "#![allow(clippy::too_many_arguments)]\n"
        "\n"
        "extern crate serde;\n"
        "extern crate reqwest;\n"
        "\n"
        "pub mod apis;\n"
        "pub mod models;\n"
    )
    (src / "apis" / "mod.rs").write_text("pub mod configuration;\n")
    (src / "apis" / "configuration.rs").write_text("pub struct Configuration {}\n")
    (src / "models" / "mod.rs").write_text("pub mod thing;\n")
    (src / "models" / "thing.rs").write_text("pub struct Thing {}\n")
    (dest / "Cargo.toml").write_text('[package]\nname = "otari-client"\n')
    (dest / "README.md").write_text("# generated\n")
    (dest / ".travis.yml").write_text("language: rust\n")
    (dest / "rustfmt.toml").write_text("disable_all_formatting = true\n")
    docs = dest / "docs"
    docs.mkdir()
    (docs / "Thing.md").write_text("# Thing\n")
    meta = dest / ".openapi-generator"
    meta.mkdir()
    (meta / "VERSION").write_text("7.0.0\n")


def test_rust_inline_module_reduces_crate_to_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Format is exercised separately; stub it so this test is deterministic and
    # does not depend on rustfmt being installed.
    monkeypatch.setattr(generate, "_rustfmt_tree", lambda _dest: None)
    dest = tmp_path / "rust"
    dest.mkdir()
    _fake_rust_crate(dest)

    generate._rust_inline_module(dest)

    # lib.rs -> mod.rs: SDK lint header replaces the crate-root attributes /
    # extern crate lines, module declarations preserved.
    mod = dest / "mod.rs"
    assert mod.exists()
    text = mod.read_text()
    assert "pub mod apis;" in text
    assert "pub mod models;" in text
    assert "#![allow(clippy::pedantic)]" in text
    assert "extern crate" not in text
    assert "clippy::too_many_arguments" not in text

    # src/ hoisted up to the module root; nested module files preserved.
    assert (dest / "apis" / "configuration.rs").exists()
    assert (dest / "models" / "thing.rs").exists()
    assert not (dest / "src").exists()

    # crate scaffolding dropped.
    for gone in ("Cargo.toml", "README.md", ".travis.yml", "rustfmt.toml", "lib.rs"):
        assert not (dest / gone).exists()
    assert not (dest / "docs").exists()
    assert not (dest / ".openapi-generator").exists()


def test_rust_inline_module_without_lib_rs_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A partial payload with no lib.rs must still yield a usable mod.rs.
    monkeypatch.setattr(generate, "_rustfmt_tree", lambda _dest: None)
    dest = tmp_path / "rust"
    (dest / "src" / "apis").mkdir(parents=True)
    (dest / "src" / "models").mkdir(parents=True)

    generate._rust_inline_module(dest)

    text = (dest / "mod.rs").read_text()
    assert "pub mod apis;" in text
    assert "pub mod models;" in text


def test_rust_inline_module_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Re-running into the same out-dir (a fresh generator payload dropped beside
    # the previous inlined output) must overwrite, not nest (no apis/apis/).
    monkeypatch.setattr(generate, "_rustfmt_tree", lambda _dest: None)
    dest = tmp_path / "rust"
    dest.mkdir()
    _fake_rust_crate(dest)
    generate._rust_inline_module(dest)

    # Second generator run drops a fresh crate payload beside the inlined output.
    _fake_rust_crate(dest)
    generate._rust_inline_module(dest)

    assert (dest / "apis" / "configuration.rs").exists()
    assert not (dest / "apis" / "apis").exists()
    assert not (dest / "models" / "models").exists()
    assert not (dest / "src").exists()


def test_rust_inline_module_skips_rustfmt_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # When rustfmt is absent, the transform still runs and only formatting is
    # skipped (with a warning), mirroring the go/gofmt path.
    monkeypatch.setattr(generate.shutil, "which", lambda _name: None)
    dest = tmp_path / "rust"
    dest.mkdir()
    _fake_rust_crate(dest)

    with pytest.warns(UserWarning, match="rustfmt"):
        generate._rust_inline_module(dest)

    assert (dest / "mod.rs").exists()
    assert (dest / "apis" / "configuration.rs").exists()
    assert not (dest / "src").exists()


def test_full_rust_target_is_inlined_module() -> None:
    # The full rust target must drop into src/_client as an inlined module, so a
    # regeneration does not recreate the separate `client/` crate that made the
    # SDK unpublishable.
    rust = generate.FULL_TARGETS["rust"]
    assert rust.target_path == "src/_client"
    assert rust.inline_module is True


def test_spec_version_python_marker(tmp_path: Path) -> None:
    generate.write_spec_version("python", tmp_path, "1.2.3", generate.FULL_TARGETS["python"])
    assert (tmp_path / "_spec_version.py").read_text() == '__spec_version__ = "1.2.3"\n'


def test_spec_version_typescript_marker(tmp_path: Path) -> None:
    generate.write_spec_version("typescript", tmp_path, "1.2.3", generate.FULL_TARGETS["typescript"])
    assert (tmp_path / "specVersion.ts").read_text() == 'export const SPEC_VERSION = "1.2.3";\n'


def test_spec_version_go_marker_uses_core_package(tmp_path: Path) -> None:
    # The full go core is package `client`; the marker must share that package so
    # the const compiles into the same package as the rest of the core.
    generate.write_spec_version("go", tmp_path, "1.2.3", generate.FULL_TARGETS["go"])
    assert (tmp_path / "spec_version.go").read_text() == 'package client\n\nconst SpecVersion = "1.2.3"\n'


def test_spec_version_go_marker_uses_control_plane_package(tmp_path: Path) -> None:
    # The control-plane go core is package `generated`; the marker follows it.
    generate.write_spec_version("go", tmp_path, "1.2.3", generate.TARGETS["go"])
    assert (tmp_path / "spec_version.go").read_text().startswith("package generated\n")


def test_spec_version_rust_marker_and_mod_declaration(tmp_path: Path) -> None:
    # mod.rs already exists (created by _rust_inline_module); the marker module
    # declaration is appended without disturbing existing declarations.
    (tmp_path / "mod.rs").write_text("pub mod apis;\npub mod models;\n")
    generate.write_spec_version("rust", tmp_path, "1.2.3", generate.FULL_TARGETS["rust"])
    assert (tmp_path / "spec_version.rs").read_text() == 'pub const SPEC_VERSION: &str = "1.2.3";\n'
    mod_text = (tmp_path / "mod.rs").read_text()
    assert "pub mod apis;" in mod_text
    assert "pub mod models;" in mod_text
    assert "pub mod spec_version;" in mod_text


def test_spec_version_rust_marker_declaration_is_idempotent(tmp_path: Path) -> None:
    (tmp_path / "mod.rs").write_text("pub mod apis;\n")
    generate.write_spec_version("rust", tmp_path, "1.2.3", generate.FULL_TARGETS["rust"])
    generate.write_spec_version("rust", tmp_path, "1.2.3", generate.FULL_TARGETS["rust"])
    assert (tmp_path / "mod.rs").read_text().count("pub mod spec_version;") == 1


def test_spec_version_rust_marker_non_inlined_crate_layout(tmp_path: Path) -> None:
    # Control-plane mode emits a standalone crate (src/lib.rs), not an inlined
    # mod.rs. The marker must land under src/ and be declared in lib.rs so it is
    # compiled, not dropped beside an unreferenced top-level mod.rs.
    src = tmp_path / "src"
    src.mkdir()
    (src / "lib.rs").write_text("pub mod apis;\npub mod models;\n")
    generate.write_spec_version("rust", tmp_path, "1.2.3", generate.TARGETS["rust"])
    assert (src / "spec_version.rs").read_text() == 'pub const SPEC_VERSION: &str = "1.2.3";\n'
    assert not (tmp_path / "spec_version.rs").exists()
    assert not (tmp_path / "mod.rs").exists()
    lib_text = (src / "lib.rs").read_text()
    assert "pub mod apis;" in lib_text
    assert "pub mod spec_version;" in lib_text


def test_spec_version_rust_marker_without_mod_rs(tmp_path: Path) -> None:
    # A partial payload with no mod.rs must still yield a usable declaration.
    generate.write_spec_version("rust", tmp_path, "1.2.3", generate.FULL_TARGETS["rust"])
    assert (tmp_path / "mod.rs").read_text() == "pub mod spec_version;\n"


def test_go_package_name_falls_back_to_target_path() -> None:
    target = generate.LanguageTarget(
        generator="go",
        additional_properties="withGoMod=false",
        sdk_repo="x/y",
        target_path="otari/client",
    )
    assert generate._go_package_name(target) == "client"
