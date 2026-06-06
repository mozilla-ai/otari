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

    messages_field = schemas["ChatCompletionRequest"]["properties"]["messages"]
    assert messages_field["items"]["$ref"].endswith("/ChatMessageInput")


def test_control_plane_tags_are_typed_management_only() -> None:
    assert generate.CONTROL_PLANE_TAGS == frozenset(
        {"keys", "users", "budgets", "pricing", "usage"}
    )
    # Excluded on purpose: proxy/inference surfaces and batches, all of which are
    # untyped in the spec (so generation would regress them).
    for excluded in ("chat", "responses", "embeddings", "batches"):
        assert excluded not in generate.CONTROL_PLANE_TAGS
