"""Unit tests for the control-plane SDK codegen spec filter."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

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


def test_control_plane_tags_are_typed_management_only() -> None:
    assert generate.CONTROL_PLANE_TAGS == frozenset(
        {"keys", "users", "budgets", "pricing", "usage"}
    )
    # Excluded on purpose: proxy/inference surfaces and batches, all of which are
    # untyped in the spec (so generation would regress them).
    for excluded in ("chat", "responses", "embeddings", "batches"):
        assert excluded not in generate.CONTROL_PLANE_TAGS
