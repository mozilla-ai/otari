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


def test_control_plane_tags_are_typed_management_only() -> None:
    assert generate.CONTROL_PLANE_TAGS == frozenset(
        {"keys", "users", "budgets", "pricing", "usage"}
    )
    # Excluded on purpose: proxy/inference surfaces and batches, all of which are
    # untyped in the spec (so generation would regress them).
    for excluded in ("chat", "responses", "embeddings", "batches"):
        assert excluded not in generate.CONTROL_PLANE_TAGS
