"""Unit tests for the endpoint-manifest parser and drift helpers.

Covers the pure logic in ``scripts/sdk_codegen/endpoint_manifest.py``. The gate
that applies it to the real spec lives in ``test_sdk_endpoint_coverage.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "sdk_codegen"))

import endpoint_manifest as em  # noqa: E402

MANIFEST = """\
# leading comment
[covered]
POST /v1/chat/completions
GET /v1/models

[excluded]
GET /v1/models/{model_id}    # redundant
post /v1/lowercase-verb
"""


def test_parse_splits_sections() -> None:
    covered, excluded = em.parse_manifest(MANIFEST)
    assert covered == {"POST /v1/chat/completions", "GET /v1/models"}
    assert "GET /v1/models/{model_id}" in excluded


def test_parse_strips_reason_trailers() -> None:
    _, excluded = em.parse_manifest(MANIFEST)
    assert "GET /v1/models/{model_id}" in excluded
    assert not any("redundant" in entry for entry in excluded)


def test_parse_uppercases_the_verb() -> None:
    _, excluded = em.parse_manifest(MANIFEST)
    assert "POST /v1/lowercase-verb" in excluded


def test_parse_ignores_entries_before_any_section() -> None:
    covered, excluded = em.parse_manifest("GET /v1/orphan\n[covered]\nGET /v1/real\n")
    assert covered == {"GET /v1/real"}
    assert not excluded


def test_spec_endpoints_skips_meta_and_non_verbs() -> None:
    spec = {
        "paths": {
            "/health": {"get": {}},
            "/health/ready": {"get": {}},
            "/healthz-not-meta": {"get": {}},
            "/v1/keys": {"get": {}, "post": {}, "parameters": [], "summary": "x"},
        }
    }
    assert em.spec_endpoints(spec) == {
        "GET /healthz-not-meta",
        "GET /v1/keys",
        "POST /v1/keys",
    }


def test_spec_endpoints_handles_missing_paths() -> None:
    assert em.spec_endpoints({}) == set()


def test_unaccounted_reports_only_unlisted() -> None:
    endpoints = {"GET /a", "GET /b", "GET /c"}
    assert em.unaccounted(endpoints, {"GET /a"}, {"GET /b"}) == ["GET /c"]


def test_unaccounted_is_empty_when_fully_classified() -> None:
    endpoints = {"GET /a", "GET /b"}
    assert em.unaccounted(endpoints, {"GET /a"}, {"GET /b"}) == []


def test_stale_reports_entries_absent_from_spec() -> None:
    assert em.stale({"GET /a"}, {"GET /a"}, {"GET /gone"}) == ["GET /gone"]


def test_canonical_manifest_parses_and_is_disjoint() -> None:
    covered, excluded = em.load_manifest()
    assert covered and excluded
    assert not covered & excluded
