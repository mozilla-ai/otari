"""The spec generator tells contributors which new routes need SDK classification."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_ROOT = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT / "sdk_codegen"))

import endpoint_manifest  # noqa: E402

import generate_openapi  # noqa: E402


@pytest.mark.parametrize("check", [False, True])
def test_generator_warns_about_unclassified_new_endpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], check: bool
) -> None:
    covered, _ = endpoint_manifest.load_manifest()
    known = sorted(covered)[0]
    method, path = known.split(" ", 1)
    spec: dict[str, object] = {"paths": {path: {method.lower(): {}}, "/api/v1/new-route": {"post": {}, "get": {}}}}
    monkeypatch.setattr(generate_openapi, "generate_openapi_spec", lambda: spec)
    output = tmp_path / "openapi.json"
    if check:
        generate_openapi.write_spec(spec, output)
    argv = ["generate_openapi.py", "--output", str(output)]
    if check:
        argv.append("--check")
    monkeypatch.setattr(sys, "argv", argv)

    assert generate_openapi.main() == 0
    warning = capsys.readouterr().err
    assert "GET /api/v1/new-route" in warning
    assert "POST /api/v1/new-route" in warning
    assert "scripts/sdk_codegen/sdk-endpoints.txt" in warning
    assert known not in warning


def test_generator_does_not_warn_when_endpoints_are_classified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    covered, _ = endpoint_manifest.load_manifest()
    method, path = sorted(covered)[0].split(" ", 1)
    monkeypatch.setattr(generate_openapi, "generate_openapi_spec", lambda: {"paths": {path: {method.lower(): {}}}})
    monkeypatch.setattr(sys, "argv", ["generate_openapi.py", "--output", str(tmp_path / "spec.json")])

    assert generate_openapi.main() == 0
    assert capsys.readouterr().err == ""
