"""The drift check between deploy/railway/template.json and the live Railway template."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_railway_template.py"
_TEMPLATE_PATH = _REPO_ROOT / "deploy" / "railway" / "template.json"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_railway_template", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check = _load()


def _snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = json.loads(_TEMPLATE_PATH.read_text())
    return snapshot


def _live_from(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Build a serializedConfig that agrees with the snapshot, shaped as Railway returns it."""
    services: dict[str, Any] = {}
    for index, service in enumerate(snapshot["services"].values()):
        deploy = service.get("deploy", {})
        live: dict[str, Any] = {
            "name": service["name"],
            "icon": None,
            "deploy": {
                "startCommand": deploy.get("startCommand"),
                "healthcheckPath": deploy.get("healthcheckPath"),
                "restartPolicyType": "ON_FAILURE",
            },
            "source": {"image": service["source"]["image"]},
            "variables": {
                name: {
                    "isOptional": variable["isOptional"],
                    "description": variable["description"],
                    "defaultValue": variable["defaultValue"],
                }
                for name, variable in service["variables"].items()
            },
        }
        networking = service.get("networking")
        if networking and networking.get("publicDomain"):
            port = networking["targetPort"]
            live["networking"] = {"serviceDomains": {f"<hasDomain>:{port}": {"port": port}}}
        services[f"00000000-0000-0000-0000-00000000000{index}"] = live
    return {"buckets": {}, "services": services}


def _live_service(live: dict[str, Any], name: str) -> dict[str, Any]:
    return next(service for service in live["services"].values() if service["name"] == name)


def test_committed_snapshot_agrees_with_a_matching_live_config() -> None:
    snapshot = _snapshot()

    assert check.diff_config(snapshot, _live_from(snapshot)) == []


def test_docker_hub_prefix_is_the_same_image() -> None:
    snapshot = _snapshot()
    live = _live_from(snapshot)
    otari = _live_service(live, "otari")
    otari["source"]["image"] = "docker.io/" + otari["source"]["image"]

    assert check.diff_config(snapshot, live) == []


def test_a_different_image_tag_is_drift() -> None:
    snapshot = _snapshot()
    live = _live_from(snapshot)
    _live_service(live, "otari")["source"]["image"] = "mzdotai/otari:0.1.0"

    [problem] = check.diff_config(snapshot, live)
    assert "otari" in problem
    assert "image" in problem
    assert "mzdotai/otari:0.1.0" in problem


def test_a_default_with_a_leading_space_is_drift() -> None:
    snapshot = _snapshot()
    live = _live_from(snapshot)
    variable = _live_service(live, "otari")["variables"]["OTARI_MASTER_KEY"]
    variable["defaultValue"] = " " + variable["defaultValue"]

    [problem] = check.diff_config(snapshot, live)
    assert "OTARI_MASTER_KEY" in problem
    assert "' ${{secret(48)}}'" in problem


def test_an_optional_flag_change_is_drift() -> None:
    snapshot = _snapshot()
    live = _live_from(snapshot)
    _live_service(live, "otari")["variables"]["PORT"]["isOptional"] = True

    [problem] = check.diff_config(snapshot, live)
    assert "PORT" in problem
    assert "optional" in problem


def test_a_reworded_description_is_drift_but_surrounding_whitespace_is_not() -> None:
    snapshot = _snapshot()
    live = _live_from(snapshot)
    variables = _live_service(live, "Postgres")["variables"]
    variables["DATABASE_URL"]["description"] = " " + variables["DATABASE_URL"]["description"]
    assert check.diff_config(snapshot, live) == []

    variables["PGPORT"]["description"] = "Something else."
    [problem] = check.diff_config(snapshot, live)
    assert "PGPORT" in problem
    assert "description" in problem


def test_a_variable_on_one_side_only_is_drift() -> None:
    snapshot = _snapshot()
    live = _live_from(snapshot)
    variables = _live_service(live, "otari")["variables"]
    del variables["OTARI_FORWARDED_ALLOW_IPS"]
    variables["OTARI_LOG_LEVEL"] = {"isOptional": True, "description": "x", "defaultValue": "debug"}

    problems = check.diff_config(snapshot, live)
    assert len(problems) == 2
    assert any("OTARI_FORWARDED_ALLOW_IPS" in p and "missing from the live template" in p for p in problems)
    assert any("OTARI_LOG_LEVEL" in p and "not in template.json" in p for p in problems)


def test_healthcheck_path_and_target_port_are_compared() -> None:
    snapshot = _snapshot()
    live = _live_from(snapshot)
    otari = _live_service(live, "otari")
    otari["deploy"]["healthcheckPath"] = "/health"
    otari["networking"] = {"serviceDomains": {"<hasDomain>:8080": {"port": 8080}}}

    problems = check.diff_config(snapshot, live)
    assert len(problems) == 2
    assert any("healthcheckPath" in p and "/health" in p for p in problems)
    assert any("public domain" in p and "8080" in p for p in problems)


def test_a_service_on_one_side_only_is_drift() -> None:
    snapshot = _snapshot()
    live = _live_from(snapshot)
    _live_service(live, "Postgres")["name"] = "Redis"

    problems = check.diff_config(snapshot, live)
    assert len(problems) == 2
    assert any("Postgres" in p and "missing from the live template" in p for p in problems)
    assert any("Redis" in p and "not in template.json" in p for p in problems)


def test_snapshot_is_not_mutated() -> None:
    snapshot = _snapshot()
    before = copy.deepcopy(snapshot)

    check.diff_config(snapshot, _live_from(snapshot))

    assert snapshot == before


def test_readme_ignores_trailing_whitespace_only() -> None:
    assert check.diff_readme("# Title\n\nBody\n", "# Title\n\nBody") == []

    diff = check.diff_readme("# Title\n\nBody\n", "# Title\n\nOther body\n")
    assert "-Body" in diff
    assert "+Other body" in diff


def test_main_reports_drift_and_exits_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    snapshot = _snapshot()
    live = _live_from(snapshot)
    _live_service(live, "otari")["source"]["image"] = "mzdotai/otari:latest"
    listing = tmp_path / "listing.md"
    listing.write_text("# Deploy and Host Otari on Railway\n")
    monkeypatch.setattr(
        check, "fetch_template", lambda code: {"readme": "# Something else\n", "serializedConfig": live}
    )

    code = check.main(["--listing", str(listing)])

    out = capsys.readouterr().out
    assert code == 1
    assert "mzdotai/otari:latest" in out
    assert "+# Something else" in out


def test_main_exits_zero_when_live_agrees(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snapshot = _snapshot()
    listing = tmp_path / "listing.md"
    listing.write_text("same\n")
    monkeypatch.setattr(
        check, "fetch_template", lambda code: {"readme": "same\n", "serializedConfig": _live_from(snapshot)}
    )

    assert check.main(["--listing", str(listing)]) == 0


def test_main_asks_for_the_code_in_template_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[str] = []
    snapshot = _snapshot()
    listing = tmp_path / "listing.md"
    listing.write_text("same\n")

    def fetch(code: str) -> dict[str, Any]:
        seen.append(code)
        return {"readme": "same\n", "serializedConfig": _live_from(snapshot)}

    monkeypatch.setattr(check, "fetch_template", fetch)
    check.main(["--listing", str(listing)])

    assert seen == [snapshot["code"]]


def test_main_exits_two_when_the_fetch_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(code: str) -> dict[str, Any]:
        raise check.FetchError("Template not found")

    monkeypatch.setattr(check, "fetch_template", fail)

    assert check.main([]) == 2
    assert "Template not found" in capsys.readouterr().err
