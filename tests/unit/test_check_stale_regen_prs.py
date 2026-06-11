"""Unit tests for the stale SDK regeneration PR alert (pure selection/render)."""

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_stale_regen_prs.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_stale_regen_prs", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check = _load()

_NOW = datetime(2026, 6, 11, tzinfo=timezone.utc)


def _pr(repo: str, number: int, age_days: float) -> dict[str, Any]:
    created = _NOW - timedelta(days=age_days)
    return {
        "repo": repo,
        "number": number,
        "title": "Regenerate SDK client core",
        "url": f"https://github.com/{repo}/pull/{number}",
        "createdAt": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def test_parse_iso8601_handles_z_suffix() -> None:
    parsed = check.parse_iso8601("2026-06-01T12:00:00Z")
    assert parsed == datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_select_stale_keeps_only_prs_past_threshold() -> None:
    prs = [
        _pr("mozilla-ai/otari-sdk-python", 1, age_days=10),
        _pr("mozilla-ai/otari-sdk-go", 2, age_days=3),
    ]
    stale = check.select_stale(prs, max_age_days=7, now=_NOW)
    assert [pr["number"] for pr in stale] == [1]


def test_select_stale_threshold_is_strict() -> None:
    # A PR exactly at the threshold is not yet stale; just past it is.
    at_threshold = _pr("mozilla-ai/otari-sdk-ts", 5, age_days=7)
    just_over = _pr("mozilla-ai/otari-sdk-ts", 6, age_days=7.01)
    stale = check.select_stale([at_threshold, just_over], max_age_days=7, now=_NOW)
    assert [pr["number"] for pr in stale] == [6]


def test_select_stale_sorts_oldest_first() -> None:
    prs = [
        _pr("a/b", 1, age_days=9),
        _pr("c/d", 2, age_days=30),
        _pr("e/f", 3, age_days=15),
    ]
    stale = check.select_stale(prs, max_age_days=7, now=_NOW)
    # createdAt ascending == oldest first.
    assert [pr["number"] for pr in stale] == [2, 3, 1]


def test_select_stale_empty_when_all_fresh() -> None:
    prs = [_pr("a/b", 1, age_days=1), _pr("c/d", 2, age_days=6)]
    assert check.select_stale(prs, max_age_days=7, now=_NOW) == []


def test_render_report_lists_each_stale_pr() -> None:
    prs = [_pr("mozilla-ai/otari-sdk-python", 12, age_days=10)]
    report = check.render_report(prs, max_age_days=7, now=_NOW)
    assert "mozilla-ai/otari-sdk-python" in report
    assert "[#12](https://github.com/mozilla-ai/otari-sdk-python/pull/12)" in report
    assert "| Repo | PR | Age (days) | Opened |" in report


def test_render_report_when_none_stale() -> None:
    report = check.render_report([], max_age_days=7, now=_NOW)
    assert "within the freshness window" in report
    assert "|" not in report  # no table rendered
