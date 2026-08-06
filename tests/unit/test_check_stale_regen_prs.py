"""Unit tests for the SDK regeneration PR alert (pure selection/render)."""

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


def _run(conclusion: str = "SUCCESS", status: str = "COMPLETED") -> dict[str, Any]:
    return {"__typename": "CheckRun", "name": "CI", "status": status, "conclusion": conclusion}


def _pr(
    repo: str,
    number: int,
    age_days: float,
    rollup: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    created = _NOW - timedelta(days=age_days)
    return {
        "repo": repo,
        "number": number,
        "title": "Regenerate SDK client core",
        "url": f"https://github.com/{repo}/pull/{number}",
        "createdAt": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "statusCheckRollup": [_run()] if rollup is None else rollup,
    }


def test_parse_iso8601_handles_z_suffix() -> None:
    parsed = check.parse_iso8601("2026-06-01T12:00:00Z")
    assert parsed == datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


# --- check_state ---------------------------------------------------------


def test_check_state_passing_when_all_succeed() -> None:
    assert check.check_state(_pr("a/b", 1, 1, [_run(), _run()])) == "passing"


def test_check_state_failing_on_any_failure() -> None:
    assert check.check_state(_pr("a/b", 1, 1, [_run(), _run("FAILURE")])) == "failing"


def test_check_state_pending_when_a_run_is_incomplete() -> None:
    pr = _pr("a/b", 1, 1, [_run(), _run(conclusion="", status="IN_PROGRESS")])
    assert check.check_state(pr) == "pending"


def test_check_state_failure_outranks_pending() -> None:
    pr = _pr("a/b", 1, 1, [_run("FAILURE"), _run(conclusion="", status="QUEUED")])
    assert check.check_state(pr) == "failing"


def test_check_state_ignores_cancelled_and_skipped() -> None:
    """A fail-fast matrix cancels siblings; that is noise, not a separate break."""
    pr = _pr("a/b", 1, 1, [_run("SUCCESS"), _run("CANCELLED"), _run("SKIPPED")])
    assert check.check_state(pr) == "passing"


def test_check_state_handles_status_context_entries() -> None:
    pr = _pr("a/b", 1, 1, [{"__typename": "StatusContext", "state": "FAILURE"}])
    assert check.check_state(pr) == "failing"


def test_check_state_unknown_without_any_checks() -> None:
    assert check.check_state(_pr("a/b", 1, 1, [])) == "unknown"


def test_check_state_unknown_is_not_flagged() -> None:
    assert check.flag_reasons(_pr("a/b", 1, age_days=1, rollup=[]), 7, _NOW) == []


# --- selection -----------------------------------------------------------


def test_failing_pr_is_flagged_regardless_of_age() -> None:
    """The #438 case: red on day one, but age would not have caught it for a week."""
    pr = _pr("mozilla-ai/otari-sdk-go", 1, age_days=0.5, rollup=[_run("FAILURE")])
    flagged = check.select_flagged([pr], max_age_days=7, now=_NOW)
    assert [p["number"] for p in flagged] == [1]
    assert flagged[0]["reasons"] == ["checks failing"]


def test_green_young_pr_is_not_flagged() -> None:
    assert check.select_flagged([_pr("a/b", 1, age_days=3)], max_age_days=7, now=_NOW) == []


def test_green_old_pr_is_flagged_on_age() -> None:
    flagged = check.select_flagged([_pr("a/b", 1, age_days=10)], max_age_days=7, now=_NOW)
    assert flagged[0]["reasons"] == ["open 10.0 days"]


def test_old_and_failing_pr_reports_both_reasons() -> None:
    pr = _pr("a/b", 1, age_days=18, rollup=[_run("FAILURE")])
    flagged = check.select_flagged([pr], max_age_days=7, now=_NOW)
    assert flagged[0]["reasons"] == ["checks failing", "open 18.0 days"]


def test_age_threshold_is_strict() -> None:
    at_threshold = _pr("a/b", 5, age_days=7)
    just_over = _pr("a/b", 6, age_days=7.01)
    flagged = check.select_flagged([at_threshold, just_over], max_age_days=7, now=_NOW)
    assert [p["number"] for p in flagged] == [6]


def test_pending_checks_do_not_flag_a_young_pr() -> None:
    pr = _pr("a/b", 1, age_days=1, rollup=[_run(conclusion="", status="IN_PROGRESS")])
    assert check.select_flagged([pr], max_age_days=7, now=_NOW) == []


def test_select_flagged_sorts_oldest_first() -> None:
    prs = [
        _pr("a/b", 1, age_days=9),
        _pr("c/d", 2, age_days=30),
        _pr("e/f", 3, age_days=15),
    ]
    flagged = check.select_flagged(prs, max_age_days=7, now=_NOW)
    assert [p["number"] for p in flagged] == [2, 3, 1]


def test_select_flagged_does_not_mutate_input() -> None:
    pr = _pr("a/b", 1, age_days=10)
    check.select_flagged([pr], max_age_days=7, now=_NOW)
    assert "reasons" not in pr


# --- rendering -----------------------------------------------------------


def test_render_report_lists_each_flagged_pr() -> None:
    prs = check.select_flagged(
        [_pr("mozilla-ai/otari-sdk-python", 12, age_days=10)], max_age_days=7, now=_NOW
    )
    report = check.render_report(prs, max_age_days=7, now=_NOW)
    assert "mozilla-ai/otari-sdk-python" in report
    assert "[#12](https://github.com/mozilla-ai/otari-sdk-python/pull/12)" in report
    assert "| Repo | PR | Age (days) | Checks | Why |" in report


def test_render_report_leads_with_the_red_prs() -> None:
    prs = check.select_flagged(
        [
            _pr("a/b", 1, age_days=18, rollup=[_run("FAILURE")]),
            _pr("c/d", 2, age_days=18),
        ],
        max_age_days=7,
        now=_NOW,
    )
    report = check.render_report(prs, max_age_days=7, now=_NOW)
    assert "1 of 2 cannot merge until CI is fixed" in report
    assert "| failing |" in report
    assert "| passing |" in report


def test_render_report_omits_the_ci_lead_when_all_are_green() -> None:
    prs = check.select_flagged([_pr("a/b", 1, age_days=10)], max_age_days=7, now=_NOW)
    report = check.render_report(prs, max_age_days=7, now=_NOW)
    assert "cannot merge until CI is fixed" not in report


def test_render_report_when_none_flagged() -> None:
    report = check.render_report([], max_age_days=7, now=_NOW)
    assert "within the freshness window" in report
    assert "|" not in report  # no table rendered
