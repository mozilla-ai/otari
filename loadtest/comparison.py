"""The checks over one ``./run.sh BASE HEAD`` run: did head get slower, or do more?

With LOADTEST_ACCEPT_REGRESSION=1 (the `perf-accepted` label) a regression is
an expected failure instead. ``report.md`` and ``perf.json`` hold the figures.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from ab import SCENARIOS, analyze

PERF = analyze(Path(os.environ["LOADTEST_AB_DIR"]))
ACCEPTED = os.environ.get("LOADTEST_ACCEPT_REGRESSION") == "1"


def _regressed(message: str) -> None:
    if ACCEPTED:
        pytest.xfail(f"accepted with perf-accepted: {message}")
    pytest.fail(message)


def test_every_planned_run_completed() -> None:
    """A turn that crashed or served nothing would otherwise shrink the comparison."""
    assert not PERF["missing_runs"], "; ".join(PERF["missing_runs"])


def test_head_served_every_request() -> None:
    """The tenants have room on every limit, so any refusal or cut stream is a failure."""
    assert PERF["head_errors"] == 0, f"head failed {PERF['head_errors']} requests"


@pytest.mark.parametrize("mode", ["direct", "spill"])
def test_no_extra_database_statements(mode: str) -> None:
    counted = PERF["statements"][mode]
    assert counted is not None, "statements were not counted for both builds"
    if counted["verdict"] == "regression":
        changed = "; ".join(f"{c['base']} → {c['head']} {c['query']}" for c in counted["changed"])
        _regressed(f"data statements per request {counted['base']} → {counted['head']}: {changed}")


@pytest.mark.parametrize("route", list(SCENARIOS))
def test_no_latency_regression(route: str) -> None:
    cell = PERF["scenarios"][route]
    if cell is None:
        pytest.fail("no runs to compare")
    if cell["verdict"] == "regression":
        turns = cell["turn_p50s"]
        _regressed(f"every head turn slower than every base turn: p50 base {turns['base']}, head {turns['head']} ms")
