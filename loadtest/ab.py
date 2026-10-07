# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27", "numpy>=2"]
# ///
"""Compare two builds of Otari run on the same machine by ``./run.sh ab``.

    uv run ab.py cpu --out results/ab-.../runs/03-head-direct-idle.cpu-before.json
    uv run ab.py report results/ab-... [--enforce-latency] [--allow-regression]

``cpu`` snapshots the replica's CPU counter. ``report`` writes ``perf.json`` and
renders ``report.md`` from it, and exits non-zero on a regression. The method
(the bootstrap interval, the verdicts, what is judged and what fails the run)
is in README.md, "Comparing two builds".
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

import httpx
import numpy as np

REPLICA_METRICS = "http://otari-1:8000/metrics"
CPU_LINE = re.compile(r"^process_cpu_seconds_total\s+(\S+)", re.MULTILINE)
VARIANTS = ("base", "head")
MODES = ("idle", "loaded")
QUANTILES = {"p50": 50, "p90": 90}
# What measures each scenario: the overhead kind, and the fake provider slot that
# must have served the request (a spill scenario's requests that did not spill
# measure something else).
SCENARIO_SAMPLES = {
    "direct": ("overhead_ms", "m2"),
    "pooled": ("overhead_ms", "m2"),
    "spill": ("overhead_ms", "m2"),
    "stream": ("ttft_overhead_ms", "m2"),
}
TRANSACTION_CONTROL = {"BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE"}
RUN_LABEL = re.compile(r"^(?P<seq>\d+)-(?P<variant>base|head)-(?P<scenario>[a-z0-9_]+)-(?P<mode>idle|loaded)$")


def cpu(out: Path) -> None:
    match = CPU_LINE.search(httpx.get(REPLICA_METRICS, timeout=10).text)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cpu_seconds": float(match.group(1)) if match else None}))


def _load_run(path: Path) -> dict[str, Any] | None:
    result = json.loads(path.read_text())
    label = RUN_LABEL.match(result["label"])
    if label is None:
        return None
    stem = path.parent / result["label"]
    before = json.loads(Path(f"{stem}.cpu-before.json").read_text())["cpu_seconds"]
    after = json.loads(Path(f"{stem}.cpu-after.json").read_text())["cpu_seconds"]
    kind, slot = SCENARIO_SAMPLES.get(label["scenario"], ("overhead_ms", None))
    by_slot = result.get("samples", {}).get(kind, {})
    if slot is not None:
        samples = by_slot.get(slot, [])
    else:
        samples = [v for s, values in by_slot.items() if s != "m3" for v in values]
    served = result["status_counts"].get("200", 0)
    errors = sum(n for status, n in result["status_counts"].items() if status != "200")
    cpu_ms = (after - before) * 1000 / served if served and before is not None and after is not None else None
    return {
        "label": result["label"],
        "seq": int(label["seq"]),
        "variant": label["variant"],
        "scenario": label["scenario"],
        "mode": label["mode"],
        "metric": kind,
        "samples": np.asarray(samples, dtype=float),
        "errors": errors + result.get("streams_without_done", 0),
        "cpu_ms_per_request": cpu_ms,
    }


def _hierarchical_bootstrap(
    base_runs: list[np.ndarray], head_runs: list[np.ndarray], iterations: int, rng: np.random.Generator
) -> dict[str, tuple[float, float]]:
    """95% intervals for head minus base at each quantile."""
    diffs = {q: np.empty(iterations) for q in QUANTILES}
    for i in range(iterations):
        pooled = []
        for runs in (base_runs, head_runs):
            picked = rng.integers(0, len(runs), len(runs))
            pooled.append(np.concatenate([rng.choice(runs[j], size=len(runs[j])) for j in picked]))
        for q, pct in QUANTILES.items():
            diffs[q][i] = np.percentile(pooled[1], pct) - np.percentile(pooled[0], pct)
    return {q: (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) for q, d in diffs.items()}


def _verdict(low: float, high: float, threshold: float) -> str:
    if low > threshold:
        return "regression"
    if high < -threshold:
        return "improvement"
    if low >= -threshold and high <= threshold:
        return "unchanged"
    return "inconclusive"


def _median(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else None


def _r(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)


def _scenario_order(scenario: str) -> tuple[int, str]:
    known = list(SCENARIO_SAMPLES)
    return (known.index(scenario) if scenario in known else len(known), scenario)


def compare_latency(runs: list[dict], args: argparse.Namespace) -> list[dict]:
    rng = np.random.default_rng(args.seed)
    cells = []
    for scenario in sorted({run["scenario"] for run in runs}, key=_scenario_order):
        for mode in MODES:
            group = sorted((r for r in runs if r["scenario"] == scenario and r["mode"] == mode), key=lambda r: r["seq"])
            by_variant = {v: [r for r in group if r["variant"] == v and len(r["samples"])] for v in VARIANTS}
            if not all(by_variant.values()):
                continue
            pooled = {v: np.concatenate([r["samples"] for r in by_variant[v]]) for v in VARIANTS}
            intervals = _hierarchical_bootstrap(
                [r["samples"] for r in by_variant["base"]],
                [r["samples"] for r in by_variant["head"]],
                args.iterations,
                rng,
            )
            # How far apart one build's own runs are: the same code, so any spread
            # is the machine. A bootstrap cannot tell a disturbed run from a slow
            # build when the disturbance lands on one build's runs more than the other's.
            run_p50s = {v: [float(np.percentile(r["samples"], 50)) for r in by_variant[v]] for v in VARIANTS}
            spread = {v: max(p) / min(p) - 1 if min(p) > 0 else None for v, p in run_p50s.items()}
            noisy = any(s is None or s > args.max_run_spread for s in spread.values())
            cell: dict[str, Any] = {
                "scenario": scenario,
                "mode": mode,
                "metric": group[0]["metric"],
                "n": {v: int(len(pooled[v])) for v in VARIANTS},
                "errors": {v: sum(r["errors"] for r in group if r["variant"] == v) for v in VARIANTS},
                "run_spread": {v: _r(s, 3) for v, s in spread.items()},
            }
            for q, pct in QUANTILES.items():
                base, head = float(np.percentile(pooled["base"], pct)), float(np.percentile(pooled["head"], pct))
                threshold = max(args.min_effect_ms, abs(base) * args.min_effect_pct / 100)
                low, high = intervals[q]
                cell[q] = {
                    "base": _r(base),
                    "head": _r(head),
                    "delta": _r(head - base),
                    "delta_pct": _r((head - base) / base * 100, 1) if base else None,
                    "ci95": [_r(low), _r(high)],
                    "threshold": _r(threshold),
                    "verdict": "noisy" if noisy else _verdict(low, high, threshold),
                }
            cpu_base = _median([r["cpu_ms_per_request"] for r in group if r["variant"] == "base"])
            cpu_head = _median([r["cpu_ms_per_request"] for r in group if r["variant"] == "head"])
            cell["cpu_ms_per_request"] = {
                "base": _r(cpu_base, 3),
                "head": _r(cpu_head, 3),
                "delta_pct": _r((cpu_head - cpu_base) / cpu_base * 100, 1) if cpu_base and cpu_head else None,
            }
            cell["runs"] = [
                {
                    "label": r["label"],
                    "variant": r["variant"],
                    "n": int(len(r["samples"])),
                    "p50": _r(float(np.percentile(r["samples"], 50))) if len(r["samples"]) else None,
                    "p90": _r(float(np.percentile(r["samples"], 90))) if len(r["samples"]) else None,
                    "cpu_ms_per_request": _r(r["cpu_ms_per_request"], 3),
                    "errors": r["errors"],
                }
                for r in group
            ]
            cells.append(cell)
    return cells


def compare_statements(directory: Path, args: argparse.Namespace) -> dict[str, dict]:
    out = {}
    for mode in ("direct", "spill"):
        counts = {}
        for variant in VARIANTS:
            path = directory / f"statements-{variant}-{mode}.json"
            counts[variant] = json.loads(path.read_text()) if path.exists() else None
        if not all(counts.values()):
            continue
        base, head = counts["base"], counts["head"]
        per_query = {v: {query: n for n, query in counts[v]["statements"]} for v in VARIANTS}
        changed = []
        for query in sorted(set(per_query["base"]) | set(per_query["head"])):
            b, h = round(per_query["base"].get(query, 0)), round(per_query["head"].get(query, 0))
            if b != h:
                changed.append({"query": query, "base": b, "head": h})
        # Transaction control is left out of the verdict: background transactions
        # (the reservation sweeper, usage-log flushes) land in the count as fractions
        # of a BEGIN, COMMIT or ROLLBACK per request, and on identical builds have
        # moved BEGIN across a rounding line. Each transaction carries data statements,
        # which are counted.
        data = {
            v: [n for query, n in per_query[v].items() if query.split(" ", 1)[0].upper() not in TRANSACTION_CONTROL]
            for v in VARIANTS
        }
        whole = sum(round(n) for n in data["head"]) - sum(round(n) for n in data["base"])
        raw = sum(data["head"]) - sum(data["base"])
        if whole >= 1 and raw >= args.min_statement_increase:
            verdict = "regression"
        elif whole <= -1 and raw <= -args.min_statement_increase:
            verdict = "improvement"
        else:
            verdict = "unchanged"
        out[mode] = {
            "base": base["per_request"],
            "head": head["per_request"],
            "delta": _r(head["per_request"] - base["per_request"]),
            "data": {"base": _r(sum(data["base"])), "head": _r(sum(data["head"])), "delta": _r(raw)},
            "changed": changed,
            "verdict": verdict,
        }
    return out


def _overall(verdicts: list[str]) -> str:
    if "noisy" in verdicts:
        return "noisy"
    if "regression" in verdicts and "improvement" in verdicts:
        return "mixed"
    for verdict in ("regression", "improvement", "inconclusive"):
        if verdict in verdicts:
            return verdict
    return "unchanged"


def report(directory: Path, args: argparse.Namespace) -> int:
    runs = [
        run
        for path in sorted((directory / "runs").glob("*.json"))
        if not path.name.endswith((".cpu-before.json", ".cpu-after.json")) and (run := _load_run(path)) is not None
    ]
    if not {r["variant"] for r in runs} >= set(VARIANTS):
        raise SystemExit(f"need runs of both builds in {directory}")
    meta_path = directory / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    latency = compare_latency(runs, args)
    statements = compare_statements(directory, args)

    failures: list[str] = []
    # A run that crashed leaves no result, which would compare on less than was
    # intended rather than fail; every scenario and mode must have run as often
    # for each build.
    planned = {(r["scenario"], r["mode"]) for r in runs}
    for scenario, mode in sorted(planned):
        counts = {
            v: sum(1 for r in runs if (r["scenario"], r["mode"], r["variant"]) == (scenario, mode, v)) for v in VARIANTS
        }
        if counts["base"] != counts["head"] or not all(counts.values()):
            failures.append(f"{scenario} {mode}: runs missing (base {counts['base']}, head {counts['head']})")
    for mode, figures in statements.items():
        if figures["verdict"] == "regression":
            failures.append(f"statements per request ({mode}) went from {figures['base']:.1f} to {figures['head']:.1f}")
    head_errors = sum(cell["errors"]["head"] for cell in latency)
    if head_errors:
        failures.append(f"the head build failed {head_errors} requests (not a 200, or a stream cut short)")
    if args.enforce_latency:
        failures += [
            f"{cell['scenario']} {cell['mode']}: p50 overhead {cell['p50']['delta']:+} ms "
            f"(95% CI {cell['p50']['ci95'][0]} to {cell['p50']['ci95'][1]})"
            for cell in latency
            if cell["p50"]["verdict"] == "regression"
        ]

    verdicts = [cell["p50"]["verdict"] for cell in latency] + [s["verdict"] for s in statements.values()]
    if head_errors:
        verdicts.append("regression")
    perf = {
        "schema": 1,
        **meta,
        "thresholds": {
            "min_effect_ms": args.min_effect_ms,
            "min_effect_pct": args.min_effect_pct,
            "max_run_spread": args.max_run_spread,
            "min_statement_increase": args.min_statement_increase,
            "bootstrap_iterations": args.iterations,
        },
        "verdict": _overall(verdicts),
        "failures": failures,
        "latency_enforced": args.enforce_latency,
        "accepted": args.allow_regression,
        "scenarios": latency,
        "statements": statements,
    }
    (directory / "perf.json").write_text(json.dumps(perf, indent=2))
    text = render(perf)
    (directory / "report.md").write_text(text)
    print(text)
    return 0 if args.allow_regression else len(failures)


VERDICT_MARK = {
    "regression": "🔴 regression",
    "improvement": "🟢 improvement",
    "unchanged": "unchanged",
    "inconclusive": "⚪ inconclusive",
    "mixed": "🟠 mixed",
    "noisy": "🟡 noisy (rerun)",
}


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def _signed(value: float | None, suffix: str = "") -> str:
    return "" if value is None else f"{value:+.1f}{suffix}"


def render(perf: dict) -> str:
    lines = [f"## Load test, base vs head: {VERDICT_MARK[perf['verdict']]}", ""]
    if "base" in perf or "head" in perf:
        lines += [f"base `{perf.get('base', {}).get('ref', '?')}`, head `{perf.get('head', {}).get('ref', '?')}`", ""]
    lines += [
        "Gateway overhead in ms: the client's latency minus the fake provider's (time to first token when "
        "streamed). `idle` is one request at a time, `loaded` an open-loop rate. A change is head minus base, "
        "with its 95% bootstrap interval.",
        "",
        "| scenario | mode | p50 base | p50 head | p50 change [95% CI] | p90 change [95% CI] | CPU/req | verdict |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for cell in perf["scenarios"]:
        p50, p90 = cell["p50"], cell["p90"]
        p50_ci, p90_ci = (f"[{_ms(q['ci95'][0])}, {_ms(q['ci95'][1])}]" for q in (p50, p90))
        lines.append(
            f"| {cell['scenario']} | {cell['mode']} | {_ms(p50['base'])} | {_ms(p50['head'])} "
            f"| {_signed(p50['delta'])} ({_signed(p50['delta_pct'], '%')}) {p50_ci} "
            f"| {_signed(p90['delta'])} {p90_ci} "
            f"| {_signed(cell['cpu_ms_per_request']['delta_pct'], '%')} | {VERDICT_MARK[p50['verdict']]} |"
        )
    if perf["statements"]:
        lines += [
            "",
            "| DB statements per request | all, base → head | data statements, base → head | verdict |",
            "|---|---:|---:|---|",
        ]
        for mode, s in perf["statements"].items():
            d = s["data"]
            lines.append(
                f"| {mode} | {s['base']:.1f} → {s['head']:.1f} ({s['delta']:+.1f}) "
                f"| {d['base']:.1f} → {d['head']:.1f} ({d['delta']:+.1f}) | {VERDICT_MARK[s['verdict']]} |"
            )
        lines += ["", "Judged on data statements; BEGIN, COMMIT and ROLLBACK pick up background transactions."]
        changed = [(mode, c) for mode, s in perf["statements"].items() for c in s["changed"]]
        if changed:
            lines += ["", "<details><summary>Statements that changed, per request</summary>", ""]
            lines += [f"- {mode}: {c['base']} → {c['head']} `{c['query']}`" for mode, c in changed]
            lines += ["", "</details>"]
    lines.append("")
    t = perf["thresholds"]
    lines += [
        f"A change counts once its whole interval is beyond max({t['min_effect_ms']} ms, {t['min_effect_pct']}% of "
        f"base). Latency {'fails' if perf['latency_enforced'] else 'is reported but does not fail'} the job. "
        "Machine-readable: `perf.json` beside this report.",
        "",
    ]
    if perf["failures"]:
        lines += ["**Failing:**", "", *[f"- {f}" for f in perf["failures"]], ""]
        if perf["accepted"]:
            lines += ["Allowed: this change is marked as accepting the cost (`perf-accepted`).", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    cpu_parser = sub.add_parser("cpu", help="snapshot the replica's CPU counter")
    cpu_parser.add_argument("--out", required=True)
    report_parser = sub.add_parser("report", help="compare base and head, exit non-zero on a regression")
    report_parser.add_argument("directory")
    report_parser.add_argument("--min-effect-ms", type=float, default=1.0)
    # What one replica on a 4-core budget resolves: identical builds have differed
    # by up to 6% at p50 between two A/A runs on a quiet machine.
    report_parser.add_argument("--min-effect-pct", type=float, default=10.0)
    report_parser.add_argument(
        "--max-run-spread", type=float, default=0.25, help="how far one build's runs may disagree, as a fraction"
    )
    report_parser.add_argument("--min-statement-increase", type=float, default=0.5)
    report_parser.add_argument("--iterations", type=int, default=2000)
    report_parser.add_argument("--seed", type=int, default=0)
    report_parser.add_argument("--enforce-latency", action="store_true")
    report_parser.add_argument("--allow-regression", action="store_true")
    args = parser.parse_args()
    if args.step == "cpu":
        cpu(Path(args.out))
        return 0
    return report(Path(args.directory), args)


if __name__ == "__main__":
    raise SystemExit(main())
