# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27"]
# ///
"""Compare two builds of Otari run on the same machine by ``./run.sh ab``.

    uv run ab.py cpu --out results/ab-.../runs/03-head.cpu-before.json
    uv run ab.py report results/ab-... [--enforce-timing] [--allow-regression]

``cpu`` sums ``process_cpu_seconds_total`` over both replicas. ``report`` reads
the runs ``./run.sh ab`` left in a directory (each run's loadgen result and the
CPU snapshots either side of it, plus each build's statement counts), writes
``report.md`` beside them, prints it, and exits non-zero when the head build
regressed:

- **Statements per request**, direct and spilled, may not grow by a whole
  statement. Background work (the reservation sweeper, usage-log flushes) adds
  fractions between runs of the same build, mostly to BEGIN and COMMIT, so the
  gate rounds each statement's count per request before summing, and also wants
  the raw total up by ``--min-statement-increase``, so one statement drifting
  across a half does not fail it alone. Counted, not timed, so always enforced.
- **Errors**: the head build must answer every request with a 200 and end every
  stream; the tenant has room on every limit, so a refusal counts too.
- **CPU per request** and **overhead p50** are timed, so a shared runner moves
  them. Each is compared against how far the base build's own runs spread, and
  fails only with ``--enforce-timing``; without it a regression is reported but
  does not fail the run.

``--allow-regression`` reports everything and exits 0, for a change whose cost
is deliberate.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import httpx

REPLICAS = ("http://otari-1:8000/metrics", "http://otari-2:8000/metrics")
CPU_LINE = re.compile(r"^process_cpu_seconds_total\s+(\S+)", re.MULTILINE)
VARIANTS = ("base", "head")


def cpu(out: Path) -> None:
    per_replica = {}
    for url in REPLICAS:
        match = CPU_LINE.search(httpx.get(url, timeout=10).text)
        per_replica[url] = float(match.group(1)) if match else None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"per_replica": per_replica}))


def _run_figures(result_path: Path) -> dict:
    """One run's figures: the loadgen result and the CPU it cost the replicas."""
    result = json.loads(result_path.read_text())
    stem = result_path.parent / result["label"]
    before = json.loads(Path(f"{stem}.cpu-before.json").read_text())["per_replica"]
    after = json.loads(Path(f"{stem}.cpu-after.json").read_text())["per_replica"]
    served = result["status_counts"].get("200", 0)
    cpu_seconds = sum(after[url] - before[url] for url in after if after[url] is not None and before.get(url))
    # The A/B tenant has room on every limit and budget, so any refusal is a failure too.
    errors = sum(n for status, n in result["status_counts"].items() if status != "200")
    return {
        "label": result["label"],
        "overhead_p50": result["gateway_overhead_ms"].get("p50"),
        "overhead_p99": result["gateway_overhead_ms"].get("p99"),
        "ttft_p50": result["ttft_ms"].get("p50"),
        "cpu_ms_per_request": cpu_seconds * 1000 / served if served else None,
        "errors": errors + result["streams_without_done"],
        "served": served,
    }


def _median(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else None


def _spread(values: list[float | None]) -> float:
    present = [v for v in values if v is not None]
    return max(present) - min(present) if len(present) > 1 else 0.0


def _fmt(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def report(directory: Path, args: argparse.Namespace) -> int:
    runs: dict[str, list[dict]] = {variant: [] for variant in VARIANTS}
    for path in sorted((directory / "runs").glob("*.json")):
        if path.name.endswith((".cpu-before.json", ".cpu-after.json")):
            continue
        figures = _run_figures(path)
        variant = figures["label"].rsplit("-", 1)[-1]
        runs[variant].append(figures)
    if not all(runs.values()):
        raise SystemExit(f"no runs for {[v for v in VARIANTS if not runs[v]]} in {directory}")

    statements = {}
    for variant in VARIANTS:
        for mode in ("direct", "spill"):
            path = directory / f"statements-{variant}-{mode}.json"
            statements[variant, mode] = json.loads(path.read_text()) if path.exists() else None

    failures: list[str] = []
    warnings: list[str] = []
    rows: list[tuple[str, str, str, str, str]] = []

    for mode in ("direct", "spill"):
        base_counts, head_counts = statements["base", mode], statements["head", mode]
        base = base_counts["per_request"] if base_counts else None
        head = head_counts["per_request"] if head_counts else None
        verdict = "n/a"
        if base_counts and head_counts:
            verdict = "ok"
            whole = head_counts["per_request_rounded"] - base_counts["per_request_rounded"]
            if whole >= 1 and head - base >= args.min_statement_increase:
                verdict = "**fail**"
                failures.append(f"statements per request ({mode}) went from {base:.1f} to {head:.1f}")
        rows.append((f"DB statements per request, {mode}", _fmt(base), _fmt(head), _delta(base, head), verdict))

    head_errors = sum(run["errors"] for run in runs["head"])
    base_errors = sum(run["errors"] for run in runs["base"])
    rows.append(("Failed requests", str(base_errors), str(head_errors), "", _ok(head_errors == 0)))
    if head_errors:
        failures.append(f"the head build failed {head_errors} requests (not a 200, or a stream cut short)")

    timed = [
        ("CPU ms per request", "cpu_ms_per_request", args.max_cpu_increase, True),
        ("Overhead p50 (ms)", "overhead_p50", args.max_overhead_increase, False),
    ]
    for title, key, allowance, relative in timed:
        base_values = [run[key] for run in runs["base"]]
        base, head = _median(base_values), _median([run[key] for run in runs["head"]])
        verdict = "n/a"
        if base is not None and head is not None:
            # A difference smaller than the base build's spread across its own runs
            # is noise this runner made, whatever the allowance says.
            limit = max(base * allowance if relative else allowance, args.noise_factor * _spread(base_values))
            verdict = "ok"
            if head - base > limit:
                message = f"{title} went from {base:.2f} to {head:.2f} (allowed +{limit:.2f})"
                if args.enforce_timing:
                    verdict = "**fail**"
                    failures.append(message)
                else:
                    verdict = "slower (not enforced)"
                    warnings.append(message)
        rows.append((title, _fmt(base, 2), _fmt(head, 2), _delta(base, head), verdict))

    for title, key in [("Overhead p99 (ms)", "overhead_p99"), ("Stream TTFT p50 (ms)", "ttft_p50")]:
        base, head = _median([run[key] for run in runs["base"]]), _median([run[key] for run in runs["head"]])
        rows.append((title, _fmt(base), _fmt(head), _delta(base, head), "reported only"))

    lines = [
        "## Load test: base vs head",
        "",
        "| | base | head | change | |",
        "|---|---:|---:|---:|---|",
        *[f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in rows],
        "",
        "Timed figures are the median over each build's runs; the builds alternated on one machine.",
        "",
        "<details><summary>Every run</summary>",
        "",
        "| run | overhead p50 | p99 | TTFT p50 | CPU ms/req | served | failed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    every = sorted(runs["base"] + runs["head"], key=lambda run: run["label"])
    lines += [
        f"| {r['label']} | {_fmt(r['overhead_p50'])} | {_fmt(r['overhead_p99'])} | {_fmt(r['ttft_p50'])} "
        f"| {_fmt(r['cpu_ms_per_request'], 2)} | {r['served']} | {r['errors']} |"
        for r in every
    ]
    lines += ["", "</details>", ""]
    if failures:
        lines += ["**Regressions:**", "", *[f"- {failure}" for failure in failures], ""]
        if args.allow_regression:
            lines += ["Allowed: this change is marked as accepting the cost.", ""]
    if warnings:
        lines += ["**Slower, not enforced:**", "", *[f"- {warning}" for warning in warnings], ""]
    if not failures and not warnings:
        lines += ["No regression.", ""]

    text = "\n".join(lines)
    (directory / "report.md").write_text(text)
    print(text)
    return 0 if args.allow_regression else len(failures)


def _delta(base: float | None, head: float | None) -> str:
    if base is None or head is None:
        return ""
    change = head - base
    percent = f" ({change / base * 100:+.0f}%)" if base else ""
    return f"{change:+.2f}{percent}"


def _ok(passed: bool) -> str:
    return "ok" if passed else "**fail**"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    cpu_parser = sub.add_parser("cpu", help="snapshot the replicas' CPU counters")
    cpu_parser.add_argument("--out", required=True)
    report_parser = sub.add_parser("report", help="compare base and head, exit non-zero on a regression")
    report_parser.add_argument("directory")
    report_parser.add_argument("--min-statement-increase", type=float, default=0.5)
    report_parser.add_argument("--max-cpu-increase", type=float, default=0.15, help="a fraction of base")
    report_parser.add_argument("--max-overhead-increase", type=float, default=2.0, help="milliseconds")
    report_parser.add_argument(
        "--noise-factor", type=float, default=2.0, help="times the base runs' spread a change must exceed"
    )
    report_parser.add_argument("--enforce-timing", action="store_true")
    report_parser.add_argument("--allow-regression", action="store_true")
    args = parser.parse_args()
    if args.step == "cpu":
        cpu(Path(args.out))
        return 0
    return report(Path(args.directory), args)


if __name__ == "__main__":
    raise SystemExit(main())
