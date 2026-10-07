# /// script
# requires-python = ">=3.12"
# dependencies = ["psycopg[binary]>=3.2"]
# ///
"""Profile a run from the outside: Postgres's statements and lock waits.

    uv run profile.py begin  --out results/profile-steady
    uv run profile.py sample --out results/profile-steady --seconds 240
    uv run profile.py end    --out results/profile-steady
    uv run profile.py statements --db ab_head --requests 100 --json-out results/ab/statements-head.json

``begin`` zeroes pg_stat_statements. ``sample`` polls pg_stat_activity during the
run, so lock waits on a shared budget row show up as what each backend was
waiting on, with the query that was blocked. ``end`` writes report.md: the
statements by total time, the wait events, and links to the py-spy recordings run.sh's PROFILE=1 leaves in the same
directory (open them in speedscope.app). ``statements`` reads pg_stat_statements
as statements per request, for a run of a known number of requests.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import psycopg

DSN = "postgresql://otari:otari@postgres:5432/otari"


def reset_statements() -> None:
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
        conn.execute("SELECT pg_stat_statements_reset()")


def begin(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    reset_statements()
    (out / "began_at").write_text(str(time.time()))
    print(f"profiling into {out}")


def sample(out: Path, seconds: float, interval: float) -> None:
    """Poll what every gateway backend is doing; write the counts as they grow."""
    waits: Counter[str] = Counter()
    blocked_queries: Counter[str] = Counter()
    max_active = samples = 0
    deadline = time.time() + seconds
    with psycopg.connect(DSN, autocommit=True) as conn:
        while time.time() < deadline:
            rows = conn.execute(
                """
                SELECT state, wait_event_type, wait_event, left(query, 160),
                       cardinality(pg_blocking_pids(pid)) > 0
                FROM pg_stat_activity
                WHERE datname = 'otari' AND pid <> pg_backend_pid() AND backend_type = 'client backend'
                """
            ).fetchall()
            samples += 1
            active = [row for row in rows if row[0] == "active"]
            max_active = max(max_active, len(active))
            for _, wait_type, wait_event, query, blocked in active:
                waits[f"{wait_type or 'CPU'}:{wait_event or 'running'}"] += 1
                if blocked:
                    blocked_queries[query] += 1
            if samples % 20 == 0:
                _write_samples(out, samples, interval, max_active, waits, blocked_queries)
            time.sleep(interval)
    _write_samples(out, samples, interval, max_active, waits, blocked_queries)


def _write_samples(
    out: Path, samples: int, interval: float, max_active: int, waits: Counter[str], blocked: Counter[str]
) -> None:
    (out / "pg-samples.json").write_text(
        json.dumps(
            {
                "samples": samples,
                "interval_sec": interval,
                "max_active_backends": max_active,
                "waits": waits.most_common(),
                "blocked_queries": blocked.most_common(15),
            },
            indent=2,
        )
    )


def end(out: Path) -> None:
    elapsed = time.time() - float((out / "began_at").read_text())
    with psycopg.connect(DSN) as conn:
        statements = conn.execute(
            """
            SELECT calls, round(total_exec_time::numeric, 1), round(mean_exec_time::numeric, 2),
                   round(max_exec_time::numeric, 1), rows, left(regexp_replace(query, '\\s+', ' ', 'g'), 220)
            FROM pg_stat_statements
            WHERE dbid = (SELECT oid FROM pg_database WHERE datname = 'otari')
              AND query NOT LIKE '%pg_stat_activity%' AND query NOT LIKE '%pg_stat_statements%'
            ORDER BY total_exec_time DESC LIMIT 20
            """
        ).fetchall()

    lines = [
        f"# Profile: {out.name}",
        "",
        f"Window: {elapsed:.0f}s",
        "",
        "## Postgres: top statements by total time",
        "",
        "| calls | total ms | mean ms | max ms | rows | query |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for calls, total, mean, worst, rows, query in statements:
        lines.append(f"| {calls} | {total} | {mean} | {worst} | {rows} | `{query.replace('|', '/')}` |")

    samples_path = out / "pg-samples.json"
    if samples_path.exists():
        samples = json.loads(samples_path.read_text())
        total = sum(n for _, n in samples["waits"]) or 1
        lines += [
            "",
            "## Postgres: what active backends were doing",
            "",
            f"{samples['samples']} samples every {samples['interval_sec']}s, "
            f"at most {samples['max_active_backends']} backends active at once.",
            "",
        ]
        lines += [f"- {event}: {n} ({n / total * 100:.0f}%)" for event, n in samples["waits"][:12]]
        if samples["blocked_queries"]:
            lines += ["", "Queries seen blocked by another backend's lock:", ""]
            lines += [f"- {n}x `{query}`" for query, n in samples["blocked_queries"]]
        else:
            lines += ["", "No backend was seen blocked on another's lock."]

    recordings = sorted(path.name for path in out.glob("otari-*.*") if path.suffix in (".txt", ".svg"))
    if recordings:
        lines += ["", "## CPU (py-spy)", "", "Open a `.txt` recording in https://www.speedscope.app.", ""]
        lines += [f"- [{name}]({name})" for name in recordings]

    report = "\n".join(lines) + "\n"
    (out / "report.md").write_text(report)
    print(report)


def statements(requests: int, out: Path | None, db: str = "otari", json_out: Path | None = None) -> None:
    """Statements per request since pg_stat_statements was zeroed, on database ``db``.

    The profiler's own queries and PostgreSQL's foreign-key checks are left out.
    """
    with psycopg.connect(DSN) as conn:
        rows = conn.execute(
            """
            SELECT calls, left(regexp_replace(query, '\\s+', ' ', 'g'), 110)
            FROM pg_stat_statements
            WHERE dbid = (SELECT oid FROM pg_database WHERE datname = %s)
              AND query NOT LIKE '%%pg_stat%%'
              AND query NOT LIKE 'SELECT 1 FROM ONLY%%'
              AND query NOT LIKE 'SELECT $_ FROM ONLY%%'
              AND calls >= %s
            ORDER BY calls DESC
            """,
            (db, max(requests // 2, 1)),
        ).fetchall()
    per_request = [(calls / requests, query) for calls, query in rows]
    total = sum(n for n, _ in per_request)
    lines = [f"Statements per request: {total:.1f} over {requests} requests", ""]
    lines += [f"{n:6.2f}  {query}" for n, query in per_request]
    text = "\n".join(lines) + "\n"
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(
                {
                    "per_request": round(total, 2),
                    "per_request_rounded": sum(round(n) for n, _ in per_request),
                    "statements": [[round(n, 2), q] for n, q in per_request],
                }
            )
        )
    print(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["begin", "sample", "end", "statements", "reset-statements"])
    parser.add_argument("--out")
    parser.add_argument("--requests", type=int, default=100, help="for statements")
    parser.add_argument("--db", default="otari", help="for statements: the database whose statements count")
    parser.add_argument("--json-out", help="for statements: also write the figures as JSON")
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--interval", type=float, default=0.25)
    args = parser.parse_args()
    out = Path(args.out) if args.out else None
    if args.step == "statements":
        statements(args.requests, out, args.db, Path(args.json_out) if args.json_out else None)
        return 0
    if args.step == "reset-statements":
        reset_statements()
        return 0
    if out is None:
        parser.error("--out is required for begin, sample and end")
    if args.step == "begin":
        begin(out)
    elif args.step == "sample":
        sample(out, args.seconds, args.interval)
    else:
        end(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
