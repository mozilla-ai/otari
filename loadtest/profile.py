# /// script
# requires-python = ">=3.12"
# dependencies = ["psycopg[binary]>=3.2", "httpx>=0.27", "redis>=5"]
# ///
"""Profile a run from the outside: gateway metrics, Postgres, Redis and CPU.

    uv run profile.py begin  --out results/profile-steady
    uv run profile.py sample --out results/profile-steady --seconds 240
    uv run profile.py end    --out results/profile-steady
    uv run profile.py statements --requests 100        # queries per request
    uv run profile.py statements --db ab_head --json-out results/ab/statements-head.json
    uv run profile.py analyze results/profile-steady/otari-1.txt

``begin`` snapshots each replica's /metrics and zeroes pg_stat_statements and
Redis's command stats. ``sample`` polls pg_stat_activity during the run, so lock
waits on a shared budget row show up as what each backend was waiting on.
``end`` snapshots again and writes report.md: what each replica did over the
run (requests, latency from the histogram buckets, CPU, memory, log-writer
flushes, rate-limit and budget counters), the slowest SQL by total time, the
wait events, Redis's per-command latency, and where each replica spent its
CPU, from the py-spy recordings run.sh's PROFILE=1 leaves in the same
directory. ``statements`` reads pg_stat_statements as queries per request, for
a run of a known number of requests after ``begin`` zeroed it. ``analyze``
breaks a py-spy collapsed-stack recording down by component.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import psycopg
import redis

REPLICAS = {"otari-1": "http://otari-1:8000/metrics", "otari-2": "http://otari-2:8000/metrics"}
DSN = "postgresql://otari:otari@postgres:5432/otari"
REDIS_URL = "redis://redis:6379/0"
LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


def scrape() -> dict[str, dict[str, float]]:
    snapshot: dict[str, dict[str, float]] = {}
    for name, url in REPLICAS.items():
        try:
            text = httpx.get(url, timeout=10).text
        except httpx.HTTPError as exc:
            print(f"could not scrape {name}: {exc}")
            snapshot[name] = {}
            continue
        series: dict[str, float] = {}
        for line in text.splitlines():
            match = LINE.match(line)
            if match and not line.startswith("#"):
                series[match.group(1) + (match.group(2) or "")] = float(match.group(3))
        snapshot[name] = series
    return snapshot


def begin(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics-before.json").write_text(json.dumps(scrape()))
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
        conn.execute("SELECT pg_stat_statements_reset()")
        conn.execute("SELECT pg_stat_reset()")
    redis.Redis.from_url(REDIS_URL).config_resetstat()
    (out / "began_at").write_text(str(time.time()))
    print(f"profiling into {out}")


def sample(out: Path, seconds: float, interval: float) -> None:
    """Poll what every gateway backend is doing; write the counts as they grow."""
    waits: Counter[str] = Counter()
    blocked_queries: Counter[str] = Counter()
    max_active = 0
    samples = 0
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
            for state, wait_type, wait_event, query, blocked in active:
                waits[f"{wait_type or 'CPU'}:{wait_event or 'running'}"] += 1
                if blocked:
                    blocked_queries[query] += 1
            if samples % 20 == 0:
                _write_samples(out, samples, interval, max_active, waits, blocked_queries)
            time.sleep(interval)
    _write_samples(out, samples, interval, max_active, waits, blocked_queries)


def _write_samples(out, samples, interval, max_active, waits, blocked) -> None:
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


def histogram_quantiles(before: dict[str, float], after: dict[str, float], family: str, match: str) -> dict:
    """Approximate quantiles of one histogram's growth, summed over matching label sets."""
    buckets: dict[float, float] = defaultdict(float)
    total_sum = total_count = 0.0
    for key, value in after.items():
        if not key.startswith(family) or match not in key:
            continue
        delta = value - before.get(key, 0.0)
        if key.startswith(family + "_bucket"):
            le = re.search(r'le="([^"]+)"', key).group(1)
            buckets[math.inf if le == "+Inf" else float(le)] += delta
        elif key.startswith(family + "_sum"):
            total_sum += delta
        elif key.startswith(family + "_count"):
            total_count += delta
    if not total_count:
        return {}
    ordered = sorted(buckets.items())

    def quantile(q: float) -> str:
        target = q * total_count
        for le, cumulative in ordered:
            if cumulative >= target:
                return "> largest bucket" if le == math.inf else f"<= {le * 1000:g} ms"
        return "?"

    return {
        "count": int(total_count),
        "mean_ms": round(total_sum / total_count * 1000, 1),
        "p50": quantile(0.5),
        "p90": quantile(0.9),
        "p99": quantile(0.99),
    }


def end(out: Path) -> None:
    after = scrape()
    before = json.loads((out / "metrics-before.json").read_text())
    (out / "metrics-after.json").write_text(json.dumps(after))
    began = float((out / "began_at").read_text())
    elapsed = time.time() - began
    lines = [f"# Profile: {out.name}", "", f"Window: {elapsed:.0f}s", ""]

    lines += ["## Gateway replicas", ""]
    for replica in REPLICAS:
        b, a = before.get(replica, {}), after.get(replica, {})
        if not a:
            lines += [f"### {replica}", "", "not scraped (down?)", ""]
            continue
        cpu = a.get("process_cpu_seconds_total", 0) - b.get("process_cpu_seconds_total", 0)
        rss = a.get("process_resident_memory_bytes", 0) / 2**20
        lines += [
            f"### {replica}",
            "",
            f"- CPU: {cpu:.1f}s over {elapsed:.0f}s ({cpu / elapsed * 100:.0f}% of one core)",
            f"- Resident memory at the end: {rss:.0f} MiB",
        ]
        for label, family, match in [
            ("Chat completions, whole request", "gateway_request_duration_seconds", "chat/completions"),
            ("Usage log flush", "gateway_usage_log_flush_duration_seconds", ""),
        ]:
            quantiles = histogram_quantiles(b, a, family, match)
            if quantiles:
                lines.append(f"- {label}: {json.dumps(quantiles)}")
        counters = []
        for key, value in sorted(a.items()):
            delta = value - b.get(key, 0.0)
            if delta and key.startswith("gateway_") and not re.search(r"_(bucket|sum|count|created)(\{|$)", key):
                counters.append(f"  - `{key}`: +{delta:g}")
        if counters:
            lines += ["- Counters that moved:", *counters[:60]]
        lines.append("")

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
        locks = conn.execute(
            "SELECT relname, n_tup_upd, n_tup_hot_upd, n_dead_tup FROM pg_stat_user_tables "
            "WHERE n_tup_upd > 0 ORDER BY n_tup_upd DESC LIMIT 10"
        ).fetchall()
        deadlocks, conflicts = conn.execute(
            "SELECT deadlocks, conflicts FROM pg_stat_database WHERE datname = 'otari'"
        ).fetchone()
    lines += [
        "## Postgres: top statements by total time",
        "",
        "| calls | total ms | mean ms | max ms | rows | query |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for calls, total, mean, worst, rows, query in statements:
        lines.append(f"| {calls} | {total} | {mean} | {worst} | {rows} | `{query.replace('|', '/')}` |")
    lines += ["", f"Deadlocks: {deadlocks}, conflicts: {conflicts}", "", "Most-updated tables:", ""]
    lines += [f"- {name}: {upd} updates ({hot} HOT), {dead} dead tuples" for name, upd, hot, dead in locks]

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

    stats = redis.Redis.from_url(REDIS_URL).info("commandstats")
    lines += ["", "## Redis: per-command latency", "", "| command | calls | usec/call |", "|---|---:|---:|"]
    for command, figures in sorted(stats.items(), key=lambda item: -item[1]["calls"])[:12]:
        lines.append(f"| {command.removeprefix('cmdstat_')} | {figures['calls']} | {figures['usec_per_call']} |")

    for recording in sorted(out.glob("otari-*.txt")):
        lines += ["", f"## CPU: {recording.stem} (py-spy, open the file in speedscope.app to explore)", ""]
        lines += analyze(recording)

    flames = sorted(path.name for path in out.glob("*.svg"))
    if flames:
        lines += ["", "## CPU flame graphs (py-spy)", ""]
        lines += [f"- [{name}]({name})" for name in flames]

    report = "\n".join(lines) + "\n"
    (out / "report.md").write_text(report)
    print(report)


# Frames that stand for a component, matched on the start of py-spy's
# "function (file:line)" label. Inclusive time, so the groups overlap.
COMPONENTS = {
    "request handling (FastAPI/Starlette routing)": "handle (starlette/routing.py",
    "SQLAlchemy statements (AsyncSession.execute)": "execute (sqlalchemy/orm/session.py",
    "SQLAlchemy cache keys": "_generate_cache_key (sqlalchemy/sql/cache_key.py",
    "SQLAlchemy compiled-statement cache": "_compile_w_cache (sqlalchemy/sql/elements.py",
    "SQLAlchemy commits": "commit (sqlalchemy/ext/asyncio/session.py",
    "asyncpg protocol": "__bind_execute (asyncpg/prepared_stmt.py",
    "provider SDK requests (openai)": "request (openai/_base_client.py",
    "provider client construction (any-llm)": "create (any_llm/any_llm.py",
    "SSL context creation": "create_default_context (ssl.py",
    "pydantic validation": "model_validate (pydantic/main.py",
}


def analyze(recording: Path, top: int = 15) -> list[str]:
    """Where one replica spent its CPU, from a py-spy ``--format raw`` (collapsed stacks) recording.

    Samples whose innermost frame is the event loop's idle wait are idle; the
    rest are busy, and each share below is of busy samples. Without py-spy's
    ``--idle`` the loop is caught idle only when it is between callbacks.
    """
    inclusive: Counter[str] = Counter()
    own: Counter[str] = Counter()
    total = busy = 0
    for line in recording.read_text().splitlines():
        stack, _, count = line.rpartition(" ")
        if not stack or not count.isdigit():
            continue
        n = int(count)
        total += n
        frames = stack.split(";")
        if frames[-1].startswith("run (asyncio/runners.py"):
            continue
        busy += n
        own[frames[-1]] += n
        for frame in set(frames):
            inclusive[frame] += n
    if not busy:
        return ["No busy samples recorded."]
    lines = [
        f"{busy} busy samples of {total} ({busy / total * 100:.0f}%).",
        "",
        "| component | share of busy |",
        "|---|---:|",
    ]
    for label, prefix in COMPONENTS.items():
        share = max((n for frame, n in inclusive.items() if frame.startswith(prefix)), default=0)
        lines.append(f"| {label} | {share / busy * 100:.1f}% |")
    lines += ["", f"Top {top} functions by their own time:", ""]
    lines += [f"- {n / busy * 100:.1f}% `{frame}`" for frame, n in own.most_common(top)]
    return lines


def statements(requests: int, out: Path | None, db: str = "otari", json_out: Path | None = None) -> None:
    """Queries per request since pg_stat_statements was zeroed, on database ``db``.

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
                    # Background work (the sweeper, log flushes) adds fractions,
                    # mostly to BEGIN and COMMIT; a request's own statements are whole.
                    "per_request_rounded": sum(round(n) for n, _ in per_request),
                    "statements": [[round(n, 2), q] for n, q in per_request],
                }
            )
        )
    print(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["begin", "sample", "end", "statements", "analyze", "reset-statements"])
    parser.add_argument("recordings", nargs="*", help="py-spy raw recordings, for analyze")
    parser.add_argument("--out")
    parser.add_argument("--requests", type=int, default=100, help="for statements")
    parser.add_argument("--db", default="otari", help="for statements: the database whose statements count")
    parser.add_argument("--json-out", help="for statements: also write the figures as JSON")
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--interval", type=float, default=0.25)
    args = parser.parse_args()
    out = Path(args.out) if args.out else None
    if args.step == "analyze":
        for recording in args.recordings:
            print(f"## {recording}\n")
            print("\n".join(analyze(Path(recording))) + "\n")
        return 0
    if args.step == "statements":
        statements(args.requests, out, args.db, Path(args.json_out) if args.json_out else None)
        return 0
    if args.step == "reset-statements":
        with psycopg.connect(DSN, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
            conn.execute("SELECT pg_stat_statements_reset()")
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
