"""Compare two builds of Otari on one machine: measure each, then analyze.

``./run.sh BASE HEAD`` drives it, swapping the image under test between calls:

    uv run ab.py prepare    --dir D     # records the plan, prints the turn order
    uv run ab.py turn       --dir D --variant base|head --turn N
    uv run ab.py statements --dir D --variant base|head

``comparison.py`` then calls ``analyze``, which writes ``perf.json`` and
``report.md`` and returns what the checks read. The method is in README.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import psycopg
import redis

GATEWAY = "http://otari:8000"
MASTER_KEY = "loadtest-master-key"
VARIANTS = ("base", "head")
ROUNDS = 4  # turns per build, alternating base head head base ...
# Per route per turn. A turn's p50 from 100 requests is already several times
# steadier than the spread between turns, so the turns carry the signal.
SECONDS = 6
RPM = 1000
# The fake provider's fixed timing (docker-compose.yml), subtracted from the client's.
LATENCY_MS = int(os.environ["FAKE_LATENCY_MS"])
TTFT_MS = int(os.environ["FAKE_TTFT_MS"])
DIRECT = "togethersim:llama-3.3-70b"  # Model 2, no policy
MODEL1 = "vertexsim:gemini-2.5-flash"
# Smallest change reported: identical builds' turns have differed by up to 6% at p50.
MIN_EFFECT_MS, MIN_EFFECT_PCT = 1.0, 10.0
MIN_STATEMENT_INCREASE = 0.5
TRANSACTION_CONTROL = {"BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE"}


@dataclass(frozen=True)
class Scenario:
    """A route or hot path: what each request asks for, and on which tenant."""

    model: str = DIRECT
    stream: bool = False
    pooled: bool = False  # the key has a shared budget every request debits
    fill_model1: bool = False  # Model 1's minute used up first, so the policy spills


SCENARIOS = {
    "direct": Scenario(),
    "pooled": Scenario(pooled=True),
    "spill": Scenario(model="summarize", fill_model1=True),
    "stream": Scenario(stream=True),
}


# The stack


def dsn(db: str) -> str:
    return f"postgresql://otari:otari@postgres:5432/{db}"


def cpu_seconds() -> float:
    text = httpx.get(f"{GATEWAY}/metrics", timeout=10).text
    return float(re.search(r"^process_cpu_seconds_total\s+(\S+)", text, re.MULTILINE).group(1))


@dataclass
class Tenant:
    """A service user and its service key, with a per-end-user budget, and optionally a
    shared pool: a scoped budget on the key, one row every end user debits."""

    service_user: str
    key: str
    key_id: str


def create_tenant(name: str, pooled: bool) -> Tenant:
    with httpx.Client(
        base_url=f"{GATEWAY}/api/v1", headers={"Authorization": f"Bearer {MASTER_KEY}"}, timeout=30
    ) as client:

        def post(path: str, payload: dict) -> dict:
            response = client.post(path, json=payload)
            if response.is_error:
                raise RuntimeError(f"POST {path} -> {response.status_code}: {response.text[:500]}")
            return response.json()

        service_user = f"loadtest-{name}-{random.getrandbits(32):08x}"
        end_user_budget = post("/budgets", {"name": f"{name} end user", "max_budget": 100, "budget_duration_sec": 3600})
        post("/users", {"user_id": service_user})
        key = post(
            "/keys",
            {
                "key_name": name,
                "user_id": service_user,
                "is_service_key": True,
                "end_user_budget_id": end_user_budget["budget_id"],
            },
        )
        if pooled:
            pool = post("/budgets", {"name": f"{name} pool", "max_budget": 10_000, "budget_duration_sec": 86_400})
            post(
                "/scoped-budgets",
                {"scope_type": "api_token", "scope_id": key["id"], "budget_id": pool["budget_id"], "name": name},
            )
    return Tenant(service_user, key["key"], key["id"])


def wait_for_drain(tenant: Tenant, db: str, timeout: float = 30) -> None:
    """Wait until none of the tenant's budget reservations is still active.

    Raises on timeout: statements counted with requests still in flight would
    undercount this build and read as a change between the two.
    """
    deadline = time.time() + timeout
    with psycopg.connect(dsn(db), autocommit=True) as conn:
        while time.time() < deadline:
            active = conn.execute(
                "SELECT COUNT(*) FROM budget_reservations r JOIN users u ON u.user_id = r.user_id "
                "WHERE u.parent_user_id = %s AND r.status = 'active'",
                (tenant.service_user,),
            ).fetchone()[0]
            if not active:
                return
            time.sleep(1)
    raise TimeoutError(f"{active} budget reservations still active after {timeout:.0f}s")


# Load


@dataclass
class Result:
    status: int  # 0 when the connection failed
    latency_ms: float
    ttft_ms: float | None = None
    served_by: str | None = None  # the fake provider slot that answered
    complete: bool = True  # a stream ended in [DONE]


async def _one(client: httpx.AsyncClient, scenario: Scenario, prompt: str, users: int) -> Result:
    body = {
        "model": scenario.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "stream": scenario.stream,
        "user": f"enduser-{random.randrange(users):05d}",
    }
    started = time.perf_counter()

    def elapsed() -> float:
        return (time.perf_counter() - started) * 1000

    try:
        if not scenario.stream:
            response = await client.post("/api/v1/chat/completions", json=body)
            if response.status_code != 200:
                return Result(response.status_code, elapsed())
            content = response.json()["choices"][0]["message"]["content"] or ""
            return Result(200, elapsed(), served_by=content.split("served-by:")[-1][:2])
        async with client.stream("POST", "/api/v1/chat/completions", json=body) as response:
            result = Result(response.status_code, 0.0, complete=False)
            async for line in response.aiter_lines():
                if line == "data: [DONE]":
                    result.complete = True
                elif line.startswith("data: ") and "served-by:" in line and result.ttft_ms is None:
                    result.ttft_ms = elapsed()
                    result.served_by = line.split("served-by:")[-1][:2]
            result.latency_ms = elapsed()
            return result
    # Anything else (a body that is not the JSON it should be) is recorded too:
    # raised, it would surface in gather() and lose the whole run's results.
    except Exception:  # noqa: BLE001
        return Result(0, elapsed(), complete=False)


async def _run(key: str, scenario: Scenario, rpm: int, seconds: float, users: int) -> list[Result]:
    """Open loop: requests go out on schedule whatever the gateway's latency, so a
    slow gateway builds a queue rather than slowing the test down."""
    prompt = " ".join(["the quick brown fox jumps over the lazy dog"] * 45)  # about 500 tokens
    limits = httpx.Limits(max_connections=1000, max_keepalive_connections=1000)
    headers = {"Authorization": f"Bearer {key}"}
    async with httpx.AsyncClient(base_url=GATEWAY, headers=headers, timeout=120, limits=limits) as client:
        begin, tasks = time.perf_counter(), []
        for index in range(int(rpm * seconds / 60)):
            await asyncio.sleep(max(0.0, begin + index * 60 / rpm - time.perf_counter()))
            tasks.append(asyncio.create_task(_one(client, scenario, prompt, users)))
        return await asyncio.gather(*tasks)


def send(key: str, scenario: Scenario, rpm: int = RPM, seconds: float = SECONDS, users: int = 200) -> list[Result]:
    return asyncio.run(_run(key, scenario, rpm, seconds, users))


# Measuring


def save(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def prepare(directory: Path) -> None:
    """Record the plan, and print the order the builds take turns in."""
    save(
        directory / "meta.json",
        {
            variant: {"ref": os.environ.get(f"{variant.upper()}_REF"), "id": os.environ.get(f"{variant.upper()}_ID")}
            for variant in VARIANTS
        }
        | {"config": {"rounds": ROUNDS, "scenarios": list(SCENARIOS), "seconds": SECONDS, "rpm": RPM}},
    )
    # base head head base ...: drift in the machine's speed over the run lands on both.
    print(" ".join(VARIANTS[:: 1 if r % 2 == 0 else -1][i] for r in range(ROUNDS) for i in range(2)))


def _tenants(directory: Path, variant: str) -> dict[bool, Tenant]:
    """The build's two tenants, without and with a shared pool, made on its first turn."""
    path = directory / f"tenants-{variant}.json"
    if not path.exists():
        save(path, {str(pooled): asdict(create_tenant(f"ab-{variant}", pooled)) for pooled in (False, True)})
    return {pooled == "True": Tenant(**fields) for pooled, fields in json.loads(path.read_text()).items()}


def turn(directory: Path, variant: str, number: int) -> None:
    """One build's turn on a freshly started replica: every scenario once."""
    tenants = _tenants(directory, variant)
    store = redis.Redis(host="redis")
    send(tenants[False].key, SCENARIOS["direct"], seconds=5)  # discarded: a fresh process's warm-up
    for name, scenario in SCENARIOS.items():
        store.flushdb()  # rate-limit windows start empty, so every turn sees Model 1's cap the same
        tenant = tenants[scenario.pooled]
        if scenario.fill_model1:
            send(tenant.key, Scenario(model=MODEL1), rpm=6000, seconds=1)  # its 100 requests, in a burst
        before = cpu_seconds()
        results = send(tenant.key, scenario)
        cpu = cpu_seconds() - before
        served = [r for r in results if r.status == 200 and r.complete]
        # The gateway's overhead: the client's latency minus the fake's (its time to
        # first token when streamed), on Model 2 only, since a spill run's requests
        # that did not spill measure something else.
        if scenario.stream:
            samples = [r.ttft_ms - TTFT_MS for r in served if r.served_by == "m2" and r.ttft_ms]
        else:
            samples = [r.latency_ms - LATENCY_MS for r in served if r.served_by == "m2"]
        label = f"{number:02d}-{variant}-{name}"
        save(
            directory / "runs" / f"{label}.json",
            {
                "label": label,
                "turn": number,
                "variant": variant,
                "scenario": name,
                "samples": [round(s, 2) for s in samples],
                "errors": len(results) - len(served),
                "cpu_ms_per_request": cpu * 1000 / len(served) if served else None,
            },
        )
        print(f"{label}: {len(samples)} samples, p50 {statistics.median(samples) if samples else 'n/a'}")


def statements(directory: Path, variant: str) -> None:
    """Statements per request over 100 identical requests on end users that already
    exist, straight to Model 2 and spilled past a full Model 1."""
    tenant, db = _tenants(directory, variant)[False], f"ab_{variant}"
    store = redis.Redis(host="redis")
    for mode, scenario in (("direct", SCENARIOS["direct"]), ("spill", SCENARIOS["spill"])):
        store.flushdb()
        send(tenant.key, scenario, rpm=1200, seconds=6, users=50)  # creates the end users, fills Model 1
        wait_for_drain(tenant, db)
        with psycopg.connect(dsn("otari"), autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
            conn.execute("SELECT pg_stat_statements_reset()")
        send(tenant.key, scenario, rpm=1200, seconds=5, users=50)  # 100 requests
        wait_for_drain(tenant, db)
        with psycopg.connect(dsn("otari")) as conn:
            # Left out: PostgreSQL's own foreign-key checks, the drain wait's query,
            # and anything run less than once every other request.
            rows = conn.execute(
                """
                SELECT calls, left(regexp_replace(query, '\\s+', ' ', 'g'), 110)
                FROM pg_stat_statements
                WHERE dbid = (SELECT oid FROM pg_database WHERE datname = %s)
                  AND query NOT LIKE '%%pg_stat%%' AND query NOT LIKE '%%budget_reservations r JOIN users%%'
                  AND query NOT LIKE 'SELECT 1 FROM ONLY%%' AND query NOT LIKE 'SELECT $_ FROM ONLY%%'
                  AND calls >= 50
                ORDER BY calls DESC
                """,
                (db,),
            ).fetchall()
        counted = [(round(calls / 100, 2), query) for calls, query in rows]
        save(directory / f"statements-{variant}-{mode}.json", counted)
        print(f"{variant} {mode}: {sum(n for n, _ in counted):.1f} statements per request")


# Analysis


def _r(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)


def _latency(runs: list[dict], scenario: str) -> dict | None:
    """Judge a route by each build's per-turn p50s.

    A regression only when every head turn is slower than every base turn by more
    than the smallest change reported, an improvement in the mirror case, and
    unchanged otherwise. Something else loading the machine during a turn can only
    blur the two apart, never make them separate, unless it hits every turn of one
    build and none of the other's, which the alternating order makes unlikely.
    """
    group = sorted((r for r in runs if r["scenario"] == scenario and r["samples"]), key=lambda r: r["turn"])
    p50s = {v: [statistics.median(r["samples"]) for r in group if r["variant"] == v] for v in VARIANTS}
    if not all(p50s.values()):
        return None
    base, head = statistics.median(p50s["base"]), statistics.median(p50s["head"])
    threshold = max(MIN_EFFECT_MS, min(p50s["base"]) * MIN_EFFECT_PCT / 100)
    if min(p50s["head"]) - max(p50s["base"]) > threshold:
        verdict = "regression"
    elif min(p50s["base"]) - max(p50s["head"]) > threshold:
        verdict = "improvement"
    else:
        verdict = "unchanged"
    cpu = {}
    for v in VARIANTS:
        measured = [r["cpu_ms_per_request"] for r in group if r["variant"] == v and r["cpu_ms_per_request"]]
        cpu[v] = statistics.median(measured) if measured else None
    return {
        "p50": {
            "base": _r(base),
            "head": _r(head),
            "delta": _r(head - base),
            "delta_pct": _r((head - base) / base * 100, 1),
        },
        "turn_p50s": {v: [_r(p) for p in p50s[v]] for v in VARIANTS},
        "threshold": _r(threshold),
        "verdict": verdict,
        "cpu_ms_per_request": {v: _r(cpu[v], 3) for v in VARIANTS},
    }


def _statements(directory: Path, mode: str) -> dict | None:
    paths = {v: directory / f"statements-{v}-{mode}.json" for v in VARIANTS}
    if not all(path.exists() for path in paths.values()):
        return None
    per_query = {v: {query: n for n, query in json.loads(paths[v].read_text())} for v in VARIANTS}
    changed = [
        {"query": query, "base": round(per_query["base"].get(query, 0)), "head": round(per_query["head"].get(query, 0))}
        for query in sorted(set(per_query["base"]) | set(per_query["head"]))
        if round(per_query["base"].get(query, 0)) != round(per_query["head"].get(query, 0))
    ]
    # Judged on data statements only: background transactions (the reservation
    # sweeper, usage-log flushes) land in the count as fractions of a BEGIN, COMMIT
    # or ROLLBACK, and on identical builds have moved BEGIN across a rounding line.
    data = {
        v: [n for q, n in per_query[v].items() if q.split(" ", 1)[0].upper() not in TRANSACTION_CONTROL]
        for v in VARIANTS
    }
    whole = sum(round(n) for n in data["head"]) - sum(round(n) for n in data["base"])
    raw = sum(data["head"]) - sum(data["base"])
    if whole >= 1 and raw >= MIN_STATEMENT_INCREASE:
        verdict = "regression"
    elif whole <= -1 and raw <= -MIN_STATEMENT_INCREASE:
        verdict = "improvement"
    else:
        verdict = "unchanged"
    return {
        "base": _r(sum(data["base"])),
        "head": _r(sum(data["head"])),
        "delta": _r(raw),
        "changed": changed,
        "verdict": verdict,
    }


def analyze(directory: Path) -> dict:
    """Read a run's results; write perf.json and report.md; return the figures."""
    meta = json.loads((directory / "meta.json").read_text())
    runs = [json.loads(path.read_text()) for path in sorted((directory / "runs").glob("*.json"))]
    plan = meta["config"]
    missing = []
    for scenario in plan["scenarios"]:
        usable = {
            v: sum(1 for r in runs if (r["scenario"], r["variant"]) == (scenario, v) and r["samples"]) for v in VARIANTS
        }
        if any(n != plan["rounds"] for n in usable.values()):
            missing.append(f"{scenario}: base ran {usable['base']} of {plan['rounds']}, head {usable['head']}")
    perf = {
        **meta,
        "scenarios": {s: _latency(runs, s) for s in plan["scenarios"]},
        "statements": {mode: _statements(directory, mode) for mode in ("direct", "spill")},
        # From every head run, not the compared ones: a run that served nothing has
        # no samples and drops out of the comparison, but its failures still count.
        "head_errors": sum(r["errors"] for r in runs if r["variant"] == "head"),
        "missing_runs": missing,
    }
    (directory / "perf.json").write_text(json.dumps(perf, indent=2))
    (directory / "report.md").write_text(render(perf))
    return perf


MARK = {"regression": "🔴 regression", "improvement": "🟢 improvement", "unchanged": "unchanged"}


def render(perf: dict) -> str:
    lines = [
        "## Load test, base vs head",
        "",
        f"base `{perf['base']['ref']}`, head `{perf['head']['ref']}`",
        "",
        "Gateway overhead in ms at 1,000 requests a minute: the client's latency minus the fake provider's "
        "(time to first token when streamed), as the median of each build's per-turn p50s. A route regressed "
        "or improved only when every turn of one build is beyond every turn of the other by more than "
        f"max({MIN_EFFECT_MS} ms, {MIN_EFFECT_PCT}% of base).",
        "",
        "| route | p50 base | p50 head | change | base turns | head turns | verdict |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for name, cell in perf["scenarios"].items():
        if cell is None:
            lines.append(f"| {name} | | | | | | no runs to compare |")
            continue
        p50, turns = cell["p50"], cell["turn_p50s"]
        lines.append(
            f"| {name} | {p50['base']:.1f} | {p50['head']:.1f} | {p50['delta']:+.1f} ({p50['delta_pct']:+.0f}%) "
            f"| {', '.join(f'{t:.1f}' for t in turns['base'])} | {', '.join(f'{t:.1f}' for t in turns['head'])} "
            f"| {MARK[cell['verdict']]} |"
        )
    lines += ["", "| data statements per request | base | head | change | verdict |", "|---|---:|---:|---:|---|"]
    for mode, s in perf["statements"].items():
        if s is None:
            lines.append(f"| {mode} | | | | not counted |")
            continue
        lines.append(f"| {mode} | {s['base']:.1f} | {s['head']:.1f} | {s['delta']:+.1f} | {MARK[s['verdict']]} |")
    changed = [(mode, c) for mode, s in perf["statements"].items() if s for c in s["changed"]]
    if changed:
        lines += ["", "<details><summary>Statements that changed, per request</summary>", ""]
        lines += [f"- {mode}: {c['base']} → {c['head']} `{c['query']}`" for mode, c in changed]
        lines += ["", "</details>"]
    lines += ["", "Machine-readable: `perf.json` beside this report.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["prepare", "turn", "statements"])
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--turn", type=int)
    args = parser.parse_args()
    if args.step == "prepare":
        prepare(args.dir)
    elif args.step == "turn":
        turn(args.dir, args.variant, args.turn)
    else:
        statements(args.dir, args.variant)


if __name__ == "__main__":
    main()
