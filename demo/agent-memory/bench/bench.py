"""A bigger memory to test automatic recall on, through the local gateway.

    bench.py seed                    erase the demo workspace's memory and store dataset.ENTRIES as they are
    bench.py eval [MODE ...]         run dataset.CASES in each mode and score what recall handed over
    bench.py filler N                add N entries about made-up services, to see recall in a crowded memory

Modes: similarity (0.65 floor, the default), broad (0.60), rerank (the deployment's rerank model, 0.8, the default).
Seed with the local consolidation model off (see README.md), so every entry is kept exactly as written.
"""

import json
import pathlib
import random
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime

from dataset import CASES, ENTRIES

ENV = pathlib.Path(__file__).resolve().parent.parent
API = "http://127.0.0.1:8010/api/v1"
MODEL = "ollama:qwen2.5:3b"
MODES = {
    "similarity": {"recall_min_similarity": 0.65},
    "broad": {"recall_min_similarity": 0.6},
    "rerank": {"recall_rerank": True, "recall_min_relevance": 0.8},
}


def _env(name: str) -> dict[str, str]:
    values = {}
    for line in (ENV / name).read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, _, value = line.removeprefix("export ").partition("=")
            values[key.strip()] = value.strip().strip("'\"")
    return values


KEYS = {**_env(".env"), **_env("keys.env")}
WS = KEYS["OTARI_DEMO_WORKSPACE"]
PG_SUPER = subprocess.run(
    ["docker", "exec", "otari-memory-pg", "printenv", "POSTGRES_USER"], capture_output=True, text=True, check=True
).stdout.strip()


def http(method: str, path: str, key: str, body: object = None, headers: dict | None = None) -> tuple[int, dict, object]:
    request = urllib.request.Request(
        f"{API}{path}",
        json.dumps(body).encode() if body is not None else None,
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **(headers or {})},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read()
            return response.status, dict(response.headers), json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read().decode()[:300]


def admin(method: str, path: str, body: object = None) -> object:
    status, _, answer = http(method, path, KEYS["OTARI_MASTER_KEY"], body)
    if status >= 300:
        raise SystemExit(f"{method} {path}: {status} {answer}")
    return answer


def tool(who: str, name: str, arguments: dict) -> dict:
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    _, _, reply = http("POST", "/memory/mcp", KEYS[f"{who}_KEY"], body, {"Accept": "application/json, text/event-stream"})
    result = reply["result"]
    if result.get("isError"):
        return {"error": result["content"][0]["text"]}
    return result.get("structuredContent") or {}


def psql(sql: str) -> list[str]:
    out = subprocess.run(
        ["docker", "exec", "otari-memory-pg", "psql", "-U", PG_SUPER, "-d", "otari_memory", "-Atc", sql],
        capture_output=True, text=True, check=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def switches(**changes: object) -> None:
    base = {"enabled": True, "workspace_scope_enabled": True, "write_policy": "auto", "offer_to_every_request": False,
            "hide_external": False, "retention_days": None, "auto_recall": False, "recall_min_similarity": 0.65,
            "recall_max_entries": 3, "recall_rerank": False, "recall_min_relevance": 0.8,
            "experiment_holdout_percent": 0}
    admin("PUT", f"/workspaces/{WS}/memory", {**base, **changes})


def wait_consolidated(timeout: float) -> None:
    started = time.monotonic()
    while (pending := int(psql("select count(*) from memory_proposals where state = 'pending'")[0])) > 0:
        if time.monotonic() - started > timeout:
            raise SystemExit(f"{pending} proposals still pending after {timeout:.0f}s")
        print(f"\r  {pending} proposals pending ", end="", flush=True)
        time.sleep(3)
    print()


def active() -> int:
    return int(psql(f"select count(*) from memory_entries where workspace_id = '{WS}' and status = 'active'")[0])


def seed() -> None:
    admin("PUT", "/organizations/me/memory", {"enabled": True})
    switches()
    admin("DELETE", f"/workspaces/{WS}/memory")
    refused = []
    for entry_id, (kind, scope, text) in ENTRIES.items():
        answer = tool("ALICE", "memory_propose", {"kind": kind, "scope": scope, "text": text})
        if "error" in answer:
            refused.append((entry_id, answer["error"]))
    wait_consolidated(600)
    print(f"{active()} active entries of {len(ENTRIES)} proposed")
    for entry_id, why in refused:
        print(f"  refused {entry_id}: {why}")


SERVICES = ["billing-api", "ledger", "router", "auth", "search", "notifier", "exporter", "scheduler", "ingest", "proxy"]
TEAMS = ["payments", "platform", "growth", "data", "edge", "identity", "core", "support tooling"]
CHANNELS = ["ops-payments", "ops-platform", "ops-data", "ops-edge", "ops-core"]


# Varied enough that few fillers are 0.95 alike, which the consolidator would keep only once.
SUBJECTS = [
    "the nightly backfill job", "the CSV export", "the webhook relay", "the audit log shipper", "the search indexer",
    "the email digest", "the thumbnail worker", "the session cleaner", "the quota reconciler", "the schema registry",
    "the feature flag service", "the PDF renderer", "the geo lookup cache", "the SMS fallback", "the cron fleet",
    "the S3 lifecycle rule", "the Kafka consumer group", "the image CDN", "the sandbox pool", "the model catalog sync",
    "the trial expiry job", "the invoice PDF job", "the partner API", "the mobile push gateway", "the data warehouse load",
    "the translation memory", "the abuse classifier", "the status page", "the dependency bot", "the license checker",
]
ACTIONS = [
    "times out when", "was rewritten in Go after", "pages the on-call engineer if", "is paused every quarter while",
    "drops messages silently when", "doubles its memory use whenever", "must be restarted by hand after",
    "falls back to a read replica when", "writes duplicate rows if", "skips a run when", "is rate limited to 50 calls a second because",
    "was moved to its own cluster since", "logs at debug level only while", "needs a manual approval before",
]
CAUSES = [
    "the upstream vendor rotates its certificates", "a tenant uploads more than ten thousand files", "the clock drifts on arm nodes",
    "the finance team closes the month", "a migration holds a table lock", "the Friday load test runs", "a region fails over",
    "the vendor's sandbox is down", "customers in Japan wake up", "the disk on the shared volume is nearly full",
    "a feature flag is flipped without a ticket", "the marketing newsletter goes out", "the token cache is cold",
    "a partner sends malformed JSON", "the backup window overlaps peak traffic", "someone renames a Kubernetes namespace",
]


def filler_text(rng: random.Random, n: int) -> tuple[str, str]:
    subject, action, cause = rng.choice(SUBJECTS), rng.choice(ACTIONS), rng.choice(CAUSES)
    owner = rng.choice(TEAMS)
    endings = [
        f" The {owner} team owns it.",
        f" Ask in {rng.choice(CHANNELS)} before touching it.",
        f" Ticket OPS-{n}.",
        f" Seen {rng.randint(2, 40)} times since {rng.choice(['June', 'July', 'August', 'September'])}.",
    ]
    text = f"{subject[0].upper()}{subject[1:]} {action} {cause}.{rng.choice(endings)}"
    return rng.choice(["fact", "fact", "procedure", "decision"]), text


def filler(count: int) -> None:
    rng = random.Random(len(SUBJECTS) + int(time.time()))
    start = int(psql(f"select count(*) from memory_entries where workspace_id = '{WS}'")[0])
    for index in range(count):
        kind, text = filler_text(rng, start + index)
        answer = tool("BOB", "memory_propose", {"kind": kind, "scope": "workspace", "text": text})
        if "error" in answer:
            print(f"  refused: {answer['error']}")
        if index % 100 == 99:
            print(f"  proposed {index + 1}")
    wait_consolidated(3600)
    print(f"{active()} active entries")


_texts: dict[str, str] = {}
TEXT_IDS = {text: entry_id for entry_id, (_, _, text) in ENTRIES.items()}


def dataset_id(entry_id: str) -> str:
    if entry_id not in _texts:
        got = tool("ALICE", "memory_get", {"id": entry_id})
        _texts[entry_id] = got.get("entry", {}).get("text", "") if "error" not in got else ""
    return TEXT_IDS.get(_texts[entry_id], "filler" if _texts[entry_id] else "unreadable")


def last_read() -> tuple[str, list[str]] | None:
    rows = psql("select id || '|' || entry_ids::text from memory_reads where tool = 'auto_recall' order by read_at desc limit 1")
    if not rows:
        return None
    read_id, _, ids = rows[0].partition("|")
    return read_id, json.loads(ids)


def ask(case: dict) -> list[str]:
    before = last_read()
    messages = [{"role": "user", "content": turn} for turn in [*case["earlier"], case["question"]]]
    status, headers, answer = http(
        "POST", "/chat/completions", KEYS[f"{case['who']}_KEY"], {"model": MODEL, "max_tokens": 1, "messages": messages}
    )
    if status != 200:
        raise SystemExit(f"{case['name']}: {status} {answer}")
    recalled = int({k.lower(): v for k, v in headers.items()}.get("otari-memory-recalled", "0"))
    if recalled == 0:
        return []
    after = last_read()
    if after is None or after == before:
        raise SystemExit(f"{case['name']}: recalled {recalled} but no read was recorded")
    return [dataset_id(entry_id) for entry_id in after[1]]


def latencies(since: str) -> dict[str, tuple[int, float, float]]:
    rows = psql(
        "select endpoint, count(*), percentile_cont(0.5) within group (order by latency_ms), "
        "percentile_cont(0.95) within group (order by latency_ms) from usage_logs "
        f"where endpoint in ('/v1/memory/recall', '/v1/memory/rerank') and timestamp >= '{since}' group by endpoint"
    )
    return {row.split("|")[0]: (int(row.split("|")[1]), float(row.split("|")[2]), float(row.split("|")[3])) for row in rows}


def mode_switches(mode: str) -> dict:
    """A named mode, or rerank:SCORE and similarity:FLOOR for other thresholds."""
    name, _, threshold = mode.partition(":")
    if not threshold:
        return MODES[name]
    if name == "rerank":
        return {"recall_rerank": True, "recall_min_relevance": float(threshold)}
    return {"recall_min_similarity": float(threshold)}


def evaluate(mode: str) -> dict:
    switches(auto_recall=True, **mode_switches(mode))
    since = datetime.now(UTC).isoformat()
    log = ENV / "gateway.log"
    log_start = log.stat().st_size
    started = time.monotonic()
    results = []
    for case in CASES:
        got = ask(case)
        want, fair = set(case["want"]), set(case["want"]) | set(case["ok"])
        results.append({**case, "got": got, "hits": len(want & set(got)), "wrong": [g for g in got if g not in fair]})
    elapsed = time.monotonic() - started
    timeouts = log.read_text(errors="replace")[log_start:].count("Automatic recall took longer than")
    return {"mode": mode, "results": results, "latency": latencies(since), "timeouts": timeouts, "seconds": elapsed}


def report(run: dict) -> None:
    results = run["results"]
    answerable = [r for r in results if r["want"]]
    handed = sum(len(r["got"]) for r in results)
    wrong = sum(len(r["wrong"]) for r in results)
    found = sum(r["hits"] for r in answerable)
    expected = sum(len(r["want"]) for r in answerable)
    clean = sum(1 for r in results if not r["want"] and not r["got"])
    nothing = sum(1 for r in results if not r["want"])
    leaks = sum(1 for r in results if r["who"] == "BOB" and any(g.startswith("ma") for g in r["got"]))
    print(f"\n━━ {run['mode']} ━━  ({active()} entries in memory, {run['seconds']:.0f}s)")
    print(f"  found      {found}/{expected} of the entries that answer ({found / expected:.0%})")
    print(f"  precision  {handed - wrong}/{handed} handed over were right ({(handed - wrong) / max(handed, 1):.0%})")
    print(f"  nothing    {clean}/{nothing} questions memory cannot answer got nothing added")
    print(f"  privacy    {leaks} of Alice's own entries reached Bob")
    print(f"  per turn   {handed / len(results):.1f} entries added on average, {run['timeouts']} recalls over the 3s budget")
    for endpoint, (calls, p50, p95) in sorted(run["latency"].items()):
        print(f"  {endpoint:20} {calls} calls, p50 {p50:.0f} ms, p95 {p95:.0f} ms")
    by_kind: dict[str, list[dict]] = {}
    for result in results:
        by_kind.setdefault(result["kind"], []).append(result)
    print("  by kind    " + ", ".join(
        f"{kind} {sum(1 for r in rs if set(r['want']) <= set(r['got']) and not r['wrong'])}/{len(rs)}"
        for kind, rs in by_kind.items()
    ) + "  (exactly right)")
    misses = [r for r in results if not (set(r["want"]) <= set(r["got"])) or r["wrong"]]
    for r in misses:
        print(f"    ✗ {r['name']:24} want {r['want'] or '-'}  got {r['got'] or '-'}")


if __name__ == "__main__":
    command, *rest = sys.argv[1:] or ["help"]
    if command == "seed":
        seed()
    elif command == "filler":
        filler(int(rest[0]) if rest else 1000)
    elif command == "eval":
        runs = [evaluate(mode) for mode in (rest or list(MODES))]
        for run in runs:
            report(run)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        out = pathlib.Path(__file__).resolve().parent / f"results-{stamp}.json"
        out.write_text(json.dumps(runs, indent=1, default=str))
        print(f"\nfull results: {out}")
    else:
        print(__doc__)
