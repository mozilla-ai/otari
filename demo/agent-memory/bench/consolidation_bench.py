"""How well a model decides what a proposal does to memory, on consolidation_cases.CASES.

    uv run python consolidation_bench.py MODEL [MODEL ...]      (from the repo root, so the gateway imports)

Each model is asked exactly as the consolidator asks it (its prompt, token budget and response format, imported
from the gateway), through the local gateway's /v1/chat/completions. Then the consolidator's own similarity
guard is applied, with the deployment's embedding model, to show what memory would actually do.

A destructive mistake is replacing or dropping what should have been kept apart: an update or no_change where the
proposal said something new, or an update of the wrong note. It loses information; the rest only adds a little.
"""

import json
import math
import pathlib
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bench import KEYS  # noqa: E402
from consolidation_cases import CASES  # noqa: E402

from gateway.services.memory._models import decision_request, parse_decision  # noqa: E402
from gateway.services.memory._service import guarded_decision  # noqa: E402

API = "http://127.0.0.1:8010/api/v1"


def post(path: str, body: dict) -> dict:
    request = urllib.request.Request(
        f"{API}{path}",
        json.dumps(body).encode(),
        {"Authorization": f"Bearer {KEYS['ALICE_KEY']}", "Content-Type": "application/json", "Otari-Memory": "off"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def embed(texts: list[str]) -> list[list[float]]:
    body = json.dumps({"model": "nomic-embed-text", "input": texts}).encode()
    request = urllib.request.Request("http://127.0.0.1:11434/api/embed", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)["embeddings"]


def cos(a: list[float], b: list[float]) -> float:
    return math.sumprod(a, b) / math.sqrt(math.sumprod(a, a) * math.sumprod(b, b))


def destructive(case: dict, action: str, target: int | None) -> bool:
    want_action, want_target = case["want"]
    if action in ("update", "no_change"):
        return want_action in ("add", "conflict") or target != want_target
    return False


def score(case: dict, action: str, target: int | None) -> bool:
    fair = [case["want"], *case["ok"]]
    return any(action == a and (t is None or target == t) for a, t in fair)


def run(model: str) -> None:
    right = guarded_right = destroyed = guarded_destroyed = empty = 0
    latencies, misses = [], []
    for case in CASES:
        messages, options = decision_request(case["proposal"], case["existing"])
        started = time.monotonic()
        try:
            answer = post("/chat/completions", {"model": model, "messages": messages, **options})
            content = answer["choices"][0]["message"].get("content") or ""
        except Exception as exc:  # noqa: BLE001 - a failed call is scored as no answer
            content = f"<{type(exc).__name__}>"
        latencies.append(time.monotonic() - started)
        decision = parse_decision(content, len(case["existing"]))
        if decision is None:  # no decision: the consolidator asks again on a later pass
            empty += 1
            misses.append(f"    ∅ {case['name']:26} no decision: {content[:60]!r}")
            continue
        action, target = decision.action, decision.target + 1 if decision.target is not None else None
        vectors = embed([case["proposal"], *case["existing"]])
        alike = [cos(vectors[0], v) for v in vectors[1:]]
        guarded = guarded_decision(decision, alike)
        g_action, g_target = guarded.action, guarded.target + 1 if guarded.target is not None else None
        right += score(case, action, target)
        guarded_right += score(case, g_action, g_target)
        destroyed += destructive(case, action, target)
        guarded_destroyed += destructive(case, g_action, g_target)
        if not score(case, g_action, g_target):
            misses.append(f"    ✗ {case['name']:26} want {case['want']}  model {(action, target)}  kept {(g_action, g_target)}")
    n = len(CASES)
    print(f"\n━━ {model} ━━")
    print(f"  right          {right}/{n} ({right / n:.0%}) as the model answered, {guarded_right}/{n} ({guarded_right / n:.0%}) after the guard")
    print(f"  destructive    {destroyed} as answered, {guarded_destroyed} after the guard")
    print(f"  empty answers  {empty}")
    print(f"  latency        p50 {statistics.median(latencies):.1f}s, max {max(latencies):.1f}s")
    for miss in misses:
        print(miss)


def similarities() -> None:
    """How alike each proposal is to the note it should replace or repeat, and to its nearest note otherwise."""
    by_kind: dict[str, list[float]] = {}
    for case in CASES:
        vectors = embed([case["proposal"], *case["existing"]])
        alike = [cos(vectors[0], v) for v in vectors[1:]]
        action, target = case["want"]
        value = alike[target - 1] if action in ("update", "no_change") else max(alike)
        by_kind.setdefault(action, []).append(value)
    print("similarity to the note it acts on (update/no_change) or to the nearest (add/conflict):")
    for action, values in by_kind.items():
        print(f"  {action:9} min {min(values):.3f}  median {statistics.median(values):.3f}  max {max(values):.3f}")


if __name__ == "__main__":
    similarities()
    for model in sys.argv[1:]:
        run(model)
