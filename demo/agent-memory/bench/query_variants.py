"""What to send the reranker as its query, tried offline on the same candidates recall would pick.

Candidates are the ten entries most alike to the turn, by recall's own rule (the better of the turn alone and
the turn after the earlier ones), with the same embedding model; then each variant's query is reranked through
the gateway's /v1/rerank and scored as bench.py scores a run.
"""

import json
import math
import sys
import urllib.request

from bench import KEYS, http
from dataset import CASES, ENTRIES

RERANK = "fireworks:accounts/fireworks/models/qwen3-reranker-8b"
THRESHOLDS = [0.5, 0.7, 0.8, 0.9]
members = {entry_id for entry_id, (_, scope, _) in ENTRIES.items() if scope == "member"}


def embed(texts: list[str]) -> list[list[float]]:
    body = json.dumps({"model": "nomic-embed-text", "input": texts}).encode()
    request = urllib.request.Request("http://127.0.0.1:11434/api/embed", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)["embeddings"]


def cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def rerank(query: str, ids: list[str]) -> dict[str, float]:
    status, _, answer = http("POST", "/rerank", KEYS["ALICE_KEY"],
                             {"model": RERANK, "query": query, "documents": [ENTRIES[i][2] for i in ids]})
    if status != 200:
        raise SystemExit(f"rerank: {status} {answer}")
    return {ids[result["index"]]: result["relevance_score"] for result in answer["results"]}


def variants(case: dict) -> dict[str, str]:
    question, earlier = case["question"], " ".join(case["earlier"])
    if not earlier:
        return {name: question for name in ("alone", "labelled", "question first", "instructed")}
    return {
        "alone": question,
        "labelled": f"Earlier in the conversation: {earlier}\nQuestion: {question}",
        "question first": f"{question}\n(Said earlier in the conversation: {earlier})",
        "instructed": (
            "Find the notes that help answer the latest question. The earlier conversation only explains what the "
            f"question refers to; a note about the earlier conversation alone does not help.\n"
            f"Earlier conversation: {earlier}\nLatest question: {question}"
        ),
    }


ids = list(ENTRIES)
vectors = dict(zip(ids, embed([ENTRIES[i][2] for i in ids])))
scored: dict[str, list[tuple[dict, dict[str, float]]]] = {}
for case in CASES:
    visible = [i for i in ids if i not in members or case["who"] == "ALICE"]
    turn = [case["question"]] + ([" ".join(case["earlier"]) + "\n\n" + case["question"]] if case["earlier"] else [])
    queries = embed(turn)
    candidates = sorted(visible, key=lambda i: -max(cos(q, vectors[i]) for q in queries))[:10]
    for name, query in variants(case).items():
        scored.setdefault(name, []).append((case, rerank(query, candidates)))

print(f"{'variant':16} {'threshold':>9} {'found':>7} {'precision':>9} {'nothing':>7} {'follow-ups':>10} {'topic chg':>9}")
for name, runs in scored.items():
    for threshold in THRESHOLDS:
        found = expected = handed = right = clean = nothing = 0
        exact: dict[str, list[bool]] = {}
        for case, scores in runs:
            got = [i for i, s in sorted(scores.items(), key=lambda kv: -kv[1]) if s >= threshold][:3]
            fair = set(case["want"]) | set(case["ok"])
            found += len(set(case["want"]) & set(got))
            expected += len(case["want"])
            handed += len(got)
            right += sum(1 for g in got if g in fair)
            if not case["want"]:
                nothing += 1
                clean += not got
            exact.setdefault(case["kind"], []).append(set(case["want"]) <= set(got) and all(g in fair for g in got))
        fu, tc = exact["follow-up"] + exact["meeting"], exact["topic change"]
        print(f"{name:16} {threshold:9.1f} {found:3}/{expected:<3} {right / max(handed, 1):9.0%} {clean:3}/{nothing:<3}"
              f" {sum(fu):6}/{len(fu):<3} {sum(tc):5}/{len(tc):<3}")
