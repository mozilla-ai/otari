#!/usr/bin/env python3
"""Seed a running gateway with routing policies and traffic, to click through.

Creates a set of policies covering the shapes the feature supports, teaches the
learned one a handful of scored examples, then drives enough traffic to populate
the Activity and Usage pages. Nothing here forces a
first-attempt failure, so an ``absorbed`` attempt row appears only if a candidate
really does fail; point ``--model`` at something broken to see one on demand.

It talks to the gateway over HTTP like any operator would, so it works against
localhost or a deployed instance and needs nothing but the master key.

    python scripts/seed_routing_demo.py --url http://localhost:8000 --key <master-key>

Nothing here is otari-internal: every call is a documented endpoint. Provider
calls will fail unless the models named actually exist on your gateway; pass
``--model`` / ``--fallback-model`` to point at models you have. Failures are
reported rather than hidden, since a failed request is itself something the
Activity page is meant to show.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any

DEMO_USER = "routing-demo"


def call(
    url: str,
    key: str,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    """One management call. Returns ``(status, parsed_body_or_text)``."""
    request = urllib.request.Request(
        url.rstrip("/") + path,
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Otari-Key": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode()
            return response.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw
    except urllib.error.URLError as exc:
        print(f"  cannot reach {url}: {exc.reason}", file=sys.stderr)
        raise SystemExit(1) from exc


def report(label: str, status: int, body: Any) -> bool:
    ok = 200 <= status < 300
    mark = "ok " if ok else "!! "
    detail = "" if ok else f"  ({status}: {json.dumps(body)[:160]})"
    print(f"  {mark}{label}{detail}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000", help="Gateway base URL.")
    parser.add_argument("--key", required=True, help="Master key.")
    parser.add_argument("--model", default="openai:gpt-4o-mini", help="Primary model for the policies.")
    parser.add_argument(
        "--fallback-model",
        default="anthropic:claude-3-5-haiku-latest",
        help="Second model, used as the on_failure candidate.",
    )
    parser.add_argument(
        "--cheap-model",
        default=None,
        help=(
            "Cheaper model for the tier-down and learned policies. Defaults to --model, which makes the "
            "learned policy's candidate pool a duplicate pair, so that policy is skipped unless this is set."
        ),
    )
    parser.add_argument("--requests", type=int, default=6, help="How many requests to send per policy.")
    parser.add_argument(
        "--examples",
        type=int,
        default=8,
        help=(
            "Minimum scored examples to teach the learned policy. Raised to the gateway's seed count if "
            "that is higher, since a pool below it never routes. Each example costs an embedding call."
        ),
    )
    args = parser.parse_args()

    cheap = args.cheap_model or args.model
    url, key = args.url, args.key

    print(f"Seeding {url}")

    status, body = call(url, key, "GET", "/health")
    if not report("gateway reachable", status, body):
        return 1

    print("\nUser and budget")
    call(url, key, "POST", "/v1/users", {"user_id": DEMO_USER, "alias": "Routing demo"})
    print(f"  ok user {DEMO_USER}")
    status, budget = call(url, key, "POST", "/v1/budgets", {"name": "routing-demo", "max_budget": 5.0})
    budget_id = budget.get("budget_id") if isinstance(budget, dict) else None
    if budget_id:
        call(url, key, "PATCH", f"/v1/users/{DEMO_USER}", {"budget_id": budget_id})
        print("  ok budget attached ($5)")

    print("\nPolicies")
    policies: list[tuple[str, dict[str, Any]]] = [
        # The shape most people want first: one model, with a second to fall over to.
        (
            "demo-failover",
            {
                "select": [{"default": args.model}],
                "on_failure": [args.fallback_model],
            },
        ),
        # The alias-equivalent: one candidate, no chain.
        ("demo-simple", {"select": [{"default": args.model}]}),
        # Budget tier-down, which makes the policy dynamic (no single price).
        (
            "demo-thrifty",
            {
                "select": [
                    {"when": {"budget_used_pct": {"gte": 60}}, "target": cheap},
                    {"default": args.model},
                ],
                "on_failure": [args.fallback_model],
            },
        ),
        # Learned routing: the router ranks the pool per request and falls back to
        # the default whenever it declines. Refused with a 400 unless both candidates
        # have pricing, which is the point of that check. Only added when the two
        # models actually differ: a pool that lists one model twice is refused as a
        # duplicate, so with no --cheap-model this policy is skipped rather than
        # reported as a failure the operator cannot act on.
        *(
            [
                (
                    "demo-learned",
                    {
                        "select": [
                            {"router": "knn", "candidates": [cheap, args.model]},
                            {"default": args.model},
                        ],
                    },
                )
            ]
            if cheap != args.model
            else []
        ),
        # A guardrail the caller cannot skip. on_unavailable=monitor so a gateway
        # without a guardrails service running still serves these requests.
        (
            "demo-guarded",
            {
                "select": [{"default": args.model}],
                "guardrails": [
                    {"profile": "prompt-injection", "mode": "block", "on_unavailable": "monitor"}
                ],
            },
        ),
    ]
    for name, spec in policies:
        status, body = call(url, key, "POST", "/v1/routing/policies", {"name": name, "spec": spec})
        report(f"policy {name}", status, body)

    print("\nWhat they compile to")
    for name, _ in policies:
        status, body = call(url, key, "POST", "/v1/routing/policies/explain", {"name": name})
        if 200 <= status < 300 and isinstance(body, dict):
            chain = " -> ".join(c["dispatch_model"] for c in body["candidates"]) or "(nothing usable)"
            dropped = "".join(f"\n      dropped {d['selector']}: {d['detail']}" for d in body["dropped"])
            print(f"  {name}: {chain}  [{body['selection_reason']}]{dropped}")
        else:
            report(f"explain {name}", status, body)

    print(f"\nTraffic ({args.requests} requests per policy)")
    for name, _ in policies:
        served = failed = 0
        for index in range(args.requests):
            status, body = call(
                url,
                key,
                "POST",
                "/v1/chat/completions",
                {
                    "model": name,
                    "user": DEMO_USER,
                    "messages": [{"role": "user", "content": f"Say hello in {index + 2} words."}],
                    "max_tokens": 24,
                },
            )
            if 200 <= status < 300:
                served += 1
            else:
                failed += 1
                if failed == 1:
                    print(f"  {name}: first failure {status}: {json.dumps(body)[:200]}")
        print(f"  {name}: {served} served, {failed} failed")

    # Teaching in one call, and enough examples to cross the gateway's seed count:
    # the point of the demo is a policy that routes, and a pool below the seed count
    # serves the default on every request, which looks identical to a broken router.
    if cheap != args.model:
        status, body = call(url, key, "GET", f"/v1/routing/status?user_id={DEMO_USER}")
        seed = body.get("seed_count", 20) if 200 <= status < 300 and isinstance(body, dict) else 20
        count = max(args.examples, seed)
        print(f"\nTeaching demo-learned ({count} examples, seed count {seed})")
        examples: list[dict[str, Any]] = []
        for index in range(count):
            easy = index % 2 == 0
            examples.append(
                {
                    "prompt": (
                        f"What is {index + 3} plus {index + 4}?"
                        if easy
                        else f"Prove, in {index + 3} steps, why entropy increases in a closed system."
                    ),
                    # Easy prompts: both answered fine, so cost decides. Hard prompts:
                    # only the stronger model is good enough.
                    "scores": {cheap: 1.0, args.model: 1.0} if easy else {cheap: 0.0, args.model: 1.0},
                }
            )
        status, body = call(
            url,
            key,
            "POST",
            "/v1/routing/preferences/rank",
            {"user_id": DEMO_USER, "examples": examples},
        )
        if report(f"{count} examples recorded", status, body) and isinstance(body, dict):
            for pool in body.get("pools", []):
                name = pool["task_id"] or "default pool"
                state = "routing" if pool["warm"] else "still warming up"
                print(f"     {name}: {pool['records']}/{body['seed_count']} ({state})")
    else:
        print("\nSkipping demo-learned: pass --cheap-model to give the router two models to choose between.")

    print("\nDone. Open the dashboard and look at:")
    print("  Routing   the policies, their chains, the dry run, and Router on the learned row")
    print("  Activity  the Routing column, and any `absorbed` row from a fallover")
    print("  Usage     spend for the traffic above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
