#!/usr/bin/env python3
"""Seed a running gateway with routing policies and traffic, to click through.

Creates a set of policies covering the shapes the feature supports, then drives
enough traffic to populate the Activity and Usage pages. Nothing here forces a
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
        help="Cheaper model for the budget tier-down policy. Defaults to --model.",
    )
    parser.add_argument("--requests", type=int, default=6, help="How many requests to send per policy.")
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

    print("\nDone. Open the dashboard and look at:")
    print("  Routing   the four policies, their chains, and the dry run")
    print("  Activity  the Routing column, and any `absorbed` row from a fallover")
    print("  Usage     spend for the traffic above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
