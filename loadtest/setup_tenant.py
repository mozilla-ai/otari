# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Create one tenant for a scenario: a service, its key and its budgets.

Each scenario gets a tenant of its own, so check.py can verify it in isolation:

- a service user and a service key (``is_service_key``), so the load
  generator's ``user`` values become end users of that service;
- a per-end-user budget, copied onto every end user on first use;
- a shared pool: a scoped budget on the key, one row every end user debits,
  which is where lock contention would show.

    uv run setup_tenant.py --name steady --end-user-usd 5 --end-user-period 120

Writes state/<name>.json with the key and ids.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def call(base: str, master: str, method: str, path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(f"{base}/api/v1{path}", data=data, method=method)
    request.add_header("Authorization", f"Bearer {master}")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{method} {path} -> {exc.code}: {exc.read().decode()[:500]}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://lb")
    parser.add_argument("--master-key", default="loadtest-master-key")
    parser.add_argument("--name", required=True)
    parser.add_argument("--end-user-usd", type=float, default=100.0)
    parser.add_argument("--end-user-period", type=int, default=3600, help="seconds between end-user resets")
    parser.add_argument("--pool-usd", type=float, default=10_000.0, help="shared ceiling on the service key")
    parser.add_argument("--pool-period", type=int, default=86_400)
    parser.add_argument("--no-pool", action="store_true", help="no shared ceiling on the key")
    parser.add_argument("--state-dir", default="state")
    args = parser.parse_args()

    suffix = time.strftime("%H%M%S")
    service_user = f"loadtest-{args.name}-{suffix}"
    base, master = args.base_url, args.master_key

    end_user_budget = call(
        base,
        master,
        "POST",
        "/budgets",
        {
            "name": f"{args.name} end user",
            "max_budget": args.end_user_usd,
            "budget_duration_sec": args.end_user_period,
        },
    )
    pool_budget = call(
        base,
        master,
        "POST",
        "/budgets",
        {"name": f"{args.name} pool", "max_budget": args.pool_usd, "budget_duration_sec": args.pool_period},
    )
    call(base, master, "POST", "/users", {"user_id": service_user, "alias": f"Load test service ({args.name})"})
    key = call(
        base,
        master,
        "POST",
        "/keys",
        {
            "key_name": f"loadtest-{args.name}",
            "user_id": service_user,
            "is_service_key": True,
            "end_user_budget_id": end_user_budget["budget_id"],
        },
    )
    scoped = (
        {"id": None}
        if args.no_pool
        else call(
            base,
            master,
            "POST",
            "/scoped-budgets",
            {
                "scope_type": "api_token",
                "scope_id": key["id"],
                "budget_id": pool_budget["budget_id"],
                "name": f"{args.name} pool",
            },
        )
    )

    state = {
        "name": args.name,
        "created_at": time.time(),
        "service_user": service_user,
        "key": key["key"],
        "key_id": key["id"],
        "end_user_budget_id": end_user_budget["budget_id"],
        "end_user_usd": args.end_user_usd,
        "end_user_period": args.end_user_period,
        "pool_budget_id": pool_budget["budget_id"],
        "pool_usd": args.pool_usd,
        "scoped_budget_id": scoped["id"],
    }
    out = Path(args.state_dir) / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(state, indent=2))
    print(f"tenant {args.name}: service user {service_user}, key {key['id']}, state in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
