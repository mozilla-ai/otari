# /// script
# requires-python = ">=3.12"
# dependencies = ["psycopg[binary]>=3.2", "httpx>=0.27"]
# ///
"""Check a scenario's tenant after its run: do the ledgers agree with the usage rows?

Run it once the load has drained (and, after a replica kill, once holds have
had time to expire). Each check prints PASS or FAIL; the exit code is the
number of failures.

    uv run check.py --name steady [--loadgen-result results/steady-*.json]

Checks:

1. Every end user's ledger matches its usage rows: the spend each reset log
   recorded plus the spend now equals the sum of its usage rows' cost.
2. No end user holds anything any more (reserved, reserved tokens and requests).
3. No end user was reset more often than its period allows.
4. The shared pool (the scoped budget on the key) matches the key's usage, holds
   nothing, and overspent its cap by at most what was in flight.
5. No budget reservation of this tenant is still active.
6. Model 1 at the fake provider never accepted more than its cap in any 60s
   window (only meaningful once the gateway caps it, see README).
7. The client saw no 429 while a candidate had room, and every failed stream
   ended in an error event. ``--allow-dropped`` keeps only the 429 check, for a
   scenario that kills a replica mid-request on purpose.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from decimal import Decimal
from pathlib import Path

import httpx
import psycopg

failures = 0


def report(ok: bool, name: str, detail: str = "") -> None:
    global failures
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")


def _wait_for_drain(cur: psycopg.Cursor, service_user: str, timeout: float) -> None:
    """Wait until no reservation of this tenant is active, so the checks read a settled ledger."""
    deadline = time.time() + timeout
    while True:
        cur.execute(
            """
            SELECT COUNT(*) FROM budget_reservations r JOIN users u ON u.user_id = r.user_id
            WHERE u.parent_user_id = %s AND r.status = 'active'
            """,
            (service_user,),
        )
        active = cur.fetchone()[0]
        if not active:
            return
        if time.time() >= deadline:
            print(f"      {active} reservations still active after {timeout:.0f}s; checking anyway")
            return
        time.sleep(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--state-dir", default="state")
    parser.add_argument("--dsn", default="postgresql://otari:otari@postgres:5432/otari")
    parser.add_argument("--fake-url", default="http://fakeprovider:9000")
    parser.add_argument("--model1-cap", type=int, default=100)
    parser.add_argument("--cap-slack", type=int, default=5, help="requests over the cap still counted a pass")
    parser.add_argument("--loadgen-result", help="a loadgen JSON result to check client-side outcomes")
    parser.add_argument("--skip-cap", action="store_true", help="skip check 6 (no per-model cap configured)")
    parser.add_argument(
        "--allow-dropped",
        action="store_true",
        help="a replica was killed on purpose: 5xx, dropped connections and cut streams are expected",
    )
    parser.add_argument(
        "--overspend-slack",
        type=Decimal,
        default=Decimal("0.05"),
        help="USD the shared pool may end over its cap, for the holds in flight when it ran dry",
    )
    parser.add_argument(
        "--drain-timeout",
        type=float,
        default=120,
        help="seconds to wait for the tenant's in-flight requests to settle before checking",
    )
    args = parser.parse_args()

    state = json.loads((Path(args.state_dir) / f"{args.name}.json").read_text())
    print(f"tenant {args.name}: service user {state['service_user']}, key {state['key_id']}\n")

    with psycopg.connect(args.dsn, autocommit=True) as conn:
        cur = conn.cursor()
        _wait_for_drain(cur, state["service_user"], args.drain_timeout)

        # 1-3: per end user.
        cur.execute(
            """
            SELECT u.user_id, u.spend, u.reserved, u.reserved_tokens, u.reserved_requests,
                   u.budget_started_at,
                   COALESCE((SELECT SUM(l.previous_spend) FROM budget_reset_logs l WHERE l.user_id = u.user_id), 0),
                   (SELECT COUNT(*) FROM budget_reset_logs l WHERE l.user_id = u.user_id),
                   COALESCE((SELECT SUM(g.cost) FROM usage_logs g
                             WHERE g.user_id = u.user_id AND g.counts_toward_budget), 0),
                   (SELECT COUNT(*) FROM usage_logs g WHERE g.user_id = u.user_id),
                   u.created_at
            FROM users u WHERE u.parent_user_id = %s
            """,
            (state["service_user"],),
        )
        rows = cur.fetchall()
        report(bool(rows), "end users were created", f"{len(rows)} end users")
        drift, holding, over_reset = [], [], []
        total_usage = Decimal(0)
        now = time.time()
        for (
            user_id,
            spend,
            reserved,
            reserved_tokens,
            reserved_requests,
            _started,
            reset_spend,
            resets,
            usage_cost,
            _usage_rows,
            created_at,
        ) in rows:
            total_usage += usage_cost
            if abs((spend + reset_spend) - usage_cost) > Decimal("0.000001"):
                drift.append(f"{user_id}: ledger {spend + reset_spend} vs usage {usage_cost}")
            if reserved or reserved_tokens or reserved_requests:
                holding.append(f"{user_id}: {reserved} USD, {reserved_tokens} tok, {reserved_requests} req")
            allowed = math.ceil((now - created_at.timestamp()) / state["end_user_period"])
            if resets > allowed:
                over_reset.append(f"{user_id}: {resets} resets, at most {allowed} periods")
        report(not drift, "end-user ledgers match usage rows", "; ".join(drift[:5]))
        report(not holding, "end users hold nothing after drain", "; ".join(holding[:5]))
        report(not over_reset, "no end user reset more than once a period", "; ".join(over_reset[:5]))
        cur.execute(
            "SELECT COUNT(*) FROM budget_reset_logs l JOIN users u ON u.user_id = l.user_id "
            "WHERE u.parent_user_id = %s",
            (state["service_user"],),
        )
        print(f"      {cur.fetchone()[0]} end-user resets recorded")

        # 4: the shared pool.
        if state["scoped_budget_id"] is None:
            print("      no shared pool on this tenant")
        else:
            cur.execute(
                "SELECT current_spend, reserved_spend, reserved_tokens, reserved_requests "
                "FROM scoped_budgets WHERE id = %s",
                (state["scoped_budget_id"],),
            )
            current_spend, reserved_spend, reserved_tokens, reserved_requests = cur.fetchone()
            cur.execute(
                "SELECT COALESCE(SUM(cost), 0), COUNT(*) FROM usage_logs "
                "WHERE api_key_id = %s AND counts_toward_budget",
                (state["key_id"],),
            )
            key_usage, key_rows = cur.fetchone()
            report(
                abs(current_spend - key_usage) <= Decimal("0.000001"),
                "shared pool matches the key's usage rows",
                f"pool {current_spend} vs usage {key_usage} over {key_rows} rows",
            )
            report(
                not (reserved_spend or reserved_tokens or reserved_requests),
                "shared pool holds nothing after drain",
                f"{reserved_spend} USD, {reserved_tokens} tok, {reserved_requests} req",
            )
            overspend = current_spend - Decimal(str(state["pool_usd"]))
            report(
                overspend <= args.overspend_slack,
                f"shared pool overspend within ${args.overspend_slack}",
                f"cap {state['pool_usd']}, spent {current_spend}, over by {max(overspend, Decimal(0))}",
            )

        # 5: reservations.
        cur.execute(
            """
            SELECT r.status, COUNT(*) FROM budget_reservations r JOIN users u ON u.user_id = r.user_id
            WHERE u.parent_user_id = %s GROUP BY r.status
            """,
            (state["service_user"],),
        )
        statuses = dict(cur.fetchall())
        report(not statuses.get("active"), "no reservation still active", json.dumps(statuses))

    # 6: model 1's cap, as the fake provider saw it.
    stats = httpx.get(f"{args.fake_url}/_stats", timeout=10).json()
    m1 = stats["m1"]
    print(
        f"\n      fake m1: accepted {m1['accepted_total']}, max in 60s {m1['max_accepted_per_60s']}, "
        f"429s {m1['throttled_quota']} quota + {m1['throttled_injected']} injected, "
        f"streams failed {m1['streams_failed']}"
    )
    print(f"      fake m2: accepted {stats['m2']['accepted_total']}, m3: accepted {stats['m3']['accepted_total']}")
    if not args.skip_cap:
        report(
            m1["max_accepted_per_60s"] <= args.model1_cap + args.cap_slack and m1["throttled_quota"] == 0,
            "model 1 held to its cap by the gateway",
            f"max {m1['max_accepted_per_60s']} in 60s against {args.model1_cap}, "
            f"{m1['throttled_quota']} calls the provider had to refuse",
        )

    # 7: what the client saw.
    if args.loadgen_result:
        result = json.loads(Path(args.loadgen_result).read_text())
        refused = result["status_counts"].get("429", 0)
        report(refused == 0, "client saw no 429", json.dumps(result["refusals_429"])[:300])
        five_xx = sum(n for status, n in result["status_counts"].items() if status.startswith("5"))
        conn_errors = sum(result["client_errors"].values())
        if args.allow_dropped:
            print(
                f"      expected with a replica killed: {five_xx} 5xx, {conn_errors} connection errors, "
                f"{result['streams_without_done']} streams cut"
            )
        else:
            report(
                five_xx == 0 and conn_errors == 0,
                "client saw no 5xx or dropped connection",
                f"{five_xx} 5xx, {conn_errors} connection errors",
            )
            report(
                result["streams_without_done"] == 0,
                "every stream ended in [DONE] or an error event",
                f"{result['streams_without_done']} streams just stopped",
            )
        print(f"      client: {json.dumps(result['status_counts'])}, served by {json.dumps(result['served_by'])}")
        print(f"      latency {json.dumps(result['latency_ms'])}")
        print(f"      overhead {json.dumps(result['gateway_overhead_ms'])}")

    print(f"\n{failures} check(s) failed")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
