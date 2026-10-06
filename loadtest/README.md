# Load test

A local stack for measuring Otari under load: two replicas behind nginx,
sharing one Postgres and one Redis, in front of a fake provider. It runs on
Docker alone. The load generator, tenant setup, checks and profilers run in
containers on the same network, so latency is not measured through Docker
Desktop's port forwarding.

```
loadgen ──> nginx (lb) ──> otari-1 ─┬─> fakeprovider  /m1 (Model 1, 100 RPM quota)
                       └─> otari-2 ─┤                 /m2 (Model 2)
                                    │                 /m3 (last-resort fallback)
                                    ├─> postgres
                                    └─> redis (rate_limit_store)
```

The gateway config (`otari-config.yml`) caps Model 1 at 100 requests a minute
with a `per: model` rule and spills to Model 2 with `router: priority`, with
per-user rate limits, a deployment-wide concurrency limit, per-end-user budgets
behind a service key and a shared budget on the key.

## Run it

```bash
./run.sh up          # builds otari:loadtest from this checkout, starts the stack
./run.sh baseline    # client + fake provider alone, the floor to subtract
./run.sh steady      # or any scenario below, or `all`
./run.sh down        # removes the stack and its database
```

To measure another branch, build the image from its checkout:
`OTARI_BUILD_CONTEXT=/path/to/checkout ./run.sh up`.

Each scenario creates its own tenant (a service user, a service key with a
per-end-user budget, and a shared budget on the key), runs the load, then runs
`check.py` against that tenant once its in-flight requests have settled. The
load generator's JSON is in `results/`, the tenant ids in `state/`.
`./run.sh check NAME` reruns the checks.

Knobs: `DURATION` (seconds per phase, default 120), `RPM_LOW`/`RPM_HIGH`
(300/1,000), `USERS` (200 end users), `STREAM_SHARE` (0.5), `FAKE_LATENCY_MS`
(300), `FAKE_M1_QUOTA_RPM` (100).

## Scenarios

| `./run.sh` | What it does |
|---|---|
| `steady` | 300 RPM, then 1,000 RPM. Latency, overhead, errors. |
| `spill` | 1,000 RPM non-streamed at the policy. Model 1 must take at most 100 a minute across both replicas. |
| `shared-budget` | 2,000 end users draining a $0.50 shared budget on the key. No overspend beyond what is in flight. |
| `budget-reset` | 50 end users on a $0.002 budget that resets every 60s. Each window resets once. |
| `provider-429` | The fake throttles every Model 1 call; traffic must move to Model 2. |
| `stream-fail` | 20% of streams drop after their first chunk. The client sees an error event, usage settles once. |
| `redis-down` | `docker compose stop redis` mid-run, then start. Requests keep flowing. |
| `kill-replica` | `kill -9` on otari-2 under long streams, restart, wait out the holds. |

## What the checks mean

`check.py` reads the database and the fake provider's counters:

- **Ledgers match usage rows.** For each end user, the spend its reset logs
  recorded plus its spend now equals the sum of its usage rows. This holds
  across resets, so it is exact rather than "close".
- **Nothing held after drain.** No reserved spend, tokens or requests on any end
  user or on the shared budget, and no reservation left `active`.
- **Resets.** No end user reset more often than its period allows.
- **Shared budget overspend.** The ceiling on the key never went past its cap by
  more than what was in flight.
- **Model 1's cap.** The most Model 1 accepted in any 60s window, as the fake
  provider counted it, and how many calls the provider had to refuse. The fake
  enforces the same 100 RPM as the gateway, so a refusal there means the
  gateway let one through. Skipped where the scenario breaks it on purpose.
- **Client side.** No 429, no 5xx, no dropped connection, and every stream
  ended in `[DONE]` or an error event.

Overhead is the client's latency on a non-streamed call minus the fake
provider's fixed latency. It includes nginx and the network hops, so compare
it with the `baseline` run's figure (about 6 ms p50 and 14 ms p99 on an
M-series Mac).

## Measuring a change

Two commands give the numbers a performance change is judged on. Build the
image from the checkout before the change, run both, then rebuild from the
checkout after it and run them again.

```bash
./run.sh bench before      # profiled 1,000 RPM, without and with a shared budget
./run.sh count             # database statements per request, straight to Model 2
./run.sh count spill       # the same for requests that spill past Model 1
```

`bench` runs `DURATION` seconds at `RPM_HIGH` on a fresh tenant with no shared
budget and then on one with it, profiling both, and checks each afterwards.
`count` sends 100 identical non-streamed requests on end users that already
exist and reads `pg_stat_statements`, leaving out PostgreSQL's own foreign-key
checks; the list says which statements a request runs and how often.

## Profiling

`PROFILE=1 ./run.sh steady` (any scenario; `bench` always profiles) writes
`results/profile-NAME-TIMESTAMP/`:

- **`report.md`**, with
  - **Gateway:** each replica's Prometheus counters and histograms over the
    run (requests by status, CPU, memory, usage-log flushes, rate-limit and
    budget counters, abandoned attempts).
  - **Postgres:** the top statements by total time from `pg_stat_statements`,
    and what the active backends were waiting on, sampled every 0.25s. A hot
    row shows up as `Lock:transactionid` waits, with the query that was
    blocked.
  - **Redis:** per-command call counts and latency.
  - **CPU:** where each replica spent its busy time, by component (request
    handling, SQLAlchemy statements and cache keys, asyncpg, the provider SDK,
    client construction, pydantic) and the top functions by their own time.
- **`otari-1.txt`, `otari-2.txt`:** the py-spy recordings as collapsed stacks.
  Open one in [speedscope](https://www.speedscope.app) for a flame graph, or
  run `docker compose run --rm tools profile.py analyze results/.../otari-1.txt`
  for the breakdown on its own. `PROFILE_FORMAT=flamegraph` records SVG flame
  graphs instead.

py-spy runs as a sidecar sharing each replica's PID namespace and records with
`--nonblocking`, so it does not pause the gateway it is measuring.

## Limits of a laptop run

- Both replicas, Postgres and the load generator share one machine, so absolute
  latency here is not a production number, and another workload on the machine
  moves it. Compare against `baseline`, compare runs with each other, and rerun
  a figure that looks off before trusting it.
- The fake provider's latency is fixed, which keeps the overhead figure clean
  but tests none of a real provider's variance.
- The request mix (prompt sizes, streaming share, user count) is an assumption.
  `loadgen.py --prompt-mix` and `--stream-share` take a real one.
- `budget_reservation_ttl_sec` and the concurrency rule's `lease_sec` are set
  to two minutes so `kill-replica` sees holds expire within the run. A request
  slower than that has its hold reclaimed while in flight, which a run where
  the gateway falls behind will show as expired reservations.
