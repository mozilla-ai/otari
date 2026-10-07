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

From this directory (`cd loadtest`), with Docker running:

```bash
./run.sh up          # builds otari:loadtest from this checkout, starts the stack
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
- **Shared budget overspend.** The ceiling on the key ended no more than $0.05
  over its cap (`--overspend-slack`), which is room for the holds in flight when
  it ran dry.
- **Model 1's cap.** The most Model 1 accepted in any 60s window, as the fake
  provider counted it, and how many calls the provider had to refuse. The fake
  enforces the same 100 RPM as the gateway, so a refusal there means the
  gateway let one through. Skipped where the scenario breaks it on purpose.
- **Client side.** No 429, no 5xx, no dropped connection, and every stream
  ended in `[DONE]` or an error event. `kill-replica` keeps only the 429 check,
  since the requests on the killed replica fail by design.

`run.sh` exits non-zero when any check failed, so `./run.sh all` can gate.
`SCENARIOS="steady spill" ./run.sh all` runs a subset.

Overhead is the client's latency on a non-streamed call minus the fake
provider's fixed latency, nginx and the network hops included.

## Comparing two builds

`./run.sh ab BASE_IMAGE HEAD_IMAGE` runs two images on this machine in turn
and compares them, which is how CI judges a PR:

```bash
docker build -t otari:base /path/to/main-checkout
docker build -t otari:head ..
export LOADTEST_PIN=1 FAKE_LATENCY_MS=50 FAKE_TTFT_MS=50   # as CI runs it
OTARI_IMAGE=otari:head LOADTEST_BUILD=0 ./run.sh up
./run.sh ab otari:base otari:head
```

The load goes straight to `otari-1`, with `otari-2` and nginx stopped: they
would add noise that belongs to neither build. `LOADTEST_PIN=1` adds `pin.yml`,
which gives `otari-1` cores of its own (`PIN_GATEWAY_CPUS`, default `0-1`) and
puts everything else on others (`PIN_OTHER_CPUS`, `2-3`); set it on every
`run.sh` call, `up` included. Each image gets a database of its own (`ab_base`,
`ab_head`), since a migration in head would break base on a shared one.

Each scenario is a route or hot path, run in two modes:

| scenario | what it sends |
|---|---|
| `direct` | non-streamed, straight to Model 2, no shared pool on the key |
| `pooled` | the same, with a shared pool on the key (one row every request locks) |
| `spill` | non-streamed at the policy with Model 1's minute already full |
| `stream` | streamed, straight to Model 2; measured at the first token |

- **idle**: one request at a time for `AB_IDLE_SECONDS` (10). Nothing queues,
  so this is the gateway's own path.
- **loaded**: `AB_RPM` (1,000) open loop for `AB_SECONDS` (12). A costlier path
  shows up here as queueing. Keep the rate below what the machine saturates at.

The builds alternate base, head, head, base, ... (`AB_ROUNDS`, default 4),
each block on a freshly started replica after a discarded warm-up, so drift in
the machine's speed lands on both and neither build is measured on an older
process (a process gets slower as it ages).
`AB_SCENARIOS` runs a subset. Then each build's statements per request are
counted, direct and spilled, and head's tenants are checked.

`ab.py report` writes `results/ab-TIMESTAMP/perf.json` and renders `report.md`
from it. The figure is the gateway's overhead: the client's latency minus the
fake provider's. For each scenario and mode, the change in p50 and p90 gets a
95% interval from a hierarchical bootstrap (resample each build's runs, then
the requests within them, so a run that went slow as a whole widens the
interval instead of reading as a difference between builds), and a verdict:

- **regression** or **improvement**: the whole interval is beyond the smallest
  change worth reporting, the larger of 1 ms and 10% of base (identical
  builds have differed by up to 6% at p50)
  (`--min-effect-ms`, `--min-effect-pct`);
- **unchanged**: the whole interval is within it;
- **inconclusive**: neither, usually because the runs disagreed;
- **noisy**: one build's own runs of the scenario disagree on their p50 by more
  than 25% (`--max-run-spread`), so something else loaded the machine during
  some of them. No verdict; rerun on a quiet machine.

Statements are counted, not timed. The run fails on a whole extra data
statement per request, direct or spilled (BEGIN, COMMIT and ROLLBACK are listed
but not judged: background transactions land in their count as fractions), on
a run missing for either build, and on any failed request on head. A latency
regression fails it only with
`AB_FLAGS=--enforce-latency`; `AB_FLAGS=--allow-regression` reports and passes.

`perf.json` is the contract for tooling and agents: per scenario and mode, each
quantile's `base`, `head`, `delta`, `ci95` and `verdict`, CPU per request, and
every run's own figures; per statement mode, the counts and the statements
whose count per request changed. `report.md` adds nothing it does not hold.

## In CI

`.github/workflows/otari-loadtest.yml` runs on a PR that touches `src/`, the
migrations, dependencies, the Dockerfile or this directory, as two jobs on
separate runners:

- **ab**: the PR's merge commit against its base, as above. The report is the
  job summary and `perf.json` is in the job's artifact. For a cost that is
  deliberate, add the `perf-accepted` label, which re-runs the job.
- **scenarios**, only on a PR labeled `loadtest`: `steady`, `spill`,
  `shared-budget`, `provider-429` and `stream-fail` at 20s phases, with every
  check, and a $0.15 shared pool (`SHARED_POOL_USD`) so it runs dry within one.
  Unlabeled, they wait for the nightly run: they check ledgers under
  concurrency, which a change rarely moves, and a busy shared runner can fail
  them on 429s it caused itself.

Nightly, `scenarios` runs every scenario at full length and `ab` compares main
with itself. That A/A run should read unchanged or inconclusive, never a
regression or an improvement, and shows how often a runner comes out noisy;
`--enforce-latency` waits on it. The fake provider answers in 50 ms in CI
rather than 300.

## Profiling

`PROFILE=1 ./run.sh steady` (any scenario) writes
`results/profile-NAME-TIMESTAMP/`:

- **`report.md`**: the top statements by total time from `pg_stat_statements`,
  deadlocks, the most-updated tables, and what the active backends were
  waiting on, sampled every 0.25s. A hot row shows up as `Lock:transactionid`
  waits, with the query that was blocked.
- **`otari-1.txt`, `otari-2.txt`:** py-spy recordings as collapsed stacks.
  Open one in [speedscope](https://www.speedscope.app) for a flame graph.
  `PROFILE_FORMAT=flamegraph` records SVG flame graphs instead.

py-spy runs as a sidecar sharing each replica's PID namespace and records with
`--nonblocking`, so it does not pause the gateway it is measuring.

## Limits of a laptop run

- Both replicas, Postgres and the load generator share one machine, so absolute
  latency here is not a production number, and another workload on the machine
  moves it. Compare builds with `ab`, which alternates them on one machine.
- The fake provider's latency is fixed, which keeps the overhead figure clean
  but tests none of a real provider's variance.
- The request mix (prompt sizes, streaming share, user count) is an assumption.
  `loadgen.py --prompt-mix` and `--stream-share` take a real one.
- `budget_reservation_ttl_sec` and the concurrency rule's `lease_sec` are set
  to two minutes so `kill-replica` sees holds expire within the run. A request
  slower than that has its hold reclaimed while in flight, which a run where
  the gateway falls behind will show as expired reservations.
