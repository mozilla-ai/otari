# Load test

Compares two builds of Otari under load on one machine, route by route, and
fails when the second one got slower or does more database work per request.
CI runs it on every PR that changes what a request does, against the PR's base,
once the PR is out of draft.

```bash
docker build -t otari:base /path/to/base-checkout
docker build -t otari:head ..
./loadtest/run.sh otari:base otari:head
```

The output is pytest's, one line per check, and the command exits non-zero
when one fails:

```
comparison.py::test_every_planned_run_completed PASSED
comparison.py::test_head_served_every_request PASSED
comparison.py::test_no_extra_database_statements[direct] PASSED
comparison.py::test_no_extra_database_statements[spill] PASSED
comparison.py::test_no_latency_regression[direct] PASSED
comparison.py::test_no_latency_regression[pooled] FAILED
...
```

followed by the report: each route's p50 per turn for both builds, the change,
and which statements a request runs more or less often.
`loadtest/results/ab-TIMESTAMP/perf.json` holds every figure for tooling and
agents; `report.md` is rendered from it.

## What runs

`docker-compose.yml` starts Postgres, Redis, a fake provider that answers after
a fixed 50 ms, and one Otari replica pinned to cores 0-1, with everything else
on cores 2-3 (`LOADTEST_OTARI_CPUS`, `LOADTEST_OTHER_CPUS`), so the replica's
time is its own. The load comes from a tools container on the same network.
`otari-config.yml` caps Model 1 at 100 requests a minute and spills the
`summarize` policy to Model 2 once it is full.

Each build gets a database of its own, since a migration in head would break
base on a shared one. The builds take four turns each, alternating base, head,
head, base, so drift in the machine's speed lands on both, and each turn starts
a fresh replica (a process gets slower as it ages) and discards a warm-up.

In each turn, each route gets 6 seconds at 1,000 requests a minute, open loop:

| route | what it sends |
|---|---|
| `direct` | non-streamed, straight to Model 2 |
| `pooled` | the same, on a key with a shared budget every request debits |
| `spill` | non-streamed at the policy, with Model 1's minute already full |
| `stream` | streamed, straight to Model 2; measured at the first token |

Then each build's database statements per request are counted over 100
identical requests, direct and spilled.

## What the checks judge

The figure is the gateway's overhead: the client's latency minus the fake
provider's 50 ms (its time to first token when streamed), as each turn's p50.

- `test_no_latency_regression[ROUTE]` fails when **every** head turn is slower
  than **every** base turn by more than the larger of 1 ms and 10% of base
  (identical builds' turns have differed by up to 6%). The report calls the
  mirror case an improvement, and anything else unchanged. Something else
  loading the machine during a turn can only blur the two builds together,
  never pull them apart, unless it hits every turn of one build and none of the
  other's, which the alternating order makes unlikely. Because the spread
  between one build's turns adds to that margin, the slowdown it takes to fail
  is larger than 10%: on a GitHub runner, about 12% on `direct` and up to 20%
  on `pooled`. A smaller regression reads unchanged.
- `test_no_extra_database_statements[direct|spill]` fails on a whole extra data
  statement per request. BEGIN, COMMIT and ROLLBACK are listed but not judged:
  background transactions land in their count as fractions.
- `test_head_served_every_request` fails on any failed request or cut stream.
- `test_every_planned_run_completed` fails when a turn crashed or served nothing.

For a cost that is deliberate, `LOADTEST_ACCEPT_REGRESSION=1` turns a
regression into an expected failure; in CI, add the `perf-accepted` label,
which re-runs the job.

In CI the report is the job summary. On a PR from this repository it is also
posted as one comment, edited on each run, once a run has something to say: a
route or a statement count changed, or a check failed. A PR where everything
reads unchanged gets no comment.

## Limits

- Absolute latency here is not a production number, and another workload on
  the machine moves it; at worst that hides a change, it does not invent one.
- The fake provider's latency is fixed, which keeps the overhead figure clean
  but tests none of a real provider's variance. The prompt is one fixed size.
- One replica: what two replicas sharing Postgres and Redis do to each other
  is not measured.
