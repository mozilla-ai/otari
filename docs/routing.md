# Routing policies

A **routing policy** is a model name you define. Callers send it in the `model`
field like any other model, and the policy decides which real model serves the
request, what to try if that model fails, and which guardrails always run.

An alias is the simplest possible policy: one name, one target. That is why
[`aliases:`](models.md#model-aliases) keeps working and is documented as the
shorthand. Reach for a policy when you want more than one target, a condition, or
an enforced guardrail.

Aliases stored in the database were **moved into policies** by migration
`b5d7f9a1c3e6`, so there is one store and one dashboard page for the concept.
Nothing about how they resolve changed: a moved alias is a policy whose `select`
is a single `default`. The `aliases:` block in `config.yml` is untouched and still
works.

> **Breaking change for `/v1/aliases` callers.** The endpoint still exists and
> still creates one-target aliases, but the rows the migration moved are no longer
> aliases. For an alias that existed before the upgrade: `GET /v1/aliases` no
> longer lists it, `DELETE /v1/aliases/{name}` returns 404, and `POST /v1/aliases`
> with that name returns 400 because the name is now a routing policy. Manage
> those through `/v1/routing/policies` or the dashboard's Routing page instead.
> The dashboard needs no change: it reads both stores. A script driving
> `/v1/aliases` against pre-upgrade names does, and rolling the binary back
> without also running `alembic downgrade` leaves those rows unreadable, because
> the old binary does not know about `routing_policies`.

Standalone mode only. In hybrid mode the connected platform resolves the model for
every request, so a policy name is not a model it knows; sending one returns a 400
that says so rather than a confusing upstream 404.

## The smallest useful policy: failover

```yaml
routing:
  policies:
    fast:
      select:
        - default: openai:gpt-5-mini
      on_failure:
        - anthropic:claude-haiku-4-5
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Otari-Key: Bearer $OTARI_KEY" \
  -d '{"model": "fast", "messages": [{"role": "user", "content": "hello"}]}'
```

If `openai:gpt-5-mini` fails before it has produced a response, Anthropic serves the request
instead and the caller never sees the difference: the response `model` says
`fast`, and the billed usage row names the model that actually served. The
attempt that failed gets its own row too, with `status: "absorbed"`, so the
failover is visible in the activity log without counting as an error or as a
second request (see [What is billed](#what-is-billed-and-what-the-caller-sees)).

Before this existed, a standalone gateway had no failover at all. A provider blip
was a 502.

## The two axes

`select` decides where the plan **starts**. `on_failure` decides what is tried
**after a failure**. They are separate keys because "this entry did not apply" and
"this entry failed" are different events, and a single ordered list cannot tell
you which one happened.

```yaml
routing:
  policies:
    thrifty:
      select:
        - when: {budget_used_pct: {gte: 80}}
          target: openai:gpt-5-nano       # tier down as the budget fills up
        - default: openai:gpt-5-mini
      on_failure:
        - anthropic:claude-haiku-4-5
```

`select` entries are evaluated in order, and the first whose `when` matches wins.
The `default` entry is the fallthrough and must come last: an entry after it could
never be reached, so the gateway refuses to start rather than leave you with a
silently dead rule. A third kind of entry, `router`, hands the choice to a backend
that orders the candidates per request: `weighted` to [split the traffic by
weight](#load-balance-across-providers-weighted-routing), `knn` to [learn which
prompts a cheaper model handles](#let-a-router-choose-learned-routing).

### `when` conditions

All conditions present in a `when` clause must match. The set is closed, so a typo
is refused at startup instead of quietly never matching.

| Condition | Type | Notes |
| --- | --- | --- |
| `budget_used_pct` | comparison | Percent of the caller's budget already committed (`spend + reserved`, the same total the budget gate enforces). |
| `budget_remaining_usd` | comparison | USD left before the cap. |
| `user_id` | string or list | Matches the billed user. |
| `key_id` | string or list | Matches the calling API key's id. |

A comparison is `{gte: 80}`, `{gt: 80}`, `{lte: 20}`, or `{lt: 20}` (exactly one).

Two rules worth knowing, because both are silent-failure traps otherwise:

- **A budget condition never matches when the number is undefined**, which is the
  case for a caller with no budget or an unlimited budget. The policy falls
  through to `default`. It does not raise: "no budget configured" must not turn
  into an error on every request. A master-key request is not one of these cases:
  it has to name the billed user, and conditions are evaluated against that
  user's budget, so a master-key request can take a tier-down branch.
- **A `budget_used_pct` threshold of `gte` or `gt` 100 or above is refused at
  startup.** The budget gate rejects a request before selection happens, so such a
  rule could never fire. Tiering down keeps a caller *under* a cap; it is not a way
  to keep serving past one. `lt`/`lte` thresholds are not restricted: "still under
  the cap" is a reachable condition.

## Load balance across providers (weighted routing)

A `select` entry can name the `weighted` router, which draws one candidate per
request in proportion to the weights you give it.

```yaml
routing:
  policies:
    balanced:
      select:
        - router: weighted
          candidates: [openai:gpt-5, anthropic:claude-sonnet-4-5]
          weights:
            openai:gpt-5: 70
            anthropic:claude-sonnet-4-5: 30
        - default: openai:gpt-5        # serves when a caller opts out
      on_failure: [gemini:gemini-2.0-flash]
```

Callers keep sending `balanced`, and about seven requests in ten go to OpenAI. The
response `model` says `balanced` whichever provider served, so moving the split is
a config change and never a client change.

Weights are **normalized, not percentages**: `{70, 30}` and `{7, 3}` are the same
split, so a ratio can be written as a ratio. What they express is capacity you are
choosing to use, not anything the gateway derives, which is why (unlike the
[learned router](#let-a-router-choose-learned-routing)) the split itself reads no
[pricing](configuration.md) row: an unpriced candidate is not refused at write
time and does not make the router decline. The gateway's own billing gate is
separate and still applies, so with `require_pricing` on (the default) a metered
caller drawn onto an unpriced candidate gets a 402 like any other unpriced model.

**A candidate left out of `weights` gets zero**, which is how a provider is drained
without being deleted:

```yaml
          weights:
            openai:gpt-5: 100
            anthropic:claude-sonnet-4-5: 0    # no weighted traffic, still catches a failure
```

A zero-weight candidate stays in the plan at its tail, so it backs up a failure
while receiving none of the weighted traffic. It is never drawn, which is not quite
the same as never serving: if it is also the policy's `default` target, it still
serves a caller who sends `Otari-Router: off`. Set both sides to zero and the policy is refused
at startup: it could never select anything. An even split is written out
(`{a: 1, b: 1}`) rather than implied by omitting the map, because omitting a
*candidate* already means zero and one key cannot mean both.

### What it does and does not promise

- **Each request is an independent draw.** Nothing is remembered between requests,
  so the split is exactly as correct behind twenty replicas as behind one, and none
  of the caveats in [Routing at scale](routing-scaling.md) apply. The ratio
  converges statistically: a burst of ten requests is not necessarily seven and
  three.
- **A failure stays inside the pool.** The draw continues without replacement, so
  the whole ordering is the plan: a candidate that fails before it has produced a
  response hands the request to another weighted provider (itself chosen by weight),
  and `on_failure` is only reached once the pool is exhausted. A provider having a
  bad minute therefore sheds its share to the others without any health tracking.
- **No stickiness.** A conversation can move between providers turn to turn, which
  can cost a warm prompt cache. If that matters more than the split, the learned
  router's `trace_sticky` behavior is the shape to look at; weighted routing
  deliberately keeps no per-conversation state.
- **No health awareness.** Weights are not adjusted when a provider starts failing.
  Failover is what handles that, one request at a time.
- **`Otari-Router: off` serves the `default` target**, exactly as it does for the
  learned router. On a weighted policy that pins the caller to one provider, which
  is a useful escape hatch during an incident.

### Reading it back

The usage row names the model that served and carries
`selection_reason: router:weighted`, which is where the split is verifiable after
the fact. A weighted decision is deliberately **not** logged per request: at load
balancer volume that line would be the log, and unlike a learned router's pick it
is reconstructable from the policy plus the usage rows. Raise the gateway to debug
to see each draw.

`otari routing explain` shows the split itself, because a weighted decision needs
no request state:

```console
$ otari routing explain balanced
balanced: 3 candidate(s), selected by router:weighted
  1. openai:gpt-5    [weighted 70%]  dispatches as openai:gpt-5
  2. anthropic:claude-sonnet-4-5    [weighted 30%]  dispatches as anthropic:claude-sonnet-4-5
  3. gemini:gemini-2.0-flash    [on_failure]  dispatches as gemini:gemini-2.0-flash
```

Shares are normalized over the candidates the caller may actually use, so a policy
whose heavy candidate is filtered out by an allow-list reports the split that is
really running (and lists the dropped candidate with its reason):

```console
$ otari routing explain balanced --allowed-model 'anthropic:*'
balanced: 1 candidate(s), selected by router:weighted
  1. anthropic:claude-sonnet-4-5    [weighted 100%]  dispatches as anthropic:claude-sonnet-4-5
  x  openai:gpt-5    dropped: is not in allowed_models for this caller
  x  gemini:gemini-2.0-flash    dropped: is not in allowed_models for this caller
```

The `on_failure` entry is filtered by the same allow-list, so this caller has no
fallback left either: a split that reads as three providers deep is one provider
deep for them, which is the kind of thing this command exists to surface.

`POST /v1/routing/policies/explain` returns the same numbers as `router_weights`.
The dashboard's Routing page shows a split too, but it is the declared one, read
from the policy and not narrowed by any caller's allow-list, so this command is
where a per-caller answer comes from.

Standalone only, like every policy. The candidate cap counts the whole pool plus
`on_failure`, so a three-provider split leaves room for two failover entries.

## Let a router choose (learned routing)

The third shape a `select` entry can take. Instead of naming the target yourself,
hand the ordering to a **router**: it ranks a pool of candidates per request and
the plan starts with its pick.

```yaml
routing:
  policies:
    smart:
      select:
        - router: knn
          candidates: [openai:gpt-5-nano, openai:gpt-5]
        - default: openai:gpt-5        # serves whenever the router declines
      on_failure: [anthropic:claude-haiku-4-5]
```

Most requests do not need your most expensive model, and which ones do is a
property of your traffic rather than of a benchmark. The `knn` router learns that
from examples you score: you show it prompts, say how good each candidate's answer
was, and it sends look-alike prompts to the cheapest candidate that was good
enough. Until it has been taught, and whenever it is not confident, `default`
serves, so a learned policy is never worse than the plain failover policy it was
written from.

The `default` target is part of the pool: the gateway appends it if `candidates`
leaves it out, because what serves on a decline is always one of the models the router
could have picked. The dashboard shows it that way too, as one list with the fallback
marked.

Standalone only, like every policy. The pool needs [pricing](configuration.md) for
every candidate: the router weighs quality against cost, so an unpriced candidate
has nothing to weigh (a stored policy with one is refused; a `config.yml` one warns
at startup and never routes).

### How it decides

For each request the router embeds the prompt, finds the `k` nearest prompts it has
been taught, and scores each candidate:

```
score(model) = mean_quality(model | neighbors) - alpha * normalized_cost(model)
```

`alpha` is the one dial: 0 ignores cost and always picks the best-predicted model;
higher leans harder on the cheaper candidate. The whole ranking becomes the plan,
so a routed request that fails over lands on the router's second choice before it
reaches `on_failure`.

It declines, and `default` serves, whenever it would be guessing:

| It declines when | Because |
| --- | --- |
| The pool has fewer than `OTARI_ROUTER_SEED_COUNT` examples | Nothing to vote over |
| Fewer than `OTARI_ROUTER_K` comparable examples exist | The neighborhood is too sparse to read |
| Neighbor support is below `OTARI_ROUTER_CONFIDENCE_FLOOR` | The nearby prompts do not back the pick |
| The request carries tools | Capability gating is minimal in v1, so tool calls stay on `default` |
| A candidate has no pricing, or the embedding call fails | It cannot compare, and it must not fail the request |
| The caller sent `Otari-Router: off` | The caller knows this one is hard |

A decline is normal operation, not an error. The usage row's `selection_reason`
says `default` when the router declined and `router:knn` when it chose, so "the
router picked the strong model" and "the router did not run" stay distinguishable
after the fact.

### Teach it

Teaching is an API job in this release: the dashboard shows a learned policy and how
warm it is, but recording examples is `POST /v1/routing/preferences/rank`. Nothing is
learned from live traffic yet either (that is a fast-follow on
[#187](https://github.com/mozilla-ai/otari/issues/187)), so every example comes from
you or from a judge you run.

**Before the first example**, four things have to be true, and each one fails
differently if it is not:

| Requirement | If it is missing |
| --- | --- |
| The user exists (`POST /v1/users`) | `rank` returns 404 naming the user |
| Every candidate has [pricing](configuration.md) | writing the policy returns 400; the router scores by cost |
| A provider is configured for `OTARI_ROUTER_EMBEDDING_MODEL` (default `openai:text-embedding-3-small`) | `rank` returns 502 naming the model |
| The score keys name the policy's candidates | `rank` returns 400 listing what it can accept |

**How many examples.** A pool routes nothing until it holds `OTARI_ROUTER_SEED_COUNT`
records (default 20) and each decision reads the `OTARI_ROUTER_K` nearest (default 5),
so aim for at least `k` examples of *each kind of prompt* you care about, not 20 of
one kind. Twenty all-ties examples warm the pool and then send everything to the cheap
model. For a first trial, restart with `OTARI_ROUTER_SEED_COUNT=8` and teach four easy
plus four hard; the seed count is read when the router is built, so it needs a restart.

```bash
# Score a batch, 0 (bad) to 1 (great) per candidate. One call, not one per example.
curl -X POST http://localhost:8000/v1/routing/preferences/rank \
  -H "Otari-Key: Bearer $MASTER_KEY" -H "Content-Type: application/json" \
  -d '{
        "user_id": "alice",
        "examples": [
          {"prompt": "what is 18 + 24?",              "scores": {"openai:gpt-5-nano": 1.0, "openai:gpt-5": 1.0}},
          {"prompt": "add 7 and 31",                  "scores": {"openai:gpt-5-nano": 1.0, "openai:gpt-5": 1.0}},
          {"prompt": "prove the halting problem is undecidable",
                                                      "scores": {"openai:gpt-5-nano": 0.0, "openai:gpt-5": 1.0}},
          {"prompt": "derive Black-Scholes from first principles",
                                                      "scores": {"openai:gpt-5-nano": 0.0, "openai:gpt-5": 1.0}}
        ]
      }'
# -> {"recorded":4,"seed_count":8,"pools":[{"task_id":null,"records":4,"warm":false}]}

# Watch each pool warm up.
curl "http://localhost:8000/v1/routing/status?user_id=alice" -H "Otari-Key: Bearer $MASTER_KEY"

# Then send a request through the policy and read what actually served: the response
# `model` field says the policy name, so the usage row is where the answer is.
curl "http://localhost:8000/v1/usage?user_id=alice&limit=1" -H "Otari-Key: Bearer $MASTER_KEY"
# -> ... "model": "gpt-5-nano", "selection_reason": "router:knn"
```

`scripts/seed_routing_demo.py` does all of the above against a running gateway,
including creating the policy and driving traffic through it, which is the quickest
way to see a routed request:

```bash
python scripts/seed_routing_demo.py --key "$MASTER_KEY" \
  --model openai:gpt-5 --cheap-model openai:gpt-5-nano
```

**A tie is the useful case:** two answers that are both fine is exactly when the
cheaper model should win. Score the cheap candidate low only on the prompts where
only the strong one is good enough. The scores do not have to come from a human
reading answers, an LLM judge works too (`"label_source": "judge"`). To see what each
candidate actually answers, send the prompt to each of them through
`POST /v1/chat/completions`; those calls are budget-checked and land in the usage log
like any other request.

**Memory is per user and per workspace, even for a workspace-wide policy.** The
examples are that user's own prompts, so sharing them across users would let one
caller's traffic steer another's, and sharing them across workspaces would do the
same across tenants for a user who holds keys in both. A policy every caller
resolves therefore warms once per caller in each workspace. `user_id` is required
on both `rank` and `status` because there is no aggregate answer; `workspace_id`
is optional there and means the deployment's default workspace, so a
single-workspace deployment never names one.

**Teaching cannot be undone through the API yet.** There is no route that lists or
deletes recorded examples, so `rank` refuses a score key that no learned policy could
ask about rather than accepting records nothing can match. A `user_id` or `task_id`
typo partitions hard, so it creates a second, invisible pool rather than an error.
Both gaps are tracked on #187.

**If a warm pool still serves the default**, the gateway says why, once per routed
request, at INFO:

```
Router 'knn' on policy 'smart': sparse neighborhood: 3/5 comparable records
  (confidence=0.00) -> policy default
```

That line carries the exact decline reason (cold pool, sparse neighborhood, confidence
below the floor, tools present, an unpriced candidate). The usage row's
`selection_reason` tells you *whether* the router chose (`router:knn`) or declined
(`default`); the log line tells you why.

### Per-request control

| Header | Effect |
| --- | --- |
| `Otari-Router: off` | Serve `default` for this request without consulting the router. Any other unrecognized value is a 400, so a client cannot believe it opted out when it did not. |
| `Otari-Conversation-Id: <id>` | The conversation's identity. With `trace_sticky` granularity (the default) the router decides once per conversation and reuses it, so an agent run does not flip models partway through and prompt caching is not thrown away. Without the header it hashes the conversation's opening turns, which cannot separate two conversations that open identically. |
| `Otari-Router-Task: <name>` | Vote only over the examples filed under this task. A hard split: the partition warms on its own and records from other tasks never influence it. Match it with `task_id` on `rank`. Omit both and everything shares one pool. |

### Tuning

Set through the environment (see the [configuration
reference](configuration.md)), not per policy, because these are properties of the
gateway's routing rather than of one name:

- `OTARI_ROUTER_ALPHA` (default `0.3`), the cost-vs-quality dial. Start low, raise
  it as the routing earns trust.
- `OTARI_ROUTER_SEED_COUNT` (default `20`), examples a pool needs before it routes.
- `OTARI_ROUTER_CONFIDENCE_FLOOR` (default `0.0`), how much neighbor support a pick
  needs.
- `OTARI_ROUTER_K` (default `5`), neighbors per decision.
- `OTARI_ROUTER_GRANULARITY` (`trace_sticky` or `step`).
- `OTARI_ROUTER_EMBEDDING_MODEL`, `OTARI_ROUTER_MAX_RECORDS_PER_USER`.

### Worth knowing before switching it on

- **`explain` cannot rank.** Ranking needs a live request, and `explain`
  dispatches nothing, so it shows the decline path plus the pool it would rank.
- **One extra embedding call** per fresh request, plus a scan of that user's
  examples. Continuations under `trace_sticky` reuse the opening decision and skip
  both.
- **Cost is list price.** Prompt-cache economics are not modeled yet, so a routed
  agent trace can lose the cache the strong model had warm. `trace_sticky` limits
  the damage; the cache-aware cost term is a fast-follow.
- **Stickiness is per process.** The decision lives in the worker that made it, so
  another replica or a restart re-decides. That is safe, just not sticky. See
  [Routing at scale](routing-scaling.md).
- **Changing `OTARI_ROUTER_EMBEDDING_MODEL` invalidates existing examples** rather
  than mixing incomparable vector spaces, so the pool goes cold until it is
  re-taught.

## Guardrails you cannot opt out of

A guardrail listed on a policy runs for every request through that policy,
whether or not the caller asked for one.

```yaml
      guardrails:
        - {profile: prompt-injection, mode: block, on_unavailable: block}
```

`mode` is **required** here. The per-request `guardrails` field defaults to
`monitor`, so an omitted mode on a policy would look like a mandate and behave as
shadow mode.

`on_unavailable` decides what happens when the guardrails service cannot be
reached at all, as opposed to reachable and flagging:

- `block` (default) fails closed. An enforcing check that could not run is not
  silently skipped. The cost is real: a guardrails outage rejects every request
  through this policy, in front of the very fallback chain the policy exists to
  provide. Mandating a `block` guardrail makes that service a hard dependency.
- `monitor` serves the request and records that the check was skipped, trading
  enforcement for availability.

Only input-direction checks are supported. The per-request field accepts
`on: [output]` without enforcing it, so a policy cannot set it: a mandate that
does nothing is worse than no mandate.

## See what a policy will do

A policy's whole job is to make a choice the caller cannot see, so there is a way
to see it. This reads config only: no database, no provider call, nothing billed.

```console
$ otari routing explain fast
fast: 2 candidate(s), selected by default
  1. openai:gpt-5-mini    [default]  dispatches as openai:gpt-5-mini
  2. anthropic:claude-haiku-4-5    [on_failure]  dispatches as anthropic:claude-haiku-4-5
  guardrails (always enforced):
    prompt-injection  mode=block  on_unavailable=block
```

Exercise a condition without waiting for real spend to cross the threshold:

```console
$ otari routing explain thrifty --budget-used-pct 85
thrifty: 2 candidate(s), selected by condition:budget_used_pct
  1. openai:gpt-5-nano    [condition:budget_used_pct]  dispatches as openai:gpt-5-nano
  2. anthropic:claude-haiku-4-5    [on_failure]  dispatches as anthropic:claude-haiku-4-5
```

And see what an API key with a restricted allow-list would actually get:

```console
$ otari routing explain fast --allowed-model 'anthropic:*'
fast: 1 candidate(s), selected by on_failure
  1. anthropic:claude-haiku-4-5    [on_failure]  dispatches as anthropic:claude-haiku-4-5
  x  openai:gpt-5-mini    dropped: is not in allowed_models for this caller
```

That last line is the reason this command exists. A policy is filtered per caller,
so a three-model chain can compile down to one attempt, and a "failover" policy
that is secretly a single attempt is worth finding before an outage does.

## Managing policies at runtime

Everything above can also be done without touching a file or restarting. Policies
created through the API live in the `routing_policies` table, are managed on the
dashboard's **Routing** page, and take effect on the worker that served the write
immediately; other workers and replicas converge within 30 seconds.

```bash
# Create or update. Omit user_id for a policy every caller in the workspace sees,
# and workspace_id for the deployment's default workspace.
curl -X POST http://localhost:8000/v1/routing/policies \
  -H "Otari-Key: Bearer <master-key>" \
  -H "Content-Type: application/json" \
  -d '{
        "name": "fast",
        "spec": {
          "select": [{"default": "openai:gpt-5-mini"}],
          "on_failure": ["anthropic:claude-haiku-4-5"]
        }
      }'

# What is in force, from config.yml and storage alike, in every scope and every
# workspace. Add ?workspace_id=<id> to narrow the stored ones to one workspace;
# config.yml policies are deployment-wide, so they are always listed.
curl http://localhost:8000/v1/routing/policies -H "Otari-Key: Bearer <master-key>"

# Rename one. `rename_from` names the policy to move; `name` is what it becomes.
# The spec goes along on the same write, so a rename cannot land half-applied.
curl -X POST http://localhost:8000/v1/routing/policies \
  -H "Otari-Key: Bearer <master-key>" \
  -H "Content-Type: application/json" \
  -d '{
        "rename_from": "fast",
        "name": "speedy",
        "spec": {"select": [{"default": "openai:gpt-5-mini"}]}
      }'

# Delete one. user_id and workspace_id select the scope; omit user_id for the
# workspace-wide policy and workspace_id for the default workspace.
curl -X DELETE http://localhost:8000/v1/routing/policies/fast \
  -H "Otari-Key: Bearer <master-key>"
```

A stored policy resolves only for requests in its own workspace, which is read
off the API key that authenticated them. Two workspaces can therefore each define
`fast` and get their own; neither sees the other's.

Within a workspace, a stored policy scoped to a user takes precedence over a
`config.yml` policy of the same name, and a workspace-wide stored policy is
refused if one already exists in `config.yml`, because config wins during
resolution and the stored one would be dead config. A `config.yml` policy has no
workspace: it comes from a file the deployment owns, so it is in force in all of
them.

A rename stays inside one scope, since who a policy applies to is the other half
of its key; to move a policy between scopes or workspaces, delete it and create
it again. The
new name goes through the same checks a fresh one does, so a rename cannot walk a
policy into a collision a create would have refused. Two things a rename does not
do: callers naming the old name start getting a 400, and usage already recorded
keeps the old `policy_name`, so per-policy spend before and after a rename does
not add up on its own. A `config.yml` policy has no row to move, so renaming it
means editing the file.

`POST /v1/routing/policies/explain` is the API form of `otari routing explain`,
and it also accepts an unsaved draft `spec`, which is what the dashboard uses to
check a policy before saving it:

```bash
curl -X POST http://localhost:8000/v1/routing/policies/explain \
  -H "Otari-Key: Bearer <master-key>" \
  -H "Content-Type: application/json" \
  -d '{"name": "fast", "allowed_models": ["anthropic:*"], "budget_used_pct": 85}'
```

Every one of these needs the master key, `explain` included: the response
enumerates the policy's targets, which is what a policy exists to keep off the
wire. See the [API reference](api-reference.md#routing-policies).

## Rules and limits

- **Candidate cap: 5** (the selected candidate plus `on_failure`; for any policy
  naming a router, learned or weighted, the whole routed pool plus `on_failure`,
  because the walker cascades through the ordering). A policy over the cap is
  refused rather than silently truncated.
- **No chaining.** A target must name a real `instance:model` or
  `provider:model`, never another policy or alias.
- **Names are checked at startup.** A policy name may not contain `:` or `/`, may
  not collide with a provider instance, and may not collide with an `aliases:`
  entry (both would claim the same caller-facing name, leaving one dead). A name
  that shadows a model declared by a provider instance currently warns and will be
  refused in a future release.
- **`enabled: false`** makes the gateway behave as though no policy were
  configured, so a misrouting policy can be switched off without deleting it.
  Policies are still validated when disabled, so re-enabling cannot surprise you.

## What is billed, and what the caller sees

Pricing, budgets, and usage rows key on the **resolved target**, exactly as they
do for an alias. The response `model` field says the **policy name**, on
non-streaming responses and on every streaming chunk, so the underlying model
stays private and a fallover is invisible to caller code.

A request that fails over writes more than one usage row: one per absorbed
attempt, plus the one for the attempt that served. They share a
`request_group_id`, and the absorbed rows carry `status: "absorbed"` with no cost.
Absorbed rows are excluded from `error_count` and from `request_count`, so a
working fallback chain never reads as an outage and a request that took two
attempts is still counted as one request. Filter the activity log to the
`absorbed` status to see them on their own.

To read a request's whole plan back, ask the usage endpoint for its group:
`GET /v1/usage?request_group_id=<id>` returns every attempt, and the parameter is
repeatable so a page of rows resolves in one call. The dashboard uses this to name
the model that served an attempt that failed, and to render the plan behind a
routed row (see [the Activity page](dashboard.md#observability)).

`GET /v1/models` lists policies. A one-target policy reports its target's price. A
policy that selects per request (a condition or a router) has no single target, so
it reports `pricing_source: "dynamic"` with a null price rather than quoting a rate
that is wrong whenever the policy does its job. A policy's candidates stay in the
listing as themselves; unlike an [alias](models.md#model-aliases), a policy hides
nothing.

A price cannot be set on a policy name. `POST /v1/pricing` refuses it (400, naming
the candidates to price instead) for the same reason it refuses an alias name:
pricing, budgets, and usage key on the model a request resolves to, so a row stored
under the policy name would be written and never read.

Tools Otari runs itself ([built-in tools](tools.md)) are billed per call onto the
row that **settled** the request, not spread across attempts. So a request that
failed over reports every attempt's search work on the row that served, and an
`absorbed` row carries the attempt's tokens and no tool charge. Absorbed rows settle
no reservation, so a charge placed there would show in the activity log and never
reach the budget it consumed. A plan whose every candidate failed still owes for the
searches it ran, and that charge lands on its error row.

## Failure behavior

| Situation | Result |
| --- | --- |
| Selected candidate fails before responding | Next candidate is tried |
| A tool loop already produced its first assistant message | No failover: that state cannot be replayed on a different provider |
| Guardrails service or sandbox unreachable | No failover: the same service serves every candidate |
| All candidates fail | 502, or 504 if the last failure was a timeout |
| Only one candidate survived filtering | Answers exactly as naming that model directly would |
| No candidate is permitted for the caller | 403 naming the policy. The per-candidate reasons go to the activity log, not to the caller: a policy exists partly to keep its targets private |
| A pricier fallback would exceed the remaining budget | The chain stops rather than overshooting the cap |
| A fallback candidate has no pricing, with `require_pricing` on | The chain stops with a 402. The gate applies to every candidate, not only the selected one, so a policy cannot be a way around it |

Failover applies on all three completion endpoints, streaming and not. Streaming
fails over while opening the upstream connection, which is before any bytes reach
the client. Once the stream is open, a mid-body failure propagates:
the client already has part of a response, and swapping models mid-answer is not
something a caller can be expected to handle.

## Where policies do not apply

Policies apply on `/v1/chat/completions`, `/v1/messages`, and `/v1/responses`. A
*static* (one-target) policy also resolves anywhere an alias does, including
`/v1/embeddings` and `/v1/batches`, because it is the same thing.

A policy that selects per request is not a resolvable model name on those other
endpoints: its candidate depends on request state that path cannot see, and
serving the default while calling it the policy would be a lie. Name a concrete
model there.
