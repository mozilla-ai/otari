# Model routing (cost optimization)

Most requests do not need your most expensive model. Otari can learn which ones
do. Point your app at a single strong model as usual; Otari watches which
prompts a cheaper model handles just as well and quietly routes those to the
cheaper model, keeping the strong one for the prompts that actually need it.

The router learns from your own preferences, not a generic benchmark. You show
it a handful of prompts and tell it which model's answer you preferred; from
then on it routes look-alike prompts the same way. It starts safe: until it has
seen enough of your data it does nothing, passing your requested model straight
through.

> Standalone mode only. In platform mode, provider routing and fallback are
> handled by otari.ai, so the local router is not mounted.

## How it works

The router keeps a small per-tenant memory of `(prompt embedding, {model:
quality})` records, one per scored example. For each new request it embeds the
prompt, finds the nearest past prompts, and picks the candidate model with the
best blend of "neighbors said this model was good here" and "this model is
cheaper." One dial, `alpha`, sets how hard it leans on cost.

It never routes blindly:

- **Cold start.** With too little data for a confident decision, it serves your
  requested model unchanged.
- **Low confidence.** When a confidence floor is set (off by default) and the
  neighbors don't clearly support the cheaper pick, it leads with your requested
  (strong) model.
- **Tools.** Requests carrying tools are left on the requested model.
- **Conversations.** Once a multi-turn conversation has been routed, later turns
  stick to the same model (no mid-conversation switching). Send a stable
  `Otari-Conversation-Id` header so the router can tell your conversations apart
  reliably (see [below](#keep-a-conversation-on-one-model)).

"Tenant" is the identity behind the request: the user the API key belongs to.
Memory is scoped per tenant, so collect preferences with the **same key** you
serve traffic with.

## Get started

### 1. Turn it on

Pick the models the router may choose among (your strong default plus one or
more cheaper options) and enable the `knn` backend:

```bash
export OTARI_ROUTER_BACKEND=knn
export OTARI_ROUTER_CANDIDATES="openai:gpt-4o,openai:gpt-3.5-turbo"
# Optional, lower the warm-up bar while you try it out (default 20):
export OTARI_ROUTER_SEED_COUNT=8
```

Make sure those models' providers are configured, and (for cost-aware routing)
that they have [pricing](configuration.md#pricing) set. Without pricing the
router still works, but the cost dial has nothing to act on.

### 2. Check status

```bash
curl http://localhost:8000/v1/router/status -H "Otari-Key: Bearer $KEY"
# -> {"backend":"knn","seed_count":20,"default_pool":{"records":0,"warm":false},"tasks":[]}
```

Routing memory is a set of independent pools, so status reports each one rather
than a single number. `default_pool` is what a request with no `Otari-Router-Task`
header routes over (every record you have); `tasks` lists each task partition
(see [Separate routing by use case](#separate-routing-by-use-case)). A pool with
`warm: false` is still passing through, which is expected before you have
collected preferences for it.

### 3. Collect a few preferences

Send a prompt to your candidate models side by side:

```bash
curl -X POST http://localhost:8000/v1/router/preferences/compare \
  -H "Otari-Key: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"prompt":"what is 18 + 24?","models":["openai:gpt-4o","openai:gpt-3.5-turbo"]}'
# -> {"prompt":"...","responses":[{"model":"openai:gpt-4o","content":"42"}, ...]}
```

Look at the answers and score each one from 0 (bad) to 1 (great):

```bash
curl -X POST http://localhost:8000/v1/router/preferences/rank \
  -H "Otari-Key: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"prompt":"what is 18 + 24?","scores":{"openai:gpt-3.5-turbo":1.0,"openai:gpt-4o":1.0}}'
# -> {"recorded":1,"task_id":null,"warm":false}
```

Here the cheap model answered fine, so it scores as high as the strong one: a
vote to route arithmetic-like prompts to it on cost. For prompts where only the
strong model is good enough, score the cheap model low. Each submission writes
one memory record (the prompt with each model's score), so the kNN votes over
distinct prompts; repeat until `/status` reports `warm: true`.

Tip: the scores do not have to come from a human eyeballing answers. You can
score the `compare` responses with an LLM judge of your own and submit them the
same way (`"label_source":"judge"`).

### 4. Serve traffic

Nothing changes in your application. Keep requesting your strong model:

```json
{ "model": "openai:gpt-4o", "messages": [{"role":"user","content":"what is 9 + 6?"}] }
```

For prompts that look like your "cheap is fine" examples, the response now comes
back from the cheaper model (the `model` field in the response reflects what
actually ran), and the usage log and cost are attributed to it. For everything
else, you get the strong model just as before.

### Opt a request in or out

Routing is a server setting, but a client can override it for a single request
with the `Otari-Router` header:

- `Otari-Router: off` serves the request from the requested model and skips
  routing entirely (handy for an eval, a golden request, or a debugging call).
- `Otari-Router: on`, or omitting the header, uses the server default, so the
  request is routed whenever a backend is enabled.

The header only affects standalone mode and cannot force routing on when no
backend is configured (`router_backend: none`). An unrecognized value is
rejected with a 400.

### Keep a conversation on one model

With `trace_sticky` granularity (the default), the router decides once per
conversation and reuses that decision for later turns, so a multi-turn agent run
does not flip models partway through. To do that it needs to recognize that two
requests belong to the same conversation. Pass a stable id you control:

```
Otari-Conversation-Id: 4f9c2b10-...   # same value on every turn of one conversation
```

If you omit it, the router falls back to hashing the conversation's opening
messages. That works for distinct openers, but two conversations that begin with
the same system prompt and first user turn (common with agent scaffolding) will
be treated as one and share a routing decision. Sending the header avoids that.
The id is namespaced per tenant, so the same value from two keys never collides.
See [Routing at scale](routing-scaling.md) for why this matters once you run
more than one replica.

### Separate routing by use case

Routing memory is always scoped per tenant (the user behind the API key), so two
users never see each other's records. That is usually enough, and `task_id` is
optional. If you want to split a *single* tenant's memory across unrelated use
cases, for example a support bot and a code reviewer that should not learn from
each other, optionally tag preferences with a `task_id` and route requests
against the matching partition.

When you collect preferences, add an optional `task_id` on the rank call:

```bash
curl .../v1/router/preferences/rank -H "Otari-Key: Bearer $KEY" \
  -d '{"prompt": "...", "scores": {...}, "task_id": "support-bot"}'
```

When you serve traffic, select the partition with a header:

```
Otari-Router-Task: support-bot
```

Partitions are a **hard** split: a request carrying a task votes only over that
task's records, and the partition warms independently, it stays in pass-through
until the task alone crosses the seed count, and records from other tasks never
influence it. Omitting `task_id` and the header keeps everything in one shared
pool, which is the default. `GET /v1/router/status` lists the shared default
pool and every task partition with each one's progress.

## Tuning

- `OTARI_ROUTER_ALPHA` (default `0.3`) is the cost-vs-quality dial. Higher saves
  more money and routes more aggressively to cheaper models; lower stays closer
  to the strong default. Start low and raise it as you trust the routing.
- `OTARI_ROUTER_CONFIDENCE_FLOOR` (default `0.0`) raises the bar for trusting a
  cheap pick: below it, the request leads with your requested model.
- `OTARI_ROUTER_SEED_COUNT` (default `20`) is how many records a tenant needs
  before routing leaves pass-through.

See the [configuration reference](configuration.md#model-routing) for the full
list.

## Limits and good to know

- **Per-tenant, per-key.** Memory is scoped to the user behind the API key. If
  you serve many end users (passing a `user` per request), each accumulates its
  own memory and warms independently. Collect preferences under the same
  identity that serves the traffic you want to optimize.
- **Embedding cost/latency.** Routing adds one embedding call per fresh request
  (the default is `openai:text-embedding-3-small`) plus a load of the tenant's
  vectors. Conversation continuations reuse the opening decision and skip it.
- **Preference calls are not budgeted.** `/preferences/compare` calls the
  candidate providers directly, outside the budget reservation and usage logging
  that the chat path enforces. Treat it as an operator/onboarding tool and gate
  who holds keys that can reach it.
- **No standalone fallback.** If the router sends a request to a cheaper model
  and that model errors, the request fails; standalone mode does not cascade to
  another model. Keep your candidate pool to models you trust to be up.
- **Scale.** The store is a linear scan over a tenant's vectors, capped by
  `OTARI_ROUTER_MAX_VECTORS_PER_TENANT` (default 5000). That is fine into the low
  thousands per tenant; larger deployments will want an ANN index (a documented
  next step, not yet built).
- **Conversation stickiness is per-process.** The decision a conversation sticks
  to lives in the worker that first routed it. With more than one replica, or
  across a restart, a later turn can land elsewhere and re-route. It stays
  correct (it just re-decides), but the stickiness is best-effort until the
  shared-store design in [Routing at scale](routing-scaling.md) lands.
