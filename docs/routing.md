# Routing policies

A routing policy is a caller-facing model name that resolves to one or more real
models. Use a policy for failover, conditional selection, traffic splitting,
spill-over past a rate limit, learned selection, or guardrails the caller cannot
remove. Use an
[alias](models.md#model-aliases) when one name always maps to one target.

Policies are a standalone feature. Hybrid gateways receive their attempt plan
from the connected control plane.

## Failover

```yaml
routing:
  policies:
    fast:
      select:
        - default: openai:gpt-5-mini
      on_failure:
        - anthropic:claude-haiku-4-5
```

Callers send `model: fast`. If the first provider fails before any response is
sent, Otari tries the fallback. The response keeps the policy name; Activity and
usage record the models that were attempted.

## Selection and failure are separate

`select` chooses where a request starts. `on_failure` lists what to try after a
retryable failure.

```yaml
routing:
  policies:
    thrifty:
      select:
        - when:
            budget_used_pct: {gte: 80}
          target: openai:gpt-5-nano
        - default: openai:gpt-5-mini
      on_failure:
        - anthropic:claude-haiku-4-5
```

Selection entries are evaluated in order. The default must be last. Supported
conditions are:

- `budget_used_pct`
- `budget_remaining_usd`
- `user_id`
- `key_id`

All conditions in one `when` block must match. Numeric comparisons use exactly
one of `gt`, `gte`, `lt`, or `lte`. A budget condition does not match a
caller with no finite budget.

## Load balance across providers (weighted routing)

The weighted router chooses independently for each request and normalizes the
configured weights:

```yaml
routing:
  policies:
    balanced:
      select:
        - router: weighted
          candidates:
            - openai:gpt-5
            - anthropic:claude-sonnet-4-6
          weights:
            openai:gpt-5: 7
            anthropic:claude-sonnet-4-6: 3
        - default: openai:gpt-5
      on_failure:
        - gemini:gemini-2.5-flash
```

`7:3` and `70:30` describe the same split. A candidate omitted from
`weights` receives no initial traffic but remains available in the failure
ordering. At least one weight must be positive.

Each request is a fresh draw. Weighted routing has no conversation stickiness or
health-adjusted weights. If the selected provider fails before responding, Otari
continues through the remaining weighted pool, then `on_failure`.

Caller allow-lists filter candidates before weights are normalized. Use
`otari routing explain` to see the effective split for a restricted caller.

## Spill over when a model is full (priority routing)

The priority router keeps its candidates in the order written. Each request goes
to the first one that has room under its
[`per: model` rate limits](configuration.md#rate-limit-rules):

```yaml
rate_limits:
  - name: flash-cap
    per: model
    models: ["vertex:gemini-2.5-flash"]
    rpm: 100

routing:
  policies:
    summarize:
      select:
        - router: priority
          candidates:
            - vertex:gemini-2.5-flash   # takes every request up to 100 a minute
            - together:llama-3.3-70b    # takes the rest
        - default: together:llama-3.3-70b
      on_failure:
        - mistral:mistral-small
```

A full candidate is skipped without being called, and Activity shows it in the
request's routing plan as skipped, naming the limit that was full. Prometheus
counts each one in `gateway_rate_limit_model_full{rule, model}`. The limit lives on the model,
not on the policy, so it is skipped the same way in `on_failure`, in a weighted
pool, and in any other policy that names it; with Redis as `rate_limit_store`,
the count holds across replicas. A candidate with room that fails before
responding falls through to the next one, as in any policy. The per-model limits
refuse a request with a 429 only when every candidate is full, naming the limit
that frees up soonest (and its `Retry-After`) but not the model, since a
policy's targets are not the caller's to see. A per-key, per-user or deployment
rule is checked before routing, so it can still refuse a request while every
candidate has room.

The policy says which is which: `candidates` handles "full", `on_failure`
handles "broke". `Otari-Router: off` skips the order and starts from the
default.

In the dashboard, create one from Routing with "Move to the next model when one
hits its rate limit", order the models with the arrows, and add the limit under
Settings, Rate limit rules, counted for "Each model".


The `knn` router uses scored examples to rank candidates for each user's
traffic:

```yaml
routing:
  policies:
    smart:
      select:
        - router: knn
          candidates:
            - openai:gpt-5-nano
            - openai:gpt-5
        - default: openai:gpt-5
      on_failure:
        - anthropic:claude-haiku-4-5
```

It embeds the request, finds similar scored prompts, and balances predicted
quality against configured model cost. Every candidate therefore needs pricing.
Until a user's pool is warm, or when the router cannot decide confidently, the
default target serves.

Teach the router through `POST /api/v1/routing/preferences/rank`. Each example
contains a prompt and a score from 0 to 1 for each candidate. Read pool status
through `GET /api/v1/routing/status`. `scripts/seed_routing_demo.py` provides a
runnable example.

Learned memory is scoped by user and workspace. Optional task IDs create separate
pools. It is not learned automatically from live traffic.

### Per-request control

| Header | Effect |
| --- | --- |
| `Otari-Router: off` | Skip learned, weighted or priority selection and use the policy default. |
| `Otari-Conversation-Id` | Reuse a learned decision for a conversation when granularity is `trace_sticky`. |
| `Otari-Router-Task` | Use examples from one task partition. |

The default learned-routing settings are `k=5`, a 20-example warm-up, and
`trace_sticky` granularity. They are deployment settings named
`router_k`, `router_seed_count`, `router_alpha`,
`router_confidence_floor`, `router_embedding_model`, `router_granularity`, and
`router_max_records_per_user`.

Trace stickiness is process-local. A restart or request routed to another replica
may choose again. The result is still valid, but prompt-cache locality is not
guaranteed across replicas.

## Mandatory guardrails

A policy can run guardrails even when the caller did not request them:

```yaml
routing:
  policies:
    safe:
      select:
        - default: openai:gpt-5-mini
      guardrails:
        - profile: prompt-injection
          mode: block
          on_unavailable: block
```

`mode` is required. `on_unavailable: block` fails closed when the guardrail
service cannot run; `monitor` lets the request continue and records the skipped
check. Only input checks can be mandated by a policy.

Organization-mandated and request-provided guardrails compose with policy
guardrails. See [Guardrails](guardrails.md).

## Explain a policy

Inspect a policy without calling a provider:

```bash
otari routing explain fast
otari routing explain thrifty --budget-used-pct 85
otari routing explain balanced --allowed-model "anthropic:*"
```

The command shows ordered candidates, filtered candidates and their reasons,
effective weighted shares, and mandatory guardrails. The API equivalent is
`POST /api/v1/routing/policies/explain`; it can also validate an unsaved draft.

## Agent model recommendations

A policy decides the model of a request that reaches Otari. A coding agent that
talks to its provider directly can still ask Otari which model a new subagent
should run on, through `POST /api/v1/routing/recommend`. A standalone
deployment puts one choice question to the decision model it configured and
recommends the candidate it picks; on otari.ai a managed recommender answers
instead. Either way the recommendation is billed to the caller, and nothing
else is dispatched. See
[Use with Claude Code](use-with-claude-code.md#let-otari-choose-a-subagents-model).

## Managing policies at runtime

Config-file policies apply to every workspace. Standalone operators can also
manage stored policies through the Routing page or `/api/v1/routing/policies`.
Stored policies belong to one workspace and can optionally be scoped to one
user.

An organization's owners and admins manage their own workspaces' policies and
aliases through `/api/v1/organizations/me/routing-policies` and
`/api/v1/organizations/me/aliases`, which the Routing page uses for a caller who
does not operate the deployment. Those routes require the workspace named, must
name a workspace of the caller's own organization, accept no user scope, and
refuse a target the organization holds no provider access for. Both reads take a
server-capped `limit`, which bounds the response rather than paginating it. A
member of the organization reads the same two lists and writes neither.

Those routes are workspace-wide entries only, on the reads as much as the
writes. A stored policy or alias can also be scoped to one user, and `user_id`
names an API-key user rather than a dashboard identity, so it is neither a
tenant's to set nor a tenant's to interpret. Managing a user-scoped entry stays
on the deployment-wide routes.

The authenticating API key determines which workspace resolves a policy.
User-scoped entries take precedence over wider entries. A config-file policy
cannot be changed through the API.

Changes apply immediately on the worker that accepts the write. Other workers
refresh stored routing configuration within 30 seconds.

Use the generated OpenAPI document for create, rename, delete, list, and explain
request schemas.

## Rules and limits

- A plan may contain at most five candidates, including router pools and
  `on_failure`.
- Targets must be concrete provider or instance selectors. Policies and aliases
  cannot chain.
- Policy names cannot contain `:` or `/` or collide with an alias or provider
  instance.
- `routing.enabled: false` disables policy resolution but still validates the
  configured policies.
- Caller model allow-lists apply to every candidate.
- Dynamic policies apply to Chat Completions, Messages, and Responses. A static
  one-target policy can resolve on other model-taking endpoints.

## What is billed and what the caller sees

Pricing, budgets, and usage use the resolved model. Completion responses use the
policy name.

Each failed attempt before a successful fallback gets an `absorbed` usage row.
All attempts share a `request_group_id`, which is the request's `Otari-Request-ID`.
Absorbed rows have no settled model cost and do not increase request or error
totals; the final row represents the caller-visible request.

Every successful Chat Completions, Messages, and Responses response, routed or
not, also says how it was served, so a caller's own metrics can see a fallback
without reading the usage log:

| Header | Value |
| --- | --- |
| `Otari-Provider` | The provider instance that served the request, by its configured name (`openai`, `azure-eu`). Never its base URL or credentials. |
| `Otari-Attempt-Count` | How many candidates the request was sent to, including the one that served. A candidate skipped without being called, such as a model a rate limit had no room on, is not counted. |
| `Otari-Fallback` | `true` when a candidate other than the plan's first served the request, whether the first failed or was skipped; otherwise `false`. |

A request naming a model or an alias reports `1` and `false`. Streaming responses
carry the same values: failover happens before the stream opens, so the serving
candidate and the attempt count are already known when the headers are sent. A
failure later in the stream is reported in the stream and changes neither. The
request's cost is `usage.cost_usd` in the body, as described in the
[API reference](api-reference.md#request-id-and-inline-cost). These headers are
sent by a standalone gateway.

Built-in tool charges settle on the final row. Candidate price and remaining
budget are checked before each attempt, so a fallback cannot silently bypass
pricing or spend limits.

## Failure behavior

Failover occurs only before response bytes reach the client. A streaming failure
after the stream begins is returned to the client because switching models
mid-answer would corrupt the response.

A tool loop that has already produced assistant state cannot be replayed on
another provider. Shared service failures, such as an unavailable mandatory
guardrail or sandbox, also do not improve by changing candidates.

If every candidate of a plan with more than one candidate fails, Otari returns a gateway error in its own words, never a provider's message. A plan with one candidate answers as naming that model directly would. When every candidate rejected the request as invalid (400), Otari returns 400 with the code `all_candidates_rejected`, because the request is at fault. When every candidate found the prompt too long, the code is `context_length_exceeded`. When no candidate knew the model (404), Otari returns 502, because Otari chose the candidates. If caller restrictions remove every candidate, Otari returns 403 without revealing the hidden targets.
