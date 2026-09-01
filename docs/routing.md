# Routing policies

A routing policy is a caller-facing model name that resolves to one or more real
models. Use a policy for failover, conditional selection, traffic splitting,
learned selection, or guardrails the caller cannot remove. Use an
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

## Let a router choose (learned routing)

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

Teach the router through `POST /v1/routing/preferences/rank`. Each example
contains a prompt and a score from 0 to 1 for each candidate. Read pool status
through `GET /v1/routing/status`. `scripts/seed_routing_demo.py` provides a
runnable example.

Learned memory is scoped by user and workspace. Optional task IDs create separate
pools. It is not learned automatically from live traffic.

### Per-request control

| Header | Effect |
| --- | --- |
| `Otari-Router: off` | Skip learned or weighted selection and use the policy default. |
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
`POST /v1/routing/policies/explain`; it can also validate an unsaved draft.

## Managing policies at runtime

Config-file policies apply to every workspace. Standalone operators can also
manage stored policies through the Routing page or `/v1/routing/policies`.
Stored policies belong to one workspace and can optionally be scoped to one
user.

An organization's owners and admins manage their own workspaces' policies and
aliases through `/v1/organizations/me/routing-policies` and
`/v1/organizations/me/aliases`, which the Routing page uses for a caller who
does not operate the deployment. Those routes require the workspace named, must
name a workspace of the caller's own organization, accept no user scope, and
refuse a target the organization holds no provider access for. A member of the
organization reads the same two lists and writes neither.

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
All attempts share a `request_group_id`. Absorbed rows have no settled model
cost and do not increase request or error totals; the final row represents the
caller-visible request.

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

If every candidate fails, Otari returns a gateway error based on the final
failure. If caller restrictions remove every candidate, Otari returns 403
without revealing the hidden targets.
