# Routing policies

A **routing policy** is a model name you define. Callers send it in the `model`
field like any other model, and the policy decides which real model serves the
request, what to try if that model fails, and which guardrails always run.

An alias is the simplest possible policy: one name, one target. That is why
[`aliases:`](models.md#model-aliases) keeps working and is documented as the
shorthand. Reach for a policy when you want more than one target, a condition, or
an enforced guardrail.

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

If `openai:gpt-5-mini` returns a retryable failure, Anthropic serves the request
instead and the caller never sees the difference: the response `model` says
`fast`, and one usage row is written for the model that actually served.

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
silently dead rule.

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
  case for a caller with no budget, an unlimited budget, or the master key. The
  policy falls through to `default`. It does not raise: "no budget configured"
  must not turn into an error on every request.
- **A `budget_used_pct` threshold at 100 or above is refused at startup.** The
  budget gate rejects a request before selection happens, so such a rule could
  never fire. Tiering down keeps a caller *under* a cap; it is not a way to keep
  serving past one.

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

## Rules and limits

- **Candidate cap: 5** (the selected candidate plus `on_failure`). A policy over
  the cap is refused rather than silently truncated.
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

`GET /v1/models` lists policies. A one-target policy reports its target's price. A
policy that selects per request (a condition or a router) has no single target, so
it reports `pricing_source: "dynamic"` with a null price rather than quoting a rate
that is wrong whenever the policy does its job.

## Failure behavior

| Situation | Result |
| --- | --- |
| Selected candidate fails retryably | Next candidate is tried |
| Selected candidate returns 400 or 422 | No failover: every provider would reject it |
| Provider returns 401 or 403 | No failover. You own these credentials, so this is a misconfiguration to fix, not a reason to move traffic and spend to another provider and hide it |
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
