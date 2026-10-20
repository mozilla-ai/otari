# Budgets

A budget caps what can be spent over a period, and Otari refuses requests that
would pass it. Budgets can cap spend in USD, tokens and request count, and they
apply at the organization or workspace level, narrowed to a member, an API key
or a provider when you need finer control.

In the dashboard a budget comes in two parts. The **budget** defines an amount
and a period. A **spend ceiling** applies that budget to something: the
organization, a workspace, or an API key, optionally for one provider only. A
budget on its own blocks nothing until a ceiling uses it.

## Scope

Combine the level with what the ceiling applies to:

- **Organization, overall**: the organization's total spend.
- **Organization, one member**: one person's spend across every workspace.
- **Workspace, all providers**: everything one workspace spends.
- **Workspace, one member**: one contributor's spend in one workspace.

Each ceiling is enforced independently. If a workspace has a $500 monthly cap
and one member has a $100 monthly cap inside it, both are checked on every
request, and the first one to run out refuses it.

## Periods

| Period | Resets |
| --- | --- |
| Daily | Every day at midnight UTC |
| Weekly | Every Monday at midnight UTC |
| Monthly | On the first of the month at midnight UTC |

Resets are automatic. Through the API, a budget can also use a rolling duration
instead of a calendar boundary; see [How enforcement works](#how-enforcement-works).

## Create a budget and a spend ceiling

1. Open **Organization** and select **Spend & Budgets**.
2. Click **Add budget**, then set an optional **Name**, the **Amount** in USD
   (blank for no limit) and the **Period**. Click **Add budget**.
3. Click **Add ceiling**, then choose what it caps (**Capping**: the
   organization or a workspace), optionally a **Provider instance** to cap one
   provider only, and the **Budget** to use. Click **Add ceiling**.

Requests matching the ceiling count toward its budget. When the budget is used
up, new matching requests are refused until the next reset. Requests already
running finish normally.

## Check a budget's status

**Spend & Budgets** shows each budget's use: green below 80%, yellow from 80% to
100%, and red once it is used up and matching requests are refused.

A refused request answers `403` with `Otari-Error-Code: budget_exceeded`; see
[Refusals and `Retry-After`](#refusals-and-retry-after).

## How enforcement works

Otari enforces budgets before dispatch and reconciles actual cost afterwards.
Requests must pass every applicable limit.

Two budget forms exist:

- A per-user budget limits each attached user independently.
- A scoped budget limits an organization, workspace, membership, or API key and
  can optionally narrow the limit to one provider.

A routing fallback to another provider must also pass that provider's budgets, as [Routing](routing.md#what-is-billed-and-what-the-caller-sees) describes.

A budget caps up to three things over its period, each set independently and
each unlimited when left unset: spend in USD (`max_budget`), total tokens
(`token_limit`), and requests (`request_limit`). A request must have room on
every axis the budget caps. Spend and tokens are held at an upper bound before
dispatch and reconciled to the measured figures afterwards; only the unused part
of an over-estimate is released, and what the request measurably used stays
charged. A request counts as one request when it is admitted. A model priced at
zero spends no dollars, and still spends tokens and one request.
A request whose dollar hold is larger than what remains is refused even when
nothing has been spent yet, and the refusal says so with both amounts:
`User 'alice' has $0.0015 of budget remaining, less than this request's
estimated cost of $0.01025`. On chat completions, messages and responses the
estimate prices the request's output token limit (`max_tokens`, or
`max_output_tokens` on responses; `budget_estimate_default_output_tokens` when it
sets none), and where the model charges for output the refusal names that
figure, so a lower limit may let the request fit. A
budget already at its cap is refused as `has exceeded budget limit`.
Endpoints that hold no token estimate (embeddings, rerank, and the other
pass-through routes) are refused once a token cap is exhausted rather than
reserving headroom for themselves, so a token cap can be passed by the requests
already in flight when it runs out.

A budget can also limit each user on it per minute: `rpm_limit` requests and
`tpm_limit` tokens. These are counted in `rate_limit_store`, so with Redis they
hold across replicas, and tokens are counted on what each request used, as
LiteLLM counts them: a request is admitted while the user's minute is under the
limit. They apply to a user's own budget on chat completions, messages,
responses and search, not to a scoped ceiling, and a refusal is a 429 naming the
rule `budget`.

Scoped budgets can use a rolling duration or a UTC calendar boundary. A key with
`exclude_from_budget`, or a deployment with `budget_strategy: disabled`,
bypasses enforcement.

Imported usage is retrospective and never counts toward a budget. Batch cost is
also settled after submission, so operators should not treat those paths as a
hard real-time cap. Batch settles dollars alone: its results arrive outside the
reservation that gated the submission, so a batch counts as the one request that
created it and contributes no tokens to a token cap, however many prompts it
carried. The same holds for the vision side-call a request makes to describe an
attachment. Cap batch-heavy workloads in dollars rather than in tokens. See
[Importing external usage](external-usage.md).

### Reservations are taken up front

The hold is the request's worst case, taken before dispatch: its prompt at
roughly four characters per token, plus its whole output token limit. A request
with `max_tokens: 8192` therefore holds the price of 8192 output tokens however
few the model ends up writing, and a user close to the cap is refused once that
hold no longer fits, even when the answer would have cost a fraction of it.
The trade is deliberate: a request is admitted only when its estimated worst
case fits, at the price of refusing some requests near the cap that would have
fit. The prompt side is an estimate, and settlement records what the request
actually cost, so a request can still settle a little past its hold (and its
budget past the cap) when the prompt tokenizes heavier than the estimate
assumes. A client that sends a realistic output limit, rather than the model's
maximum, keeps more of its allowance usable near the cap.

Holds are also what bound concurrent requests from one user or end user. Each
admitted request keeps its hold until it settles, and a new request is admitted
only while recorded spend, every hold still in flight and its own hold together
fit under the cap, checked in one conditional update so two requests cannot both
take the last of it. Ten parallel requests near the cap are therefore admitted
only as far as their combined worst case fits, and the rest are refused. A hold a crashed request leaves behind is returned after
`budget_reservation_ttl_sec` (15 minutes by default). Scoped ceilings hold the
same way, so a key's ceiling bounds all of its end users' in-flight requests
together.

### Refusals and `Retry-After`

A refusal is a 403 with `Otari-Error-Code: budget_exceeded`, and
`Otari-Budget-Scope` names the budget that refused: `user`, or the ceiling's
scope. When that budget resets, the refusal also carries `Retry-After`, the
whole seconds (rounded up) until its current window ends: the next UTC
midnight, ISO week or month for a calendar budget, and the end of the user's or
ceiling's own period for a rolling one, which only the gateway knows. Where
several budgets apply, it is the window of the one that refused. A budget with
no period never resets and sends no `Retry-After`, and neither does a refusal of
a hold larger than the whole cap, since a fresh window would refuse it too.

## Workspace-scoped spend

Usage is attributed to the workspace bound to the authenticating API key.
Organization and workspace usage views then apply the signed-in identity's
membership. Deployment operators can read the operator usage API, which covers
every workspace of the operator's active organization (the default organization
for the master key).

Routing fallback and built-in tools can create several internal attempts, but a
successful request remains one caller-visible request. The activity log records
the attempt group and the model that served it.
