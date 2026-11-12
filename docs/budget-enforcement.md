# Budget enforcement

How Otari enforces a budget: the forms, the limits, how holds are taken and
settled, and what a refusal carries. 

[Budgets](budgets.md) explains scopes and
periods, and [Set a budget and ceiling](set-a-budget-and-ceiling.md) sets one up
in the dashboard.

Otari checks budgets before dispatch and reconciles actual cost afterwards.

## Budget forms

- A **per-user budget** limits each attached user independently.
- A **scoped budget** (a spend ceiling) limits an organization, workspace,
  membership or API key, optionally for one provider only.

A request must fit every budget that applies to it. A routing fallback to
another provider must also pass that provider's budgets; see
[Routing](routing.md#what-is-billed-and-what-the-caller-sees).

## What a budget limits

Each field is optional and unlimited when unset:

| Field | Limits | Over |
| --- | --- | --- |
| `max_budget` | Spend in USD | The budget's period |
| `token_limit` | Total tokens | The budget's period |
| `request_limit` | Requests | The budget's period |
| `rpm_limit` | Requests, per user | One minute |
| `tpm_limit` | Tokens, per user | One minute |

A request must have room on every axis the budget caps. It counts as one request
when admitted. A model priced at zero spends no dollars, so it is admitted when
no dollars are left, but it still spends tokens and one request, and a priced
tool it calls must fit the dollar cap.

## Holds and settlement

Spend and tokens are held at an upper bound before dispatch and settled to the
measured figures afterwards. Only the unused part of an over-estimate is
released.

The hold is the request's worst case: its prompt at roughly four characters per
token, plus its whole output token limit (`max_tokens`, or `max_output_tokens` on
Responses; a deployment default of 1024 tokens when the request sets none). A
request with `max_tokens: 8192` holds the price of 8192 output tokens however few
the model writes, so a user near the cap is refused once that hold no longer
fits. A client that sends a realistic output limit keeps more of its allowance
usable near the cap.

The prompt side is an estimate, so a request can settle a little past its hold
(and its budget past the cap) when the prompt tokenizes heavier than assumed.

Holds are priced at the rules of the model's price the request meets before the
call: estimated context, requested tier, image size, video resolution and start
time. Settlement prices the call at the rules it actually met, including the
tier the provider reports, so a request served at a dearer tier can settle above
its hold; see [Overrides and multipliers](configuration.md#overrides-and-multipliers).

### Endpoints without a token estimate

Embeddings, rerank and the other pass-through routes hold no tokens. They are
refused once a token cap is exhausted, so requests already in flight can pass
it. In dollars each holds what it can measure before the call:

| Endpoint | Dollar hold |
| --- | --- |
| Image generation | The images asked for, at the `size` asked for |
| Speech | The characters of its input |
| Moderations | One request |
| Transcription | Nothing (the audio's length is not read) |

See [Each endpoint in its unit](configuration.md#each-endpoint-in-its-unit).

### Concurrent requests

Each admitted request keeps its hold until it settles. A new request is admitted
only while recorded spend, every hold in flight and its own hold together fit
under the cap, checked in one conditional update so two requests cannot both
take the last of it. Scoped ceilings hold the same way, so a key's ceiling bounds
all of its end users' in-flight requests together.

A hold left by a crashed request is returned after `budget_reservation_ttl_sec`
(15 minutes by default); see [Budget enforcement](configuration.md#budget-enforcement).

## Per-minute limits

`rpm_limit` and `tpm_limit` count what each request used: a request is admitted
while the user's current minute is under the limit. They apply to a user's own
budget on chat completions, messages, responses and search, not to a scoped
ceiling. A refusal is a 429 naming the rule `budget`. A self-hosted deployment
chooses where they are counted; see
[Budget enforcement](configuration.md#budget-enforcement).

## Periods and exemptions

Scoped budgets use either a UTC calendar boundary (daily, weekly from Monday,
monthly from the first) or a rolling duration. A key marked **Exempt from
budget** (`exclude_from_budget`) logs its cost without enforcement. A self-hosted
deployment can turn enforcement off entirely; see
[Budget enforcement](configuration.md#budget-enforcement).

## Imported usage, batches and video

These are settled after the fact, so do not treat them as a hard real-time cap:

- **Imported usage** never counts toward a budget; see
  [Import usage from outside Otari](external-usage.md).
- **Batches** settle dollars only. A batch counts as the one request that created
  it and adds no tokens to a token cap, however many prompts it carried. Cap
  batch-heavy workloads in dollars, not tokens.
- **Attachment descriptions** (the vision side-call for an attachment) settle
  dollars only.
- **Video jobs** reserve the video's length at the model's per-second price when
  the job starts, and settle in dollars once the provider reports the job
  complete.

## Refusals

A refusal is a 403 with `Otari-Error-Code: budget_exceeded`.
`Otari-Budget-Scope` names the budget that refused: `user`, or the ceiling's
scope. The `detail` distinguishes two cases:

- A spent budget: `... has exceeded budget limit`.
- A hold larger than what remains, even if nothing is spent yet:
  `User 'alice' has $0.0015 of budget remaining, less than this request's
  estimated cost of $0.01025`. Where the model charges for output, the detail
  names the output limit it priced, so a lower limit may let the request fit.

On a service key, a refusal by the end user's own budget names the end user as
the request did (`End user 'alice'`), not by Otari's internal id, and carries it
as `end_user` in the body; see [End users](end-users.md).

### `Retry-After`

When the refusing budget resets, the refusal carries `Retry-After`: the whole
seconds, rounded up, until its current window ends (the next UTC midnight, ISO
week or month for a calendar budget, or the end of the rolling period). Where
several budgets apply, it is the window of the one that refused.

A budget with no period sends no `Retry-After`, and neither does a refusal of a
hold larger than the whole cap, since a fresh window would refuse it too.

See [Error codes](api-reference.md#error-codes).
