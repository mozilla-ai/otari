---
icon: square-dollar
---

# Budgets

A budget caps how much can be spent in a period and blocks requests when the limit is reached. You can set budgets at the organization or workspace level, and narrow them to specific members or providers when you need finer control.

A budget is applied with a **spend ceiling**, which names what it caps, or by putting a service key's [end users](end-users.md) on it. A budget nothing uses blocks nothing.

## Scope: where the budget applies

Every ceiling has two scope dimensions.

**Level:**

* Organization caps spend across the whole organization.
* Workspace caps spend within a single workspace.

**Applies to:**

* Overall is one total cap for everything at that level.
* A member caps one person's spend within a workspace. It is set on **Members & roles**, beside the person.
* A provider caps spend on one provider only.

**Combine the two to get a specific scope.**

* **Organization > Overall**: total organization spend cap.
* **Organization > A specific provider**: cap spend on one provider across all workspaces.
* **Workspace > All providers**: cap spend on every provider used by that workspace.
* **Workspace > A specific member**: cap one contributor's spend within one workspace.

Each budget is enforced independently. If a workspace has a $500/month overall cap and one member has a $100/month cap inside that workspace, both limits are checked on that member's requests. The first budget to run out blocks the request. Other members' requests are checked against the workspace cap only. On otari.ai a request must also fit the organization's [balance](billing.md#budgets-are-a-different-control).

To cap a single API key, set a budget on the key itself; see [Cap a key's spend](api-keys.md#cap-a-keys-spend).

## Period: how often it resets

| Period | Resets |
| --- | --- |
| **Daily** | Every day at UTC midnight. |
| **Weekly** | Every Monday at UTC midnight. |
| **Monthly** | On the first of the month at UTC midnight. |

All resets are automatic. No cleanup is required.

## Enforcement

When a budget is exhausted, Otari blocks matching requests until the next period starts.

Existing in-flight requests complete normally. New requests at or above the cap fail until the next reset.

A request is also blocked when its estimated cost no longer fits in what is left, even before the budget is fully spent. The estimate includes the request's whole output token limit (`max_tokens`), so a lower limit can let a request fit.

## Status indicators

**Spend & budgets** shows each budget's spend against its limit as a bar:

| Spend | Bar |
| --- | --- |
| **Under 80%** | Healthy. |
| **80–100%** | Approaching the limit: the part past 80% is shown in red. At exactly 100% the budget is used up and matching requests are blocked. |
| **Over 100%** | Exhausted: the whole bar and the amount turn red. Matching requests are blocked until reset. |

## What a budget limits

Spend limits are defined in USD. Budgets can also cap tokens and request count.

Usage you import never counts toward a budget, and batch jobs are settled after the fact rather than blocked in real time; see [Importing external usage](external-usage.md) and [Budget enforcement](budget-enforcement.md#imported-usage-batches-and-video).

## Refusals and `Retry-After`

A blocked request gets an error with the code `budget_exceeded` and, when the budget resets, a `Retry-After` header with the seconds until the next period. See [Error codes](api-reference.md#error-codes).

How holds, estimates, per-minute limits and refusals work is in [Budget enforcement](budget-enforcement.md).

Ready to set up a budget? See [Set a budget and ceiling](set-a-budget-and-ceiling.md).
