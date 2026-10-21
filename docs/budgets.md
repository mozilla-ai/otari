# Budgets

A budget caps what can be spent over a period, and Otari refuses requests that
would pass it. This guide sets one up in the dashboard. How enforcement works
(the budget forms, the spend, token and request axes, reservations, refusals and
`Retry-After`) is in [Access control](access-control.md#budgets).

In the dashboard a budget comes in two parts. The **budget** defines a limit and
a period. A **spend ceiling** applies that budget to something: the
organization or a workspace, optionally for one provider only. A budget on its
own blocks nothing until a ceiling uses it. Each ceiling is checked on its own,
so a request has to fit every ceiling that applies to it. On otari.ai a request
also has to fit the organization's [balance](billing.md#budgets-are-a-different-control).

## Create a budget and a spend ceiling

1. Open **Organization** and select **Spend & budgets**.
2. Click **Add budget**, then set an optional **Name**, the **Limit (USD)**
   (blank for no dollar limit) and how often it **Resets**: Daily, Weekly
   (Mondays) or Monthly (the first of the month), at midnight UTC. Click
   **Add budget**.
3. Click **Add ceiling**, then choose what it caps under **Capping** (the
   organization or one of its workspaces), optionally a **Provider instance** to
   cap one provider only, the **Budget** to use, and an optional **Name**. Click
   **Add ceiling**.

Requests matching the ceiling count toward its budget. When the budget is used
up, new matching requests are refused until the next reset. Requests already
running finish normally.

To cap one API key, set its budget on the key itself; see
[Cap a key's spend](api-keys.md#cap-a-keys-spend). Through the API, a budget can
also cap tokens and requests, limit each user per minute, or use a rolling
duration instead of a calendar period; the fields are in
[Access control](access-control.md#budgets).

## Check a budget's status

**Spend & budgets** shows each budget's spend this period against its limit,
and marks a budget once it passes 80% of it and once it is used up.

A refused request answers `403` with `Otari-Error-Code: budget_exceeded`; the
headers it carries are in
[Refusals and `Retry-After`](access-control.md#refusals-and-retry-after).

