---
description: Create a spending budget, then apply it with a ceiling.
icon: square-dollar
---

# Set a budget and ceiling

A budget says how much can be spent and over what period. On its own it limits
nothing: a **spend ceiling** applies it to the organization or a workspace.

## Create a budget

1. Open **Organization**, then **Spend & budgets**.
2. Click **Add budget**.
3. Set an optional **Name**, the **Limit (USD)** (blank for no dollar limit),
   and how often it **Resets**: daily, weekly (on Mondays) or monthly (on the
   first of the month), at midnight UTC.
4. Optionally, open **Token and request limits** to also cap the tokens or the
   number of requests per period. Blank caps nothing.
5. Click **Add budget**.

## Apply a spend ceiling

1. In **Spend & budgets**, click **Add ceiling**.
2. Under **Capping**, choose the organization or one of its workspaces.
3. Optionally, enter a **Provider instance** to cap spend on that provider only.
   Leave it blank to cap spend across every provider.
4. Choose the **Budget** the ceiling uses, and optionally a **Name**.
5. Click **Add ceiling**.

Requests the ceiling matches count toward its budget. When the budget is used
up, new matching requests are refused until the next reset; requests already
running finish normally.

Every ceiling that matches a request applies on its own. With a ceiling on the
organization and another on a workspace, a request from that workspace must fit
both, and is refused when either is used up.

To cap a single API key, set a budget on the key itself when you create or edit
it.

## Check how much is left

**Spend & budgets** shows each budget's spend this period against its limit,
and marks a budget once it passes 80% of it and again once it is used up.

How budgets are enforced, and what a refused request looks like, is in
[Budgets](budgets.md).
