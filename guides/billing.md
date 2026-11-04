# Billing

Requests served with otari.ai's own provider keys, and otari.ai's hosted tools, are paid from your organization's balance. Requests that use your own provider keys are billed by that provider, not from the balance.

Billing is visible to the organization's owners and admins. Members see neither the balance nor the transactions. To find it, open **Organization** and select **Billing**.

## The balance

Every request is charged to the balance when it finishes. While a request is running, its expected cost is held, so the Billing page shows what is **Available**: the balance minus what running requests hold. A request starts only when something is available.

When nothing is available, requests are refused until funds arrive. A request that was already running is allowed to finish, so the balance can end slightly below zero.

A new organization starts with $2, so its first requests work before anyone adds funds.

## Three ways to add funds

| | How it charges | Who controls it |
| --- | --- | --- |
| **One-off top-up** | Once, now, for an amount from $5 to $10,000. | Any owner or admin. |
| **Auto top-up** | Whenever the balance falls below a threshold you choose. | The person who set it up (the cardholder). |
| **Recurring top-up** | On a schedule: every day, every week, or on a day of the month. | The cardholder. |

Payment happens on Stripe. Every charge adds a 4.5% service fee on top of the amount, and VAT is added at checkout based on the billing address. Only the amount itself is added to the balance. The person who paid receives the receipt by email, and each deposit is listed under **Transactions** with its receipt.

## Auto top-up

An auto top-up keeps the balance from running out without anyone watching it. You set three amounts:

* **Threshold**: when a request leaves the balance below this, a top-up is charged. It is checked as requests are billed or refused, and when you save.
* **Amount**: what each top-up adds. It has to be at least the threshold, so one top-up always lifts the balance back above it.
* **Monthly limit**: the most the card is charged in a calendar month (UTC), fee and VAT included. Once a top-up would pass it, no more are charged until the 1st. Requests keep running on whatever balance is left.

Top-ups are at least one minute apart, so a burst of traffic cannot charge the card many times in a row.

**The card belongs to the cardholder.** Whoever sets up the auto top-up saves their own card on Stripe and becomes its cardholder. Only they can change the card, the threshold, the amount or the limit. Any other owner or admin can turn it off, and can then set up their own. If the cardholder stops being an owner or admin of the organization, the auto top-up turns off, even while it is paused, so their card is never charged for an organization they no longer manage.

**When it pauses.** An auto top-up stops charging, and tells you why on the Billing page and by email, when:

* the card is declined three times in a row, including because it expired or Stripe would no longer charge it;
* the card is removed on Stripe;
* Stripe cannot prepare the charge, usually because the billing address is incomplete;
* the cardholder's bank asks to confirm a charge (it waits up to three days for the confirmation) or the confirmation is not completed.

If the card is removed while a charge waits on the bank, auto top-up stays paused for a new card even once that charge is confirmed or canceled.

The cardholder resumes it by changing the card, or by opening **Manage auto top-up** and saving it again. Until then, any owner or admin can keep the organization running with a one-off top-up.

## Being told before requests stop

otari.ai emails every active owner and admin of the organization when the balance runs out, when auto top-ups pause or reach the monthly limit, and when the limit is getting close. Each email says what happened, why, and who can fix it. Members are not emailed, since billing is visible to owners and admins only. To have an on-call inbox or a shared alias receive them too, invite it to the organization as an admin.

Each alert is sent once while its condition lasts, however many requests it affects. Every subject starts with the organization's name, so an inbox rule or an on-call tool can match on it. When an empty balance, a reached monthly limit or a pause ends, an all-clear follows, sent to the people who received the alert.

If the balance runs out again within an hour of an empty balance you were told about being refilled, the alert waits for the rest of that hour, and is not sent at all if funds arrive first. A balance that each auto top-up refills and requests empty again a minute later therefore sends at most one alert an hour, not two emails a minute.

If an alert cannot be delivered, for example because the mail server is down, it is tried again after 10 minutes, then after longer gaps for about eight hours. Each retry goes only to the people who have not received it yet, and shows the balance and the other figures as they are when it is sent. An alert whose condition ends before it is delivered is not sent.

Each alert opens with how serious it is: **Outage** when requests are refused, **Action needed** when auto top-up has stopped, **Heads-up** when it will stop soon or was turned off, and **Resolved** for an all-clear. The fix and its button come right under the opening sentence. The limit warnings show this month's charges against the limit as a bar.

| Alert | When | Who receives it |
| --- | --- | --- |
| **Requests are being refused** | The balance is used up, so requests paid from it are refused. Says why no top-up refilled it. Requests that use your own provider keys are not affected. All-clear (**Requests are being served again**) when funds are added. At most one an hour. | Owners and admins |
| **Monthly limit reached** | An auto top-up was due, but it would pass the monthly limit. Top-ups resume on the 1st (UTC), or sooner if the limit is raised. All-clear when the next top-up is paid. | Owners and admins |
| **Auto top-up paused** | The card was declined three times in a row (including because it expired or Stripe would no longer charge it), the card was removed on Stripe, Stripe could not prepare the charge (usually an incomplete billing address), or a bank confirmation was not completed. All-clear when top-ups resume. | Owners and admins |
| **Waiting on the bank** | The cardholder's bank asked them to confirm a top-up. The cardholder is emailed separately, with the link to confirm it. All-clear when it is confirmed. If it is not confirmed within three days, the charge is canceled and **Auto top-up paused** follows. | Owners and admins, except the cardholder |
| **Last auto top-up this month**, **One auto top-up left** and **On pace to reach the limit** | After a top-up is paid: no further top-up fits in the monthly limit, only one more fits, or the pace of this month's earlier top-ups reaches the limit at least two days before it resets (so never after the month's first top-up). Each is sent at most once a month. | Owners and admins |
| **Auto top-up turned off** | Someone other than the cardholder turned it off, the cardholder no longer manages the organization, or the Stripe customer holding the card was deleted. When the cardholder turns it off while an alert about it is open, the people who received that alert are told it ended. Sent once. | Owners and admins, except whoever turned it off |

Only the cardholder can change the card or the terms of an auto top-up, because it charges their card. Their alerts link to the fix. Everyone else's say who can fix it and offer what they can do instead, usually **Add funds**, which any owner or admin can do.

Each button opens what it names on the Billing page: **Raise the limit**, **Change the card** and **Resume auto top-up** open the auto top-up's settings, **Add funds** opens a one-off top-up, and **Set up auto top-up** starts a new auto top-up. If you belong to more than one organization and the dashboard is showing another one, it switches to the organization the email is about and says so.

Receipts are separate from these alerts: each one goes only to the person who paid.

## Budgets are a different control

The balance limits what the organization can pay for. [Budgets](budgets.md) limit what is spent within the organization: by a workspace, a member, or a provider, over a period. A request has to fit both.
