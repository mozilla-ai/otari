---
description: How your organization's balance is funded and spent, and what keeps it from running out.
icon: wallet
---

# Billing

Your organization's balance pays for requests served with otari.ai's own
provider keys and for otari.ai's hosted tools. Requests that use your own
provider keys are billed by that provider instead.

Only owners and admins see billing: open **Organization** and select
**Billing**.

## The balance

Each request is charged to the balance when it finishes. While it runs, its
expected cost is held, so the Billing page shows what is **Available**: the
balance minus what running requests hold.

When nothing is available, new requests paid from the balance are refused until funds arrive. Requests on your own provider keys are not affected.
Requests already running finish, so the balance can end slightly below zero.

A new organization gets $2 once its creator answers every question in the
welcome questionnaire.

## Three ways to add funds

| | Charges | Controlled by |
| --- | --- | --- |
| **One-off top-up** | Once, for $5 to $10,000 | Any owner or admin |
| **Auto top-up** | When the balance falls below a threshold | The cardholder who set it up |
| **Recurring top-up** | Every day, every week, or on a day of the month | The cardholder |

Payment happens on Stripe. Each charge adds a 4.5% service fee and VAT based on
the billing address; only the amount itself goes to the balance. The person who
paid gets the receipt by email, and each deposit is listed under
**Transactions**.

## Redeeming a voucher

A voucher is a code from otari.ai that adds credit to the balance. Any owner or
admin can redeem one under **Redeem a voucher** on the Billing page; the credit
goes to the organization the dashboard is showing.

- Each person can redeem a voucher once. Another owner or admin of the same
  organization can redeem it too.
- A voucher expires, may have a total redemption limit, and works only in the
  region it was issued for.
- Voucher credit is spent before paid funds and does not expire once redeemed.

Credits, including vouchers and the welcome credit, are listed under
**Transactions** without a receipt.

## Auto top-up

An auto top-up refills the balance automatically. You set:

- **Threshold**: when the balance drops below it, a top-up is charged.
- **Amount**: what each top-up adds. It must be at least the threshold.
- **Monthly limit**: the most the card is charged per calendar month (UTC),
  fee and VAT included. Once reached, no more top-ups until the 1st; requests
  keep running on what is left.

Top-ups are at least one minute apart.

Whoever sets up the auto top-up saves their own card on Stripe and becomes the
**cardholder**. Only they can change the card or the amounts. Any other owner
or admin can turn it off and set up their own. If the cardholder stops being an
owner or admin, the auto top-up turns off.

### When it pauses

An auto top-up pauses, and says why on the Billing page and by email, when:

- the card is declined three times in a row, for example because it expired;
- the card is removed on Stripe;
- Stripe cannot prepare the charge, usually because the billing address is
  incomplete;
- the cardholder's bank asks to confirm a charge and it is not confirmed within
  three days.

The cardholder resumes it by changing the card, or by opening
**Manage auto top-up** and saving it again. Meanwhile, any owner or admin can
add a one-off top-up.

## Being told before requests stop

otari.ai emails every owner and admin when something needs attention. Each
alert is sent once while its condition lasts, its subject starts with the
organization's name, and an all-clear follows when it ends. To include an
on-call inbox or shared alias, invite it to the organization as an admin.

| Alert | Sent when |
| --- | --- |
| **Requests are being refused** | The balance is used up. At most once an hour. All-clear: **Requests are being served again**. |
| **Monthly limit reached** | An auto top-up would pass the monthly limit. |
| **Auto top-up paused** | One of the pause reasons above. |
| **Waiting on the bank** | The cardholder's bank asked for confirmation. The cardholder gets a separate email with the link. |
| **Last auto top-up this month**, **One auto top-up left**, **On pace to reach the limit** | The monthly limit is getting close. At most once a month each. |
| **Auto top-up turned off** | Someone other than the cardholder turned it off, or the cardholder no longer manages the organization. |

Each email's button, such as **Add funds** or **Change the card**, opens the
fix on the Billing page, switching to the right organization if needed. Only
the cardholder's emails offer changes to the auto top-up; everyone else's offer
**Add funds**.

## Budgets are a different control

The balance limits what the organization can pay for. [Budgets](budgets.md)
limit what a workspace, a member or a provider spends over a period. A request has to fit
both.
