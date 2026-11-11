# Review usage and requests

**Overview** opens with what needs attention, then three tabs over one time
range and one set of filters:

- **Usage**: spend, tokens and requests over time, with breakdowns.
- **Requests**: the per-request log.
- **Sessions**: agent sessions; see [Sessions](sessions.md).

The time range and the model, user, API key and status filters stay in place when
you switch tabs, and are kept in the URL. **Status** narrows Requests, and
Sessions for errors and successes; Usage ignores it. Clicking a chart bar narrows the range
to that bar. **Today** starts at local midnight. Every tab shows everyone's
requests in the workspace; pick yourself under **User** to see only yours.

For every workspace at once, an owner or admin opens **Organization** →
**Usage**.

## What needs attention

The strip at the top of the Overview appears only when there is something to act
on, and each item links to the page that explains it. It ignores the page's
filters.

- A member sees their own error rate when it is high.
- An owner or admin sees the organization's error rate and any spend ceiling
  near its limit.
- An operator sees the deployment's error rate, provider health and budgets.

## Find one request

To find one request, open `/requests?request=<id>` with the `Otari-Request-ID`
its response carried. Routed requests share a request group, so the log shows
each model tried and the one that served the response. Imported usage is labeled
by its source and does not use a budget (see
[Import usage from outside Otari](external-usage.md)).
