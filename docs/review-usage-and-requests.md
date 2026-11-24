# Review usage and requests

**Home**, the page the dashboard opens on, shows what a time window cost and
where the spend went. The per-request log is the **Requests** view of
**Sessions**, beside its list of agent sessions (see [Sessions](sessions.md)).

## Home

Pick the window at the top: **Today**, **Yesterday**, **7D**, **30D**, **3M** or
**6M**; Home opens on **30D**. Days are your browser's calendar days. **Today**
runs from midnight to now and **Yesterday** is the whole day before it, both
read by the hour; the others are whole days ending with today. On a phone the
same windows are in a select. What Home shows depends on your role.

An organization owner or admin sees the whole organization, or one workspace
picked beside the window when there is more than one:

- What the window cost, against the same span before it (**Today** against
  yesterday up to the same time of day), with spend per hour or per day.
- How much of the organization's spend limit is used, and what the period is on
  pace to cost. With no limit set, **Set a team budget** sets one.
- **Failures**: the share of requests that failed.
- Billed tokens, and how much of the input was read from cache.
- **People active**: how many members spent anything in the window, and how
  many sessions ran. **See the sessions** opens them.

**Where it went** then breaks the window's spend down by **People**,
**API keys**, **Models**, **Providers** or **Workspaces** (offered when there is
more than one workspace). Each row shows what it spent, its tokens and its share
of the total; outside **Models**, it also shows its top model, and outside
**Models** and **Providers** the spend limit it works under. The five largest
rows show first, and **Show all** lists the rest. A row opens its sessions over
the same window, a provider's row its requests, and a workspace's row narrows
the table to that workspace's people. A row's menu sets or changes its spend
limit, and a key's row can be opened in **API keys**.

**Providers** names the provider each request went to, or the instance an
operator named, which is the figure to match against each provider's own bill
when your organization brings its own provider keys. Requests that recorded no
provider, such as imported usage that did not name one or requests refused
before a provider was picked, share one **Unknown provider** row, so the rows
still add up to the total.

Anyone else sees only their own requests in the selected workspace: what they
spent, the spend limit on them there, their tokens and their sessions.
**Where it went** breaks their spend down by **API keys**, **Models** or
**Providers**, and a row opens its sessions, or a provider's row its requests.
Someone with no active key and no session yet is offered
**Create API key**; someone who may not create keys is told whom to ask.

### Get started

On a gateway with nothing to route to and nothing served yet, Home opens with
**Get started with Otari**. The operator is offered **Add your first provider**,
an organization owner or admin **Add a provider key**, and anyone else is told
to ask an owner or admin.

## Requests

Open **Sessions** and pick **Requests** for the per-request log. Sessions and
Requests share one time range and the model, user, API key and status filters,
which stay in place when you switch between them and are kept in the URL.
**Status** narrows Requests, and Sessions for errors and successes. **Provider** narrows Requests
only, so it stays behind when you switch to Sessions. On Requests, dragging
across the chart narrows the range. Both views show everyone's requests in the workspace; pick
yourself under **User** to see only yours.

The **Router savings** tab beside them has its own period control and none of
these filters; it is covered under
[Agent model recommendations](routing.md#agent-model-recommendations).

## Find one request

To find one request, open `/requests?request=<id>` with the `Otari-Request-ID`
its response carried. Routed requests share a request group, so the log shows
each model tried and the one that served the response. Imported usage is labeled
by its source and does not use a budget (see
[Import usage from outside Otari](external-usage.md)).
