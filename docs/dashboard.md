# Admin dashboard

The dashboard is where you manage Otari in the browser. On otari.ai, sign in at
[otari.ai](https://otari.ai); a self-hosted gateway serves it at its root. It
uses the same management API as scripts, so the server decides what you can see
and do, and a page the gateway does not serve is not offered.

What a self-hosted operator sets up and sees (the master key, the first run and
the deployment view) is in [Run the dashboard](run-the-dashboard.md).

## Signing in

Sign in with a password, a passkey, or Google or GitHub, depending on what the
deployment offers. Where signup is open, a new address can create an account
with its own organization, after accepting the terms of service where they are
published. A link to a dashboard page opened while signed out shows the sign-in
screen first, then that page. See [Sign-in and accounts](sign-in.md).

Signing in can also add you to an organization that has proven your email
domain; see [Email-domain auto-join](members.md#email-domain-auto-join).

**Account settings** holds the name you are known by in the account menu, the
organization roster and the Overview's user column. Someone added by address has
no name until they set one; with no name, you are shown by your sign-in address.

## The setup guide

Overview shows a short setup flow until the selected workspace serves its first
successful request from your own code. Messages sent in the Playground do not
count. The key the guide creates is an ordinary workspace API key.

Skipping the guide hides it for that workspace and does not revoke a key it
created. A self-hosted gateway can turn it off; see
[The setup guide](run-the-dashboard.md#the-setup-guide).

## Navigation

Page names and availability depend on configuration and installed extensions.
The running dashboard is the source of truth.

The workspace view holds day-to-day gateway work:

| Page | What it is | More |
| --- | --- | --- |
| Overview | What needs attention, then Usage, Requests and Sessions as tabs, and the setup guide | [The Overview](#the-overview), [Agent traces](traces.md) |
| Playground | Chat with a model or compare two | [Playground](#playground) |
| Models | The catalog, grouped by model | [Models and Routing](#models-and-routing) |
| Routing | Routing policies, Automatic fallback, and the workspace's guardrails | [Models and Routing](#models-and-routing), [Routing](routing.md) |
| Tools | Web search, code execution and MCP servers | [Built-in tools](tools.md) |
| API keys | Workspace API keys | [API keys](api-keys.md) |
| Members | Workspace members | [Members and invitations](members.md) |

The organization view holds tenant-wide administration:

| Page | What it is | More |
| --- | --- | --- |
| Usage | Organization-wide usage | [Workspace-scoped spend](workspaces.md#workspace-scoped-spend) |
| Workspaces, Members & roles | Workspaces and organization members, with invitations | [Organization](#organization) |
| Email domains | Domains whose colleagues join automatically | [Email-domain auto-join](members.md#email-domain-auto-join) |
| Spend & budgets | Budgets and the spend ceilings holding them | [Budgets](budgets.md) |
| Providers > Inference | The organization's own provider keys, their models, and what the organization pays for them | [Organization provider keys](hosted-providers.md#organization-provider-keys), [Who is shown which models](models.md#who-is-shown-which-models) |
| Providers > Web search | The organization's own search and fetch keys | [Web search keys](#web-search-keys) |
| Guardrails | Guardrails mandated across the organization | [Manage guardrails](manage-guardrails.md) |
| Org settings | The organization's name and settings | [Organization](#organization) |

Organization owners and admins manage Providers: the keys, the models each key
offers, and the rate each model is billed at (see
[Pricing](hosted-providers.md#pricing)). A self-hosted operator also has a
deployment view, opened from the account menu; see
[The deployment view](run-the-dashboard.md#the-deployment-view).

### Web search keys

**Providers > Web search** lists the organization's search keys and fetch keys,
one table each.

- **Add a key**: pick the provider, then enter a name and the API key. **More
  settings** holds a custom endpoint and the provider's options.
- **Star a key** to make it the default for every workspace that has not pinned
  another. Click a filled star to unset it. A key that is neither starred nor
  pinned is not used.
- **Archive a key** to take it out of use; restore or delete it later.

A workspace uses the key its owners or admins pinned, else the starred key, else
the deployment's own search or fetch tool. Editing a key never shows its secret,
and moving it to another endpoint asks for the API key again. A key the
deployment cannot decrypt is marked unreadable. Prices for search and fetch calls
are set by the deployment's operator, not here.

### What a page shows depends on who is signed in

- **Spend & budgets.** An organization owner or admin manages their
  organization's budgets and ceilings. A deployment operator also sees the
  deployment's own budgets and the gateway users assigned to them.
- **API keys.** An operator manages every key in the organization and picks each
  key's owner. A member sees only their own keys, always billed to themselves and
  never budget-exempt. An operator, or an organization owner or admin, can cap a
  key's spend from its create and edit forms, and the Budget column shows each
  key's spend against its limit. Where the deployment enables it, a key's actions
  menu offers **Workload identity…**; see
  [Rules on a member's own key](workload-identity.md#rules-on-a-members-own-key).
- **The Overview's attention strip** appears only when there is something to act
  on, and each item links to the page that explains it. It ignores the page's
  filters. A member sees their own error rate when it is high; an owner or admin
  sees the organization's error rate and any spend ceiling near its limit; an
  operator sees the deployment's error rate, provider health and budgets.
- **The Overview's tabs and Routing** show an operator their active organization
  only.

## Models and Routing

**Models** shows one card per model, with filters beside the list. A model's page
compares every offering in a table, with each offering's limits and the price
your organization pays. **Use this model** shows a request to copy, for the
route the offering is served on.

- Prices are shown in the unit they bill, such as "$0.50 / 1M" for tokens or
  "$0.04 per image". A card shows the cheapest price when all offerings bill in
  one unit, and "priced in several units" otherwise.
- **Price rules** under the table lists each offering's overrides and multipliers
  in words, such as "×0.5 on every price when service tier is flex". They are
  read-only here; see [Overrides and multipliers](configuration.md#overrides-and-multipliers).
- Each row shows the provider's own model ID, which tells apart offerings on one
  provider, such as Bedrock's regional inference profiles.
- **Use this model** opens a drawer with a request to copy.

The page is read-only. An organization sets its own rate on **Providers >
Inference**, which an offering on one of its keys links to. See
[Browsing the grouped catalog](models.md#browsing-the-grouped-catalog).

**Routing** lists each policy with its main model, its backups in order, and who
it applies to. In the policy forms you pick a model, then a provider. See
[Failover](routing.md#failover).

**Automatic fallback**, on the Routing page, lists every model your organization
sent requests to in the last 30 days, most requested first, by catalog id (see
[Catalog spellings](models.md#catalog-spellings)). Each row shows:

- its request count;
- the share of requests that sent the catalog id and so could fall back (a dash
  when one provider serves the model, flagged when below half);
- its providers in the order a request tries them, with the input price that
  order ranks by.

That order is a baseline: automatic routing can reorder it per request, and a
key's allow-list or a missing credential can leave a provider out. A row whose
requests named a provider or a policy opens on the API keys that sent them. An
owner or admin sees every workspace in the organization; anyone else, the
workspaces they belong to.

## Entering a rate

The rate form asks for the unit first: tokens, requests, images, megapixels,
characters, minutes of audio, seconds of video, or web searches. Enter each rate
the way providers publish it:

| Unit | Entered per |
| --- | --- |
| Tokens, characters | 1,000,000 |
| Requests, images, searches | 1,000 |
| Megapixels, minutes, seconds | 1 |

A minute of audio has two rates, one for audio the model hears and one for audio
it speaks. The model lists show each rate in its unit, for example "$0.002 per
request" or "$0.06 per second". How each unit is stored and billed is in
[Each endpoint in its unit](configuration.md#each-endpoint-in-its-unit).

Some limits of the form:

- An override that prices seconds of both audio and video shows its audio rates;
  change its video price through the API.
- An edit in the same unit keeps the override's context tiers and rules, which
  the form does not show.
- The hosted model form stores per-million fields only, so it does not offer
  seconds of video and cannot reprice a per-second model.

**Price this model** on a request in the Requests tab opens the same form, in the
unit the request's endpoint bills. It is offered on a served request that carried
no cost.

## Playground

The Playground is the in-product chat page: pick a model and talk to it, or put
two side by side and record which answered better. Requests run through the same
path as any completion, so routing policies, guardrails, budgets and tools all
apply, and every request appears on the Overview. How to use it is in
[Playground](playground.md).

**Billing.** A Playground request uses no API key. It is billed to you, in the
workspace selected in the switcher, against your own budget and rate limit, and
its usage row has no key. A viewer cannot send one. Playground requests do not
complete the setup guide.

**Files.** Attach files with the paperclip or by dropping them on the page. The
model reads them like an API request's [uploaded files](files.md); in Compare,
both models get them. Removing a file from a question keeps the upload, and the
**Files** dialog lists your uploads in the workspace to reattach or delete. These
controls are hidden when file uploads are off.

**Tools.** The **Tools** menu adds web search, code execution and MCP servers to
the next message, in any mix. Each tool call shows above the reply as a
collapsed row with its status, its arguments and what it returned. This is shown
only in the Playground and is not saved; an API request still gets only the final
message (see [What clients receive](tools.md#what-clients-receive)).

**What is stored.** Each request is recorded in usage, as any request is. Its
content is kept only once you save a conversation or record a comparison. The first time, you are asked to agree to content retention; you can
withdraw that later, which blocks new saves and deletes nothing. Saved
conversations, comparisons and pinned models are visible only to you, and each
can be deleted.

## Observability

### The Overview

The Overview opens with what needs attention (see
[What a page shows depends on who is signed in](#what-a-page-shows-depends-on-who-is-signed-in)),
then three tabs over one time range and one set of filters:

- **Usage**: spend, tokens and requests over time, with breakdowns.
- **Requests**: the per-request log.
- **Sessions**: agent sessions (see [Agent traces](traces.md)).

The time range and the model, user, API key and status filters stay in place when
you switch tabs, and are kept in the URL. Clicking a chart bar narrows the range
to that bar. **Today** starts at local midnight. Every tab shows everyone's
requests in the workspace; pick yourself under **User** to see only yours.

### Requests and Sessions

**Requests** is the per-request log. To find one request, open
`/requests?request=<id>` with the `Otari-Request-ID` its response carried. Routed
requests share a request group, so the log shows each model tried and the one
that served the response. Imported usage is labeled by its source and does not
use a budget.

**Sessions** lists agent sessions where the deployment records traces.

- A model filter matches sessions with an LLM call on that model.
- The Levels column counts each session's errors, tool errors included; a tool
  error alone does not fail a session (see
  [Sessions and failures](traces.md#sessions-and-failures)).
- Click Last activity, Levels, Latency, Tokens or Total cost to sort; click again
  to reverse.
- A session opens in a side panel. Before you pick a span, it shows what the
  session cost against its key's spend ceiling and where the gateway stepped in.
- **View trace** in a request's detail opens its session on that call.

## Organization

An organization owns workspaces. The workspace switcher decides which keys,
members, usage, tools, aliases and routing policies are in view; the organization
section manages what those workspaces share. What you can see or change depends
on your organization and workspace roles; see [Access control](access-control.md).

**Invitations** waiting for you are listed under Invitations, reached from the
organization switcher. Someone signed in who belongs to no organization is taken
there when an invitation is waiting; otherwise Overview says how to get in.
Creating an organization is offered only where signup is open.

Opening an invitation link while signed in as a different address says who the
invitation is for and offers to sign out, because accepting always adds the
invited address. After you set a password from an invitation link, the sign-in
screen opens with your address filled in.

On **Members & roles**, an invited row offers **Reissue** and **Revoke**; see
[Manage existing members](members.md#manage-existing-members). When an
invitation email does not go out, you get the accept link to share; see
[Invitations](members.md#invitations).
