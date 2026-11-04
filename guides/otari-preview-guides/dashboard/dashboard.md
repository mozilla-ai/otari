# Admin dashboard

Otari serves its dashboard at the gateway root. The dashboard uses the same management API available to scripts, so access and behavior are determined by the server rather than duplicated in the frontend.

The dashboard reads `GET /api/v1/bootstrap` before rendering. That response controls the available sign-in methods and navigation surfaces, so a page the deployment does not serve is not offered.

## Sign-in and secrets

The master key and `OTARI_SECRET_KEY` are different credentials:

| Credential         | Purpose                                                              |
| ------------------ | -------------------------------------------------------------------- |
| Master key         | Authorizes deployment management and signs in the first operator     |
| `OTARI_SECRET_KEY` | Encrypts provider and search-tool credentials stored in the database |

If `OTARI_MASTER_KEY` is unset on a new database, Otari generates a master key and prints it once. The dashboard exchanges it for an HttpOnly session cookie; it does not store the master key in browser storage.

Set `OTARI_SECRET_KEY` before saving provider or search-tool credentials through the dashboard:

```bash
otari gen-secret-key
```

Keep this key separate from the database. Losing it makes stored credentials undecryptable. Providers configured in `config.yml` do not require it.

The operator can claim a new deployment by setting an email address and password from Account settings. After that, the sign-in page uses the password rather than the master key. The master key remains valid for management API calls and can reset the operator password. It resets the operator's password only: to help a member who forgot theirs, send them a password link from Accounts, which emails it where mail is configured and otherwise gives you the link to share (see [Deployment-wide account administration](https://docs.otari.ai/self-hosting/sign-in#deployment-wide-account-administration)).

When the sign-in page offers the password form, and when it offers the master-key box beside it, is described in [Dashboard sessions and identity](https://docs.otari.ai/self-hosting/sign-in#dashboard-sessions-and-identity).

Account settings also carries the name you are known by, which is not a credential: it is what the account menu, the organization roster and the user column on the Overview shows. Someone an admin added to the roster by address has no name until they set one here. Clearing it puts those surfaces back to naming you by your sign-in address. The dashboard reads who is signed in from `GET /api/v1/auth/profile`, which answers whether or not you belong to an organization, so the account menu names you even before you join one.

## First-run walkthrough

1. Start Otari.
2. Open the gateway root, usually `http://localhost:8000/`.
3. Sign in with the configured or generated master key.
4. Add a provider. If storing its credential in the dashboard, configure `OTARI_SECRET_KEY` first.
5. Test the provider connection or declare model IDs for a backend that has no model-listing endpoint.
6. Create an API key from the setup guide or the Keys page.
7. Send a request and confirm it appears on the Overview, in Requests.

The [Self-hosted quickstart](https://docs.otari.ai/self-hosting) includes a complete request example.

## The setup guide

Overview offers a short setup flow until the selected workspace serves its first successful gateway request from outside the product. A message sent in the Playground does not close it: the guide marks the moment somebody's own code first reached the gateway, which is the point of the flow. The key it creates is an ordinary workspace API key. Skipping the guide hides it for that workspace; it does not revoke a key already created. Set `activation_guide: false` to disable the flow for the deployment.

## Navigation

Exact page names and availability can change with configuration and installed extensions. The running dashboard is the source of truth.

The workspace view contains day-to-day gateway operations:

| Page       | What it is                                                                                                                             | Rules                                                                                                                                        |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Overview   | The landing page: what needs attention, then Usage, Requests and Sessions as tabs, and the [setup guide](dashboard.md#the-setup-guide) | [The Overview](dashboard.md#the-overview), [Agent traces](../observability/traces.md)                                                        |
| Playground | In-product chat and comparisons                                                                                                        | [Playground](dashboard.md#playground)                                                                                                        |
| Models     | The catalog grouped by model                                                                                                           | [Models and Routing](dashboard.md#models-and-routing), [Models](https://docs.otari.ai/reference/gateway/models#the-catalog-grouped-by-model) |
| Routing    | Routing policies, aliases and Automatic fallback                                                                                       | [Models and Routing](dashboard.md#models-and-routing), [Routing](https://docs.otari.ai/reference/gateway/routing)                            |
| Tools      | Built-in tools, MCP servers and guardrails for the workspace                                                                           | [Built-in tools](https://docs.otari.ai/reference/gateway/tools)                                                                              |
| API keys   | Workspace API keys                                                                                                                     | [API keys](../organizations-and-access/api-keys.md)                                                                                          |
| Members    | Workspace members                                                                                                                      | [Members and invitations](../organizations-and-access/members.md)                                                                            |

The organization view contains tenant-wide administration:

| Page                                     | What it is                                                                                                | Rules                                                                                                                                                                                                                         |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Usage                                    | Organization-wide usage                                                                                   | [Workspace-scoped spend](../organizations-and-access/organizations-and-workspaces.md#workspace-scoped-spend)                                                                                                                  |
| Workspaces, Members                      | Workspaces and organization members, with invitations                                                     | [Organization](dashboard.md#organization)                                                                                                                                                                                     |
| Email domains                            | Domains whose colleagues join automatically                                                               | [Email-domain auto-join](../organizations-and-access/members.md#email-domain-auto-join)                                                                                                                                       |
| Spend and budgets                        | Budgets and the spend ceilings holding them                                                               | [Budgets](../organizations-and-access/budgets.md)                                                                                                                                                                             |
| Providers > Inference (under General)    | The organization's own provider keys, the models each one offers, and what the organization pays for them | [Organization provider keys](https://docs.otari.ai/self-hosting/providers/hosted-providers#organization-provider-keys), [Who is shown which models](https://docs.otari.ai/reference/gateway/models#who-is-shown-which-models) |
| Providers > Web search (under General)   | The organization's own search and fetch keys                                                              | [Web search keys](dashboard.md#web-search-keys)                                                                                                                                                                               |
| Guardrails, Org settings (under General) | Organization guardrail policy and settings                                                                | [Guardrails](../guardrails/guardrails.md)                                                                                                                                                                                     |

The deployment view, opened from the account menu, holds what belongs to the gateway process rather than to any organization on it:

| Page                   | What it is                                                                                   | Rules                                                                                                                                 |
| ---------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Settings               | The running configuration, and the Rate limit rules card                                     | [Settings](dashboard.md#settings)                                                                                                     |
| Accounts               | Every account on the deployment                                                              | [Deployment-wide account administration](https://docs.otari.ai/self-hosting/sign-in#deployment-wide-account-administration)           |
| Hosted providers       | The provider keys the deployment serves inference on                                         | [The Hosted providers page](https://docs.otari.ai/self-hosting/providers/hosted-providers#the-hosted-providers-page)                  |
| Subagent router        | The decision models that pick the model a coding agent's subagent runs on                    | [Subagent router](dashboard.md#subagent-router), [Subagent routers](https://docs.otari.ai/reference/gateway/routing#subagent-routers) |
| Web search             | The search and fetch tools the deployment runs for every workspace, and what each call costs | [Web search and Code execution pricing](dashboard.md#web-search-and-code-execution-pricing)                                           |
| Code execution pricing | What the deployment charges per successful call of `otari:code_execution`                    | [Web search and Code execution pricing](dashboard.md#web-search-and-code-execution-pricing)                                           |

Accounts and Hosted providers are listed only for a deployment operator, and answer anyone else that they are not available to them. Web search and Code execution pricing are listed for one too, and say the same to anyone else who opens them. Tools > Web search links an operator to the Web search page.

Providers answers to the organization role rather than to deployment authority: its owners and admins manage the keys, the models each key offers, and the rate each model is billed at. Which rate applies is in [Pricing](https://docs.otari.ai/self-hosting/providers/hosted-providers#pricing).

### Web search keys

Providers > Web search is the organization's own search and fetch keys, in one table per kind. Adding a key asks for its provider first, from the providers the search and fetch libraries serve, and then for what that provider needs: a name, the API key, and under More settings an endpoint of its own and the provider's options. Editing a key never shows its secret; moving it to another endpoint needs the API key entered again. The star on a key sets it as the workspace default, the key every workspace uses unless it pins another, and clicking a filled star unsets it. A workspace uses the key its owners or admins pinned, otherwise the starred key, otherwise the deployment's own search or fetch tool, so a key added and not starred changes nothing until a workspace pins it. Archiving a key takes it out of use, and an archived key can be restored or deleted. A key this deployment cannot decrypt is marked unreadable. The page sets no prices: the deployment's operator prices an organization's search and fetch calls.

### Subagent router

Subagent router lists the decision models that pick the model a coding agent's subagent runs on when the otari-router plugin for Claude Code, or pi-otari, asks `POST /api/v1/routing/recommend`. A router is a provider (TypeSafe, OpenRouter or a llama.cpp server), a model, an optional API base, a key stored encrypted and shown by its last four characters, and the price per million input and output tokens the caller is charged. Several may be added and one is switched on at a time: the first one added is on, and the Serving switch moves it. Both prices at 0 make a free router, such as a local model. With none added, or every router switched off, no model is picked. This page is the only place the router is set. See [Subagent routers](https://docs.otari.ai/reference/gateway/routing#subagent-routers).

### Web search and Code execution pricing

Web search holds the search and fetch tools the deployment runs for every workspace and what each call costs. Its two tables list the search tools and the fetch tools, `builtin_fetch` first, each with its key's last four and its price per call, and end with the `otari:web_search` or `otari:web_fetch` rate, which prices a call on an organization's own key, and `otari:web_search` a search on the legacy settings too. A tool is priced at its own rate, or at one stored under its old `<provider>:<name>` model key, and a tool with neither is unpriced. A row opens its endpoint, enrichment and options; its star makes it the default the model uses, and Test runs one search or fetch and says how many hits or characters came back. Add search tool and Add fetch tool pick a provider first, then ask for what it needs (its key, an endpoint when it has none of its own) and a price, with the endpoint, timeout, enrichment and the provider's options under More settings, typed from the provider catalog or written as JSON. Edit and Remove change a stored tool; a `config.yml` tool, `builtin_fetch` and the two `otari:` rates take only a price, and a rate from `price_per_call` is changed in `config.yml`. Below the tables, The model's web tools turns the model's search off, and sets the calls a request may make, the results per search, reading pages for hits without text, and running a provider's own web search here. The legacy SearXNG backend's URL and engines show while they are set.

Code execution pricing lists `otari:code_execution`.

On both pages, Edit sets a rate in dollars per call, in force from the moment it is saved, and 0 prices a tool as free. With `require_pricing` on, the page warns while a tool it lists has no rate, because a request that would run it is refused. See [Pricing a gateway-run tool](https://docs.otari.ai/reference/gateway/tools#pricing-a-gateway-run-tool).

### What a page shows depends on who is signed in

What a page shows can depend on who is signed in, not only on the deployment:

* **Spend and budgets.** An organization owner or admin manages their own organization's budgets and the spend ceilings holding them, while a deployment operator gets the deployment's own budgets, their active organization's, and the gateway users assigned to them.
* **API keys.** An operator manages every key in the organization and chooses each key's owner, while a member sees the same page scoped to their own keys, always billed to themselves and never budget-exempt. An operator, or an organization owner or admin, can also cap a key's own spend: the key's create and edit forms take an existing budget or a custom amount and reset, and a Budget column shows each key's spend this period against its limit. Where the deployment turns on `federation_member_rules`, a key's actions menu offers Workload identity… to the operator, and to an organization owner or admin on their own keys; see [Rules on a member's own key](https://docs.otari.ai/self-hosting/operations/workload-identity#rules-on-a-members-own-key).
* **The Overview's attention strip.** It appears only when there is something to act on, above the tabs on every one of them, and each item links to the page that explains it: a high error rate today opens Requests on today's failures. It is about what the caller answers for, so it does not follow the page's filters: for an organization member it carries their own error rate when it is high; an owner or admin sees the organization's error rate and a spend ceiling at or near its limit; a deployment operator sees the deployment's error rate, provider health and the deployment's own budgets.
* **The Overview's tabs and Routing** show an operator their active organization, not every organization on the deployment.

## Models and Routing

Models is the catalog grouped by model: one card per model, with a rail of filters beside it, and a page per model with its facts and every offering compared in a table, each with its own limits and the price your organization is charged. Each row shows the provider's own model ID under the provider, which tells apart two offerings on one provider, such as Bedrock's regional inference profiles. "Use this model" opens a drawer beside the table with the request to copy, sent to the gateway's pick or to a provider pinned by its selector. The page is read-only; an organization's own rate is set on Providers > Inference, which an offering on one of that organization's keys links to. An offering the deployment supplies the credential for says so instead, because its rate is the deployment price list's rather than a tenant's. What the page requests is in [Browsing the grouped catalog](https://docs.otari.ai/reference/gateway/models#browsing-the-grouped-catalog).

Routing lists each policy with its main model, its backups in order, and who it applies to; the empty list offers a fallback directly. In the policy forms, models are chosen by model and then by provider. How to create and edit a policy there is in [Failover](https://docs.otari.ai/reference/gateway/routing#failover).

Routing's **Automatic fallback** section, read from `GET /api/v1/organizations/me/automatic-fallbacks`, lists every model your organization sent requests to in the last 30 days, most requested first, by catalog id (see [Catalog spellings](https://docs.otari.ai/reference/gateway/models#catalog-spellings)). Each row shows its requests (an absorbed attempt is not one), the share of them that sent the catalog id and so could fall back, and its providers in the order such a request tries them for your organization, each at the input price that order ranks by. That order is the baseline; automatic routing can reorder it per request. The share is a dash for a model one provider serves, and is flagged below half. A row with requests that named a provider or a routing policy opens on the API keys that sent them, with the provider each reached; a policy's own fallback still applied to the ones that named a policy. The order is the catalog's, before a key's allow-list or a missing credential leaves an offering out of a request's plan. Requests from before a deployment recorded the catalog id they sent count as having named one. An owner or admin reads every workspace in the organization, anyone else the workspaces they belong to. The response also says who runs the built-in tools on each offering (`own_tools`, `otari_tools`), for a caller that declares them in the provider's own words and sends no `Otari-*` executor header.

## Settings

Settings shows the effective non-secret configuration. Some values can be changed at runtime and others require a restart. The server marks that distinction in the settings response. Its Rate limit rules card adds, edits and removes limits on requests per minute, tokens per minute and requests in flight; rules from config.yml are listed there read-only (see [Rate limit rules](https://docs.otari.ai/self-hosting/configuration#rate-limit-rules)).

### Entering a rate

A rate is per million tokens, per request, or per image, and the rate form asks which. A request or image rate is entered per 1,000, the way providers publish one: $2 per 1,000 searches for a rerank model is entered as 2 and stored as 2000 per million, which is how every rate travels on the wire. The model lists name such a rate by its unit, for example "$0.002 per request". The same choice is on the Requests view's "Price this model", offered on a served request that carried no cost (a failed attempt billed nothing, so its error is the guide instead), which opens per request for a rerank, audio or moderation request and per image for an image generation. What each unit bills is in [Per-request pricing](https://docs.otari.ai/self-hosting/configuration#per-request-pricing-audio-moderations-rerank-and-completion-models).

## Playground

The Playground is the in-product chat page: pick a model the gateway serves and talk to it, or put two side by side and record which answered better. Requests run through the same path as any other completion, so routing policies, guardrails, budgets and the tool loop all apply, and every request appears on the Overview.

A question can carry files. Attach them with the paperclip in the composer or drop them anywhere on the page; each uploads at once and shows as a chip with its name and size until the question is sent. The model reads them the way an API request's [uploaded files](https://docs.otari.ai/reference/gateway/files) are read, so a text-only local model is given their extracted text. When comparing, both models get the same files. A saved conversation remembers which files each question carried. Removing a chip takes the file off that question and keeps the upload: the Files dialog lists everything you uploaded in the workspace, to attach again or to delete. These controls do not appear when file uploads are off.

The Tools menu attaches web search, code execution and any number of MCP servers to the next message, in any mix: the model is offered all of them in one tool loop. A tool the gateway runs shows in the transcript above the reply as a collapsed row with its status, opening to the arguments the model sent and what the tool returned (a long result is shortened for display). This activity is shown in the Playground only and is not saved with a conversation; a request made through the API still returns just the final message, as [What clients receive](https://docs.otari.ai/reference/gateway/tools#what-clients-receive) describes.

Two things are worth knowing about how it is billed and what it stores.

A Playground request carries no API key. It is authorized by the dashboard session and billed to whoever sent it, in the workspace the switcher has selected, against that person's own budget and rate limit; its usage row therefore has no key attached. A viewer cannot send one, since it spends like a key, so the page tells them so instead of offering a composer. Picking a different model changes which provider credential the gateway resolves, never who pays. A request made here also does not count as the workspace's first request, so the setup guide keeps waiting for one from outside the product, which is the milestone it exists to mark.

Nothing is stored until asked for. Saving a conversation or recording a comparison prompts once for content retention, and the flags are per person and revocable; withdrawing one blocks new saves and deletes nothing. Saved transcripts, recorded comparisons and pinned models are visible only to the person who created them, even to colleagues in the same workspace, and each can be deleted from the page.

## Observability

### The Overview

The Overview is the landing page. It opens with what needs attention (see [What a page shows depends on who is signed in](dashboard.md#what-a-page-shows-depends-on-who-is-signed-in)), then reads the gateway three ways, as tabs over one time window and one set of filters: Usage (spend, tokens and requests over time, with breakdowns, at `/`), Requests (the per-request log, at `/requests`) and Sessions (agent sessions, at `/sessions`). Below the tabs, the time range and the model, user, API key and status filters are always shown; they live in the URL and travel when you switch tab, and clicking a chart bar narrows the window to that bar. Today, a time range on Usage, starts the window at local midnight. Every tab opens on everyone's requests in the workspace; pick yourself in the User filter to see only your own. The old `/usage` and `/activity` addresses open the matching tab with their filters.

### Requests and Sessions

The Overview's Requests tab is the per-request log. A request opens from the `Otari-Request-ID` its response carried, at `/requests?request=<id>`, on the attempt that served it or the one that failed last, wherever it sits in the log; its detail shows that same id. Its Sessions tab lists [agent sessions](../observability/traces.md) where the deployment records traces (`trace_capture_enabled`, on by default), and is not offered where it does not. Sessions applies the user, API key, model and status filters carried from the other tabs; a model filter matches the sessions with an LLM call on that model, and a status with no session meaning is shown as not applied rather than dropped. Narrowing the chart to a window that holds a single session opens it. Its chart counts sessions per hour or day as succeeded or failed, and its Levels column counts each session's errors, tools' errors included; a tool's error alone does not fail a session (see [Sessions and failures](../observability/traces.md#sessions-and-failures)). The Last activity, Levels, Latency, Tokens and Total cost headers sort the list across every page: a header's first press puts the largest first and a second press flips it, and the order is kept in the URL with the filters. A session opens in a side panel as a tree of its turns, LLM calls and tool calls, a timeline of the same spans, or a flat log. The token count in its header breaks down into input, the share of it the prompt cache served, and output, for the session or the span picked. Before a span is picked, the panel shows what the session cost against its key's own spend ceiling and where the gateway stepped in. A request in Requests whose session was recorded links to it: **View trace** in the request's detail opens the session on that LLM call. The Usage tab provides aggregates and time series; its spend by session groups requests by the same sessions, keyed by trace id. Routed requests share a request-group ID, so the request log can show the attempted models and the model that served the response. Imported usage is labeled by source and does not consume a budget.

Use Prometheus at `/metrics` for process-level monitoring when `enable_metrics` is enabled. The scrape needs the `metrics` extra (`pip install gateway[metrics]`); the Docker image installs it, and a source install that sets `enable_metrics` without it refuses to start rather than serving an empty scrape.

## Organization

Organizations own workspaces. The workspace switcher determines which keys, members, usage, tools, aliases, and stored routing policies are in view. The organization section manages resources shared across those workspaces.

What a signed-in person can see or change is enforced by their organization and workspace roles. Deployment-wide operations require an operator. See [Access control](../).

Invitations waiting on you are listed under Invitations, reached from the organization switcher. After you accept one, the page offers to take you into that organization. Someone signed in who is not yet a member of any organization is taken to Invitations when one is waiting; otherwise Overview says how to get in, and the sidebar holds only Overview. Creating an organization is offered to them only where `open_signup` is on, since the gateway refuses it otherwise.

Opening an invitation link in a browser signed in as a different address, or as an identity with no address, says who the invitation is for and offers to sign out and continue, because accepting always adds the invited address, whoever is signed in. After a password is set from an invitation link, the sign-in screen opens with the invited address filled in. When the link cannot set a password and the address has none yet, the page says how to get in instead: a Google or GitHub sign-in, or "Forgot your password?", whichever the deployment offers. The other pages in front of sign-in carry the address the same way: from the sign-in screen to "Forgot your password?", from signup or a resend to the page saying which address was mailed and on to another resend, and back to the sign-in screen after a verification or a recovery request. A password never travels with it.

On Members, an invited row offers Reissue and Revoke; see [Manage existing members](../organizations-and-access/members.md#manage-existing-members).

## Authentication options

Password sign-in works for an existing account. Optional passkeys, Google OAuth and GitHub OAuth add more ways to sign in, and where the deployment sets `open_signup: true` a new address can also create an account with its own organization, through the signup form or a Google or GitHub sign-in. Setup and the rules for each are in [Sign-in and accounts](https://docs.otari.ai/self-hosting/sign-in).

Where the deployment publishes terms (`terms_url`), the signup page asks for them before any of those, and the invitation page asks before "Accept and sign in with a provider instead". A provider sign-in from the sign-in screen that would set up an account without them stops on the page the provider returns to, which asks for the terms and starts the sign-in again.

A link to a dashboard page opened while signed out shows the sign-in screen first, then that page once you have signed in, whichever method you use.

Signing in can also add a membership, where an organization has proven the email domain of the identity's verified address; see [Email-domain auto-join](../organizations-and-access/members.md#email-domain-auto-join).

Mail is optional: when an invitation email does not go out, you get the accept link to share. See [Invitations](../organizations-and-access/members.md#invitations) and [Mail](https://docs.otari.ai/self-hosting/configuration#mail).

## Custom documentation

The dashboard's Documentation link opens docs.otari.ai unless `docs_url` names another site; see [Documentation links](https://docs.otari.ai/self-hosting/configuration#documentation-links).

## Legal pages

`terms_url` and `privacy_url` name where this deployment's terms of service and privacy notice live. The account menu links each row it has an address for; see [Configuration](https://docs.otari.ai/self-hosting/configuration#legal-pages).

## Development

The dashboard source is under `web/`. The published Docker image includes the bundle. A source checkout needs `make dashboard` for the gateway to serve it, or `pnpm --dir web dev` for frontend development. See [web/README.md](https://github.com/mozilla-ai/otari/blob/v0.37.0/web/README.md).
