# Admin dashboard

Otari ships with a web admin dashboard for operators. It browses the model
catalog, sets model pricing, manages routing policies, adds and edits provider API
keys, manages members, keys, and budgets, and toggles runtime settings, all
against the local management API.

The management pages are a **standalone-mode** feature. In standalone mode Otari
serves the whole dashboard at the gateway root (`/`). A gateway connected to
otari.ai serves the same page there, but it hosts no local management API, so
what it shows is a landing page: whether the gateway is healthy, whether it can
reach otari.ai, the base URL to point a client at, and a link to otari.ai, where
that gateway is managed. Everything below assumes standalone mode.

## The two-key model

The dashboard involves two separate secrets. They do different jobs, and
confusing them is the most common first-run snag, so it helps to keep them
straight.

| | Master key | `OTARI_SECRET_KEY` |
| --- | --- | --- |
| **Purpose** | Authorizes every management API call, and signs in to the dashboard until you set an operator password | Encrypts provider API keys stored through the dashboard (encryption at rest) |
| **Set via** | `OTARI_MASTER_KEY` (or `master_key` in `config.yml`); generated on first run if unset | `OTARI_SECRET_KEY` only; never generated for you |
| **Format** | Any string you choose, or a generated `otari-mk-…` value | A Fernet key (generate with `otari gen-secret-key`) |
| **Where it lives** | Only its SHA-256 hash is stored; the browser never writes the key to storage, keeping only the session cookie it is exchanged for | Supplied out of band at runtime; never written to the database |
| **If you lose it** | Rotate or reset it; nothing else is affected | Every provider key stored in the dashboard becomes undecryptable |

A few consequences worth internalizing:

- **The master key is an admin credential for the whole deployment.** It gates
  every management route, so anyone with it can read and change gateway
  configuration. It is also how you sign in to the dashboard on a new install,
  and it stops being that as soon as you give the operator identity an email
  address and a password (step 5 below). It stays the API credential either way:
  nothing you have scripted against it changes.
- **`OTARI_SECRET_KEY` is deliberately separate from the master key.** The
  gateway may rotate the master key; the encryption key must not move with it, or
  encryption at rest would be theatre against a stolen database. Otari never
  auto-generates it, never stores it next to the ciphertext, and never derives it
  from the master key.
- **You only need `OTARI_SECRET_KEY` to store provider keys in the dashboard.**
  If your providers are configured entirely in `config.yml`, you can run the
  dashboard without it. The moment you try to save a provider key in the UI, Otari
  needs it, and returns a clear "set `OTARI_SECRET_KEY` to store credentials"
  error if it is missing.

See [Configuration](configuration.md) for the full list of environment
variables and the [Runtime provider management](configuration.md#runtime-provider-management)
section for the underlying behavior.

## First-run walkthrough

This walks through going from a fresh gateway to a working request driven
entirely from the browser.

### 1. Start Otari in standalone mode

Launch the gateway however you normally would, for example:

```bash
uv run otari serve --config config.yml
```

or through Docker Compose (`docker compose up`). You do not need any providers
configured in `config.yml` up front; you can add them from the dashboard in a
later step.

### 2. Find your master key

If you set `OTARI_MASTER_KEY` (or `master_key` in `config.yml`), that is your
sign-in key and Otari never overrides it.

If you left it unset, Otari generates one on first startup, stores only its
hash, and prints the plaintext **once** to the logs. Look for the line:

```text
Your master key: otari-mk-…
```

For a container, `docker logs <container>` surfaces it. The plaintext is never
logged again, so copy it now. If you miss it, you can rotate to a new generated
key from the Settings page later (see below), or set `OTARI_MASTER_KEY`
explicitly and restart.

### 3. Set `OTARI_SECRET_KEY` before storing provider keys

If you plan to add provider API keys from the dashboard, set `OTARI_SECRET_KEY`
before you save the first one. Generate a Fernet key with:

```bash
otari gen-secret-key
```

Set the output as `OTARI_SECRET_KEY` in the gateway's environment and restart.
Keep it safe and separate from the database: losing it makes every stored
provider key undecryptable, and a database dump alone cannot decrypt them. You
can skip this step if all your providers live in `config.yml`.

### 4. Open the dashboard and sign in

Browse to the gateway root, for example `http://localhost:8000/`. You land on a
sign-in screen. Paste your master key and select **Sign in**. The key is sent
once to this gateway and exchanged for a session cookie; the browser never writes
the key to storage, so it is gone on reload and cannot be read back out of the
page afterwards. The sign-in lasts
`dashboard_session_ttl_hours` (a week by default) and survives closing the tab,
so you normally sign in once and not again. It survives restarting the gateway
too, as long as the gateway's database does: sessions are rows in it, so a
container running the default SQLite file with no mounted volume starts every run
signed out. If you are on a fresh install and are not sure where your key is, the
"First run? Where to find your key" hint on the sign-in screen points you back at
the logs.

### 5. (Optional) Claim the deployment with an email and a password

The master key signs you in because the operator identity has no password yet.
Giving it an address and a password makes that pair the sign-in from then on,
which is what you want as soon as more than one person needs the dashboard, or
as soon as you would rather not paste a deployment-wide credential into a
browser. Do it before you add anyone: until you claim, the sign-in screen offers
the master key rather than the password form, so a member who has signed up
cannot reach the dashboard from it.

In the dashboard, open the account control at the foot of the sidebar and select
**Account settings**. While the deployment is unclaimed that page asks for an
address and a password and nothing else, because there is no current password to
prove. Saving it claims the deployment; you stay signed in, and the page becomes
the ordinary change-password form.

The same call over HTTP, if you would rather claim from a terminal:

```bash
curl -X PUT http://localhost:8000/v1/auth/password \
  -H "Otari-Key: <master-key>" \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "new_password": "<a password>"}'
```

> **Claiming cannot be undone.** No endpoint clears a password, so a deployment
> that has been claimed cannot be put back to signing in with the master key.
> The sign-in screen switches to asking for the address and password, and the
> master key goes on authenticating the whole management API, so nothing becomes
> unreachable. It is simply a one-way step.

Three more things to know before you do it:

- **The master key stops being accepted at the sign-in screen**, and stays the
  credential for everything else. Scripts, CI jobs, and the `curl` examples in
  these docs are unaffected.
- **You cannot lock yourself out of the API.** If you forget the password, run
  the same command again with the master key and a new one; no
  `current_password` is needed when the master key is in the header. This is the
  one part with no dashboard equivalent, because reaching **Account settings**
  needs the session you have lost.
- **It signs out every browser holding a session for that identity**, including
  the one you claimed from if you claimed with `curl`.

Signup and password reset by email have since landed, so a member an admin adds
by address can set a password of their own. Setting one claims that member's
account and not the deployment, so it does not retire master-key sign-in: only
the operator identity's own password does that. **Account settings** also
manages
passkeys: sign in with your device instead of typing a password, once the
deployment knows its own address. Setting `public_base_url` is enough on its
own; `webauthn_rp_id` only overrides which domain the passkey is bound to and
still needs an address to serve the ceremony from. A
passkey is added to an identity that can already sign in rather than being a way
in of its own, and it never replaces the password. OAuth sign-in is the rest of
this track. See
[Access control](access-control.md#dashboard-sessions-and-identity) for the full
picture, including why a passkey is bound to one domain and what happens if that
domain changes.

### 6. Add a provider

Open **Providers** from the sidebar and add a provider (for example OpenAI),
pasting its API key. Stored keys are encrypted at rest with `OTARI_SECRET_KEY`,
and the API only ever echoes the last four characters back to the UI; the
plaintext key is write-only. Providers configured in `config.yml` also appear
here, marked `config` and read-only; keys you add in the UI are marked `stored`
and can be edited, tested, and deleted.

### 7. Test the connection

On the Providers page, use **Test the connection** for the provider you just
added. Otari makes a live call to confirm the credential works before you route
real traffic through it.

The check lists the provider's models, so a backend that does not implement a
`/v1/models` endpoint cannot be verified this way. That case is reported as
"No model discovery" rather than "Unreachable": the key may be perfectly good,
and the provider can still serve requests. Declare the model ids it serves under
that provider's `models:` key in `config.yml` to have them appear in the
catalog. You can also price them one at a time from the dashboard, with no
config edit or restart: the Models page offers **Price a model** in the warning
it shows for a provider without discovery, and in its empty state when a search
finds nothing.

### 8. Send your first request

Go back to **Overview**. Now that a provider exists, the page offers a setup
guide: press **Create a setup key** and it issues an API key for the selected
workspace and shows the two calls that use it, with the base URL and a model
filled in, ready to paste. Leave the page open and it watches the workspace's
traffic; when your request lands it says so, and if the request fails it names
the cause and the page that fixes it.

Nothing depends on the guide. Any OpenAI-compatible client pointed at this
gateway with an Otari API key (or the master key) and a model in
`provider:model` form does the same thing, and the
[Quickstart](quickstart.md) is the full end-to-end example. See
[The setup guide](#the-setup-guide) for what it records and how to turn it
off.

### 9. (Optional) Set up keys, people, and budgets

For multi-user or multi-app deployments, hand out scoped API keys from
**Access** on the workspace rail, then define budgets on the organization rail
and attach them to people or to a whole workspace, so spend is enforced before
each call. These are optional: a single-operator setup can run on the master key
alone.

## Page-by-page reference

The dashboard has two sidebars, and only one is on screen at a time.

The **workspace rail** is the default, and the switcher above it chooses which
workspace you are looking at. That selection scopes members, API keys, usage,
and the request log; routing and provider credentials are deployment-wide, and
the switcher's own popover says which is which. The same popover names the
organization above the workspaces, and where you belong to more than one it
switches between them; **Create organization** at its foot makes another, with
you as its owner and a default workspace inside it.

The **organization rail** holds what belongs to the tenant rather than to one
workspace. It is reached from the **Organization** entry at the foot of the
workspace rail, and left by the link at its top. Members, budgets, and settings
live there.

At the very bottom sits the account control, which holds **Account settings**
(the password you sign in with), an **Appearance** row that cycles through
system, light, and dark, the hosted legal pages where there are any, and **Log
out**. The bundled user guide is
**Documentation**, in the top bar; on a narrow screen, where the top bar has room
for the trail and nothing else, the account control carries it instead. A
deployment that has set `docs_url` points both of those at its own documentation
site instead, and this guide stays served at `/#/docs`; see
[Documentation links](configuration.md#documentation-links).

The groups below match the current dashboard, rail by rail.

### Overview

The landing page. An at-a-glance view of spend, traffic, and health across the
gateway.

#### The setup guide

Until a workspace has served a successful request, the Overview also carries the
guide from step 8 above. It is the second half of a pair: with no provider
configured, this page and the Providers page both say to add one, and once one
exists the guide takes over and offers the key and the call.

Four things are worth knowing about it:

- **The key it issues is an ordinary API key**, scoped to the selected
  workspace and listed on the API keys page as **Setup guide**. It is shown
  once, like every key this gateway mints, so opening the guide again issues a
  new one and retires the previous one. Issuing it is a workspace management
  action: an owner or admin of the workspace (or of the organization) sees the
  offer, and a member who only has read access does not.
- **"Skip this guide" is permanent, per workspace**, and it retires the card and
  nothing else. The offer does not come back on the next page load, for you or
  for a colleague, and the key stays exactly as it is: you asked for it, you may
  well have pasted it somewhere already, and revoking one is the API keys page's
  job.
- **Whether the workspace has activated is read from its own traffic**, not
  recorded separately: the first successful gateway request in the workspace is
  what closes the guide. Usage imported through
  `POST /v1/usage/external-events` deliberately does not count, since it was
  served by something else.
- **A deployment can turn the flow off** with `activation_guide: false` (or
  `OTARI_ACTIVATION_GUIDE=false`). The endpoints stay mounted and report every
  workspace ineligible, so a dashboard that is already open stops offering it
  too.

One consequence worth knowing if you are upgrading rather than starting fresh: a
workspace whose usage arrived only through `POST /v1/usage/external-events` has
never actually called this gateway, so the guide appears for it after the
upgrade. Skip retires it, or turn the flow off with the setting above.

### Observe

- **Activity**: the per-request log of what the gateway served, with filters.
  Use it to inspect individual requests, their models, and their outcomes.
  The table is a snapshot, and it holds still: it loads when you open the page and
  reloads when you ask it to, by pressing refresh or by changing a filter, the
  window, or the page. Nothing reorders itself while you are reading, which on a
  busy gateway is the difference between a log you can inspect and a feed you
  cannot. Two controls beside the refresh button carry what is happening
  meanwhile.
  **N in flight** counts the requests the gateway is serving right now, and opens
  the list of them: what model, for which user, and a wait that ticks up as you
  watch. A usage row is only written when a request settles, so without this a
  30-second call to a local model would read as nothing happening. The control is
  there only while something is running, with one exception: a list you have opened
  stays open when the last request lands, reading **0 in flight**, rather than
  vanishing at the moment you were waiting for. Close it and it goes. Completions,
  embeddings, image generations, audio, and searches all appear, each from the
  moment it clears the budget and access checks until its response has been fully
  sent (a streamed answer stays listed for as long as it is still producing
  tokens). Batches are the exception: the work runs on the provider's side after
  the submission returns, so there is no in-flight window to show. A request
  refused on budget, access, or model-resolution grounds never appears here: it was
  never running, and it lands in the log with its reason. A completion refused
  later, by an input guardrail or a bad tool declaration, can be listed for as long
  as that check takes, since the gateway really is working on it by then. The count
  is gateway-wide and is not narrowed by the filters above it: a request in
  progress has no outcome, cost, or token count for those filters to match on. It
  is read from the process that answers the poll, so a deployment running several
  otari processes behind a load balancer shows one process's traffic at a time, and
  which one it shows can change between polls. There is no deployment-wide total:
  the `gateway_active_requests` Prometheus metric is close but not the same number,
  as it counts every HTTP request a process is handling, dashboard polls included.
  **N new** appears once requests have landed since the page was drawn; press it to
  load them. It is a count only, so the table stays where you left it until you say
  otherwise. It is offered on the first page of a window that is still open, the
  only place where loading newer rows would bring them into view. If that count
  cannot be read, the strip says **Newer rows unknown** rather than falling silent,
  which would leave a busy gateway looking like an idle one. Paging re-reads the
  row total along with the rows, so a log that has grown since you opened it does
  not leave its oldest entries stranded past the end of the paginator.
  The **Routing** column names the policy a caller asked for, if any, plus where
  this row sits in that policy's plan and how it turned out: "served on attempt 2
  of 2 (a fallback candidate)", or, on an attempt a fallback recovered from,
  "attempt 1 of 2 failed, served by openai:gpt-4o", which names the model that
  served in its place.
  Expanding a routed row shows the whole **routing plan**: every candidate that
  ran, in order, with why it was selected, what it did, its cost, and the elapsed
  time when it settled (measured from the start of the request, as everywhere else
  in the log, so it is not a per-candidate duration), and the attempt that served
  marked. That is the place to answer "a fallback
  fired, so what actually served me", since each attempt is its own row.
  A row with the `absorbed` status is an attempt a policy recovered from
  by trying the next candidate: the request itself was served, so an absorbed row
  is deliberately not counted as an error and not counted as an extra request.
  That is what keeps a working fallback chain from reading as an outage in the
  error rate. Requests the gateway refused are logged too, so filtering to the
  `error` status shows what is being dropped: no pricing under `require_pricing`, a
  model outside a key's allow-list, a blocked or over-budget user, a `user`
  field that does not match the key, and a selector that no longer resolves to a
  configured provider. Those rows carry no cost, so they never move spend. Not
  every refusal is logged: a rejected API key (401) has no user to attribute the
  row to, and a rate-limited request (429) is an expected throttle rather than
  dropped traffic, so neither is recorded. Click an error row to see its
  diagnostic and the HTTP status that classified the failure, whether a fixed
  gateway rejection message or the raw upstream provider error. A row that
  carries no cost, whether the model has no price or the request was refused
  before it could be billed, offers **Price this model**, which sets that
  model's price from the exact selector the caller sent. Later requests are
  costed at those rates; rows already logged keep the cost they were served
  with.
  A request that used a tool Otari ran itself (`otari_web_search`,
  `otari_code_execution`, or an MCP tool) is marked next to its model with the
  number of calls, and the **Tool** filter narrows the log to one of them. The
  request detail lists the calls, how many failed, and what they cost. Query text
  is never stored: the log records counts and names only.
  The **User**, **Model**, and **API key** filters here take several values too, so
  a drill-down from Usage arrives intact and a comparison can be read as one list.
  The Model box also accepts a name that is not in its suggestions: press Enter to
  add it, since the suggestions only cover models with traffic in the window. When
  "select all N matching" is used for a bulk delete or reprice, the selection is
  scoped to exactly the values shown in the chips.
- **Usage**: aggregate usage and analytics, showing spend and volume over time,
  broken down by model and by user, plus a switchable breakdown by session,
  endpoint, provider, or source. The **User**, **Model**, and **API key** pickers
  each take several values, so a chart can compare a set ("these two models across
  this team's keys") rather than one entity at a time; every pick becomes its own
  chip, and the chip's ✕ removes just that value. Clicking any row opens the
  Activity log scoped to that group, carrying the whole selection with it, so
  "spend went up" leads straight to the requests behind it.
  When the window contains gateway-run tool calls, a **Gateway-run tools** table
  shows calls, failures, and spend per tool, so "what did search cost me last
  week" has an answer that is not one request at a time. MCP tools are excluded
  from that table: their names come from your own server, so they appear per
  request instead.
  The share icon in the chart's bottom-right corner turns the view into an image
  to post. It shares whatever the page is filtered to, so change the window or the
  filters to change what the card says, and the card names its own scope so a
  filtered figure cannot be read as the whole gateway. The dialog controls only how
  it looks: which stat leads, a title, square or wide, dark or light, how many
  model rows, and whether dollar amounts appear at all. Those choices are
  remembered; the data scope is always taken fresh from the page. Model names are
  shortened to the model itself, so a routed selector like
  `otari.ai:fireworks/accounts/deepseek-v4-flash` reads as `deepseek-v4-flash`. A
  spend figure is marked with an asterisk whenever the window holds requests with
  no configured price, and a stat the window has no value for is left off rather
  than published as a dash. From $100 up, a spend figure on the card is rounded to
  whole dollars: at the size the lead figure is set, the cents cost more width than
  they carry. The page's own tiles and tables keep them. Copy the image straight
  to the clipboard, or download it; copying needs a secure (https) origin, so on a
  plain-HTTP LAN address only the download is offered.

### Copying ids

Identifiers you have to paste somewhere else (a model id, an alias target, a
user id, a budget id, a request id) can be taken two ways: highlight the text
with the mouse as usual, or press the copy control beside it, which confirms with
a brief "Copied!" over the icon. The copy is the reliable one where the displayed
text is not the whole value: the Models table shows a name with the provider
prefix stripped, and the Budgets table shows only the first characters of a
budget id.

Copying works over plain HTTP, which is how a self-hosted dashboard is usually
reached. If a browser blocks the clipboard outright, the control says so rather
than reporting a copy that did not happen, and the text is still selectable by
hand.

### Gateway

- **Models**: browse the model catalog and set per-model pricing, with specs
  and modality metadata where available (from models.dev). The copy control next
  to a model puts its full `provider:model` id on your clipboard, which is what
  a caller sends as `model`; the name in the table drops the provider prefix.
  A provider that serves no `/v1/models` listing still answers requests, so a
  model you can call may never appear here on its own. Three places offer to
  price one by hand, all opening the same form: the warning shown for a provider
  without model discovery, the empty state when a search finds nothing (seeded
  with what you searched for, so searching the selector you just called is the
  quickest route), and **Price this model** in an Activity request detail. Give
  the selector callers send as `model`, prefix included (for example
  `vllm:mistral-small`), with its per-1M input and output rates. The model is
  then listed as custom ("not discovered"), its requests are costed, and its
  spend counts against budgets. Open **Edit price** on the new row afterwards for
  cache rates, the 1-hour cache rate, and long-context tiers. The same form
  re-prices a model that is already listed, so a key that already has a price
  replaces it.
- **Routing** expands on the sidebar into **Policies** and **Guardrails**.
  Guardrails is a view of the Tools settings, grouped here because a guardrail
  decides what a request may do rather than adding a capability to it.
  Policies is the page described below.
- **Policies**: every named model your callers can send, in one place. A simple
  one-target name (what used to be called an alias) still works exactly as
  before; a policy adds what to try when the first model fails, a tier-down to a
  cheaper model as a budget fills up, and guardrails a caller cannot skip.
  Aliases were folded into this page: stored ones were moved into policies by a
  migration, and any left in `config.yml` are listed here, read-only, tagged
  `alias`. "Serves" summarises the chain, and
  a `Dynamic` chip marks a policy whose choice depends on the request (so it has
  no single price). **Dry run** compiles the policy and shows the plan without
  sending anything to a provider or billing anything; it lists the candidates
  that were *dropped* as well as the ones kept, which is how you catch a fallback
  chain that has quietly filtered down to a single attempt. A policy from
  `config.yml` is read-only here. **Edit** lets you change the policy name, which
  renames it in place: callers have to send the new name from then on, and usage
  already recorded stays under the old one. Who a policy applies to is fixed once
  it exists, so moving one between scopes still means delete and recreate.
  See [Routing policies](routing.md).
  A policy can also hand its choice to a **router** that sends easy prompts to a
  cheaper model and keeps the strong one for the rest: open the policy form and use
  **Let a router pick the cheapest good-enough model**, then name the models it may
  choose between and mark the one that **serves when unsure**. That marked model is
  the policy's target, so there is one list rather than a separate "Serves" field:
  the fallback is always one of the models the router may pick. Those rows are tagged
  `Learned`.
  To spread load instead of choosing per prompt, use **Split traffic across providers
  by weight**: name the models and give each a **share**. Shares are relative, so 70
  and 30 mean the same as 7 and 3, and the form shows what each comes to as a
  percentage. A share of zero drains a provider without removing it: it takes no
  weighted traffic and still catches a failure. The marked model here is what serves a
  caller who sends `Otari-Router: off`, which is the one way a zero-share model still
  serves. Those rows are tagged `Weighted` and summarised by
  their split. See [weighted routing](routing.md#load-balance-across-providers-weighted-routing).
- **Examples**, on a learned policy's row: opens inline under that row and answers the
  question the table cannot. A router chooses nothing until it has scored examples, and
  until then the policy serves its default target on every request, which looks exactly
  like a broken router. This panel names the pool, says which model serves when the
  router declines, and shows how many examples each pool has against how many it needs.
  Pick whose memory first: the examples are one user's own prompts, so a policy every
  caller shares warms once per caller. Recording examples is an API job in this release
  (`POST /v1/routing/preferences/rank`); the panel links to the recipe. It is offered
  for `config.yml` policies too, since reading readiness is safe for a policy this page
  cannot change, and not at all for a weighted policy, which has nothing to teach. See
  [learned routing](routing.md#let-a-router-choose-learned-routing).

### Access

- **API keys**: issue and revoke gateway API keys, optionally restricting the
  models a key may call and setting an expiry (leave blank for a key that never
  expires). A key belongs to one workspace, the one selected above, and every
  request on it is billed there.
- **Providers**: listed here because a provider credential is what a key
  spends upstream. Add, edit, test, and delete provider credentials at runtime
  (standalone only). Stored keys are encrypted with `OTARI_SECRET_KEY`; config
  providers appear read-only. See the first-run walkthrough above. The add and
  edit forms also take **Client options (JSON)**, the `client_args` passed to the
  provider's client (a request timeout, custom headers); on the known-provider
  form they sit under Advanced. A backend that can take longer than 10 minutes to
  answer a non-streaming request needs an explicit `{"timeout": 1800}` here.
- **Members**: who is assigned to the selected workspace and their role in it.
  A workspace's members are always a subset of the organization's, so someone
  joins the organization first, on the organization rail.

Budgets moved to the organization rail; see below. For how people, keys, and
budgets fit together and the management endpoints behind these pages, see
[Access control](access-control.md). The API still calls a person a `user`,
which is the field name a caller sends and the one those endpoints use; the
dashboard shows that person as a member of the organization.

### Tools

One page per service, each a filtered view of the same settings. **Tools** on
the sidebar expands to:

- **Web search**: the backend behind `otari_web_search`, plus the **Search
  tools** card that configures what
  [`POST /v1/search`](api-reference.md#search) can run against. That endpoint no
  longer needs a config file: add a `searxng` tool with no backend URL and it
  uses the web-search URL set just above it, so the backend you already run
  answers the direct search endpoint too. Turning on `web_search_intercept`
  makes the provider-named keywords (`web_search`, `web_search_<date>`)
  acceptable too, which is what lets a client like Claude Code reach your search
  backend without knowing Otari's own tool name. See
  [Web-search interception](tools.md#web-search-interception).
- **Code execution**: the sandbox backend that runs generated code.

**Guardrails** is the third view, and it sits under Gateway → Routing rather
than here, because a guardrail decides what a request may do rather than adding
a capability to it. Under the deployment-wide settings there, **Organization
guardrails** is the layer above them: an entry runs on every request from the
workspaces you scope it to whether the caller asked for it or not, and an entry
marked for every workspace covers ones created later. It may name an https guardrails
endpoint of its own, with a credential to authenticate to it, or leave both
blank and use the URL set just above. See [Guardrails](guardrails.md#organization-guardrails).

Two things are true of every one of these views:

- Each tool Otari runs itself carries a **price per call**. Those calls cost you
  money at a search provider or a sandbox, so they are billed onto the request
  that triggered them, and an unpriced tool is refused with a 402 while
  `require_pricing` is on. See
  [Built-in tools](tools.md#pricing-a-gateway-run-tool).
- Each gateway-run tool shows **how to call it**: the `tools[].type` values this
  deployment accepts, and a request you can copy.

Tools declared in a config file are listed read-only; edit those where the file
is defined.

## The organization rail

Reached from the **Organization** entry at the foot of the workspace rail, and
left by the link at its top.

The tenant this deployment's workspaces, members, and roles belong to. One is
provisioned on first boot, and for most deployments that is the only one: a
self-hosted gateway is one organization with several people in it. Where there
is more than one, the switcher above the workspace rail is what moves between
them and creates another; **General** renames the one you are in and offers no
delete.

The master key is the bootstrap credential: the first authenticated request
provisions the organization, one default workspace, and one owner identity, and
every later request resolves that same operator. Giving that operator an email
address and a password is what turns it into a sign-in rather than a label, and
what retires the master key as the dashboard login (step 5 of the walkthrough
above). Members added after that hold a role and can be placed in workspaces, but
nothing here sets a password for someone else: a member claims their own
address from the sign-in screen instead, through **Added to this gateway?
Claim your account**, and confirms it by following the emailed link. The
address they are added by is the handle that claims it. **Forgot your
password?** on the same screen mails a reset link. Both need this deployment
to be able to send mail, and with none configured the links are absent and
neither flow has a fallback: a member cannot claim an address or recover a
password on a gateway that cannot mail them a link. The master-key recovery in
[Access control](access-control.md#dashboard-sessions-and-identity) is not that
fallback, because `PUT /v1/auth/password` always acts on the caller's own
identity: it is how the operator gets back in, and it sets nobody else's
password. Configure mail before you expect members to sign in.

### People & access

- **Workspaces**: create, rename, and delete workspaces. The last workspace
  cannot be deleted. Rosters are not managed here: a workspace's members are
  **Members** on the workspace rail, for whichever workspace the switcher has
  selected, and the organization's are **Members & roles**. Two roster pages,
  one per scope, and each picks the roles for its own.

  The create and edit forms carry a **Default member budget**, and the list has
  a column naming it: pick a budget and every member of that workspace is held
  to it, each with an allowance of their own rather than a shared pool. Someone
  in two workspaces therefore holds two, one per workspace. A budget is optional
  here and everywhere, and a key can be marked exempt from budget enforcement
  entirely, so neither a workspace default nor a person's own budget is a
  guarantee that every request is capped.

  Changing the default applies to members who join afterwards; members already
  there keep the budget they were given. Editing that *budget* is the retroactive
  act, and it moves everyone held to it. The edit form also manages defaults
  narrowed to one provider, which apply on that provider instead of the one
  above.
- **Members & roles**: who belongs to the organization, their role
  (owner, admin, member, viewer), and their status. Adding someone directly
  takes an email address and optionally the workspaces to put them in straight
  away, with the membership active immediately. Inviting instead leaves the
  membership `invited` until they follow an emailed accept link (see
  [Configuration](configuration.md#mail) for what mail needs to actually
  send); with none configured, the invite is still created and the dashboard
  hands you the link to share yourself. A pending invitation can be revoked
  before it is accepted.

  The roster also carries what a person may spend and what their keys may call.
  **Model access** is the default a key issued to them inherits, which that key
  can narrow but never exceed; **Workspaces** names the ones they are in and the
  budget, if any, they hold in each; **Spend** is what they have spent, plus
  anything held in flight by a request whose cost is not settled yet; and
  **Block** stops their keys making requests without removing them from the
  organization or touching their history.

  All of that lives on a gateway identity the member is linked to, and the link
  is optional: a member added by address before any key was issued to them has
  none yet, and one whose identity was deleted has none any more. Those rows show
  a role and a status, empty access and spend cells rather than zeros, and cannot
  own a key until an identity exists for them.

  **Edit** opens all of it at once: model access, which workspaces they are in
  and at what role, and the budget they hold in each. A budget is picked, never
  an amount: what a cap is worth and how often it resets are properties of the
  budget, so editing one moves everyone held to it, and giving one person a
  different limit means giving them a different budget. Those are three tables
  underneath, so saving is several writes, and they are ordered: a workspace
  budget is attached to the membership, so joining a workspace happens before the
  budget for it can be set. Adding someone to a workspace that has a default
  member budget gives them that budget unless another is chosen.

### Cost & billing

- **Spend & budgets**: the only page that says what a limit is worth. A budget
  is an amount and a reset period, and everything that enforces one names it
  rather than restating the figure, so editing a budget moves every place it
  applies.

  How it is enforced depends on what it is attached to. Assigned to people from
  this page, everyone on it is held to the full amount separately, so five people
  on one budget hold five allowances. Attached to a workspace or to one person's
  membership of one, it is a single allowance everyone under that scope draws on
  together. **Default for** names any workspace that hands this budget to its
  members, which is set on the workspace rather than here.
- **Model pricing**: what the gateway meters a request at, which is one rate per
  model for the whole deployment. The page opens with what an *unpriced* model
  costs, because that decides what the table under it means: with default pricing
  on, the table is the models you have overridden, and with it off, the table is
  everything that can be billed and `require_pricing` decides whether anything
  else is refused (HTTP 402) or served for free. Below that, **Check for price
  updates** fetches the upstream `genai-prices` catalog and shows what would
  change before anything is saved; custom prices are never touched by it. A single
  model's rate is still edited beside the model, on Models.

  Under the catalog, **Rate overrides** is what this organization pays instead,
  for the models it has negotiated. Each override names a model, its per-million
  rates, and the period they apply for; a request in that period is billed at
  them, and a model with no override is priced by the catalog above. Two
  overrides for one model may not cover the same moment, so an overlapping period
  is refused rather than quietly taking precedence, and the end of a period is
  exclusive, which is what lets the next one start at exactly that moment.
  Deleting an override returns the model to the catalog rate from the next
  request; usage already billed keeps the cost it was charged. Owners and admins
  can change these; every member can read them.

### General

- **Org settings**: rename the organization you are acting in. Creating another, and moving between them, is the switcher above the workspace rail.
- **Settings**: search and toggle runtime settings, and rotate the generated
  master key. Rotating the master key issues a fresh `otari-mk-…` value and keeps
  your current session signed in. **Email delivery** at the bottom of the page
  reports which transport is selected and whether it is ready to send, and sends
  a test message. Note that `console` is a configured transport that writes each
  message to the gateway log and delivers to nobody, so "ready" there means
  ready to log. Mail is optional, so a gateway with none configured says which
  settings would turn it on (see [Configuration](configuration.md#mail)) and
  disables the test send rather than offering one that would fail.
  **Maintenance mode**, below it, freezes new dashboard sign-ins so you can
  redeploy without anyone starting a session mid-migration. It is deliberately
  narrow: sessions already open keep working, so it never signs you out of the
  tab you flipped it in, and it does not touch the API, so keys and completions
  carry on serving. The switch is stored rather than held in memory, so one
  toggle freezes every replica at once and the freeze survives a restart. The
  way back out does not depend on signing in: the master key still authenticates
  the management API through the header, so you can turn it off from a fresh
  browser even while sign-ins are frozen. Keep that key to hand before you set
  this, because it is what lifts the freeze once your own session is gone;
  without it the recovery is setting `OTARI_MASTER_KEY` and restarting.

## Install it on your phone

The dashboard ships a web app manifest and app icons, so you can keep it on a
phone home screen instead of hunting for a tab.

- **Android (Chrome)**: open the dashboard, then **⋮** → **Add to Home screen**
  or **Install app**.
- **iOS (Safari)**: open the dashboard, then **Share** → **Add to Home Screen**.

It launches without browser chrome, under the Otari icon and the name "Otari".
On iOS the installed app keeps its own cookie storage, so you sign in to it once,
separately from Safari; an Android install shares Chrome's session.

Installing needs HTTPS, or `http://localhost` / `http://127.0.0.1` for local
access. Those loopback addresses are the only HTTP origins browsers treat as
secure, so a gateway reached over plain HTTP at a LAN address or hostname gets a
plain bookmark shortcut rather than an installed app. That is one more reason to
put it behind HTTPS, as the security notes below describe.

## Security notes

- **The master key is an admin credential.** Anyone who has it can read and
  change gateway configuration through the management API, and can set the
  operator's dashboard password without knowing the old one. Rotate it if you
  suspect it leaked.
- **Use HTTPS for anything but local access.** The `http://localhost:8000/`
  examples here assume you are on the same machine (loopback). The master key
  authorizes every management request, and a sign-in password travels in the same
  request bodies, so neither must ever cross cleartext HTTP: put the gateway
  behind HTTPS or a trusted reverse proxy before signing in from another host.
- **A session cookie, not a stored credential.** The dashboard trades whichever
  credential you signed in with for an HttpOnly cookie (`SameSite=Strict`, and
  `Secure` whenever the request arrives over HTTPS), so neither the master key
  nor your password is ever written to browser storage, and script on the page
  cannot read the cookie that replaces them. Signing out revokes the session on the server, expires the cookie,
  and clears any cached admin data. Rotating the master key revokes every session
  and re-mints the one you are using, so other signed-in browsers are logged out;
  changing a password does the same for that identity's own sessions.
- **A session is signed in as someone.** It names the operator identity it was
  minted for, which is what the Organization pages scope themselves to; see
  [Access control](access-control.md#dashboard-sessions-and-identity). Deleting
  or deactivating that identity signs its browsers out immediately rather than
  when the cookie expires. A deactivated identity's sessions are also discarded
  rather than held, on the next request that presents one, so re-activating it
  does not hand a cookie that was refused in the meantime its access back.
- **Log out on a machine you share.** A session runs for its full
  `dashboard_session_ttl_hours` with no idle timeout, so an unattended browser
  stays signed in until the cookie expires. Use **Log out** when you are done on
  a shared or public machine, or shorten `dashboard_session_ttl_hours`. Rotating
  the master key, or setting a new password with it, is the way to revoke a
  session you can no longer reach.
- **Provider keys are write-only over the API.** Once stored, the plaintext is
  never returned; the UI shows only the last four characters. Losing
  `OTARI_SECRET_KEY` makes stored keys undecryptable, so back it up separately
  from the database and rotate it by prepending a new key (see
  [Configuration](configuration.md#runtime-provider-management)).

## See also

- [Configuration](configuration.md): every environment variable and config
  field, including `OTARI_MASTER_KEY` and `OTARI_SECRET_KEY`.
- [Quickstart](quickstart.md): get the gateway running and make your first
  request.
- [Modes](modes.md): standalone versus hybrid, and why the management pages are
  standalone-only.
