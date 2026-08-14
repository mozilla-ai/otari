# Admin dashboard

Otari ships with a web admin dashboard for operators. It browses the model
catalogue, sets model pricing, manages routing policies, adds and edits provider API
keys, manages users, keys, and budgets, and toggles runtime settings, all
against the local management API using the master key.

The dashboard is a **standalone-mode** feature. In standalone mode Otari serves
the dashboard at the gateway root (`/`). In hybrid mode there is no local
management API, so the root keeps serving the get-started tutorial (`/welcome`)
instead. Everything below assumes standalone mode.

## The two-key model

The dashboard involves two separate secrets. They do different jobs, and
confusing them is the most common first-run snag, so it helps to keep them
straight.

| | Master key | `OTARI_SECRET_KEY` |
| --- | --- | --- |
| **Purpose** | Signs in to the dashboard and authorizes every management API call | Encrypts provider API keys stored through the dashboard (encryption at rest) |
| **Set via** | `OTARI_MASTER_KEY` (or `master_key` in `config.yml`); generated on first run if unset | `OTARI_SECRET_KEY` only; never generated for you |
| **Format** | Any string you choose, or a generated `otari-mk-…` value | A Fernet key (generate with `otari gen-secret-key`) |
| **Where it lives** | Only its SHA-256 hash is stored; the browser never keeps the key itself, just the session cookie it is exchanged for | Supplied out of band at runtime; never written to the database |
| **If you lose it** | Rotate or reset it; nothing else is affected | Every provider key stored in the dashboard becomes undecryptable |

A few consequences worth internalizing:

- **The master key is your dashboard password.** It gates every management route
  exactly like an operator-set key would. Anyone with it can read and change
  gateway configuration, so treat it like an admin credential.
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
once to this gateway and exchanged for a session cookie; the browser never stores
the key itself, so it cannot be read back out of the page. The sign-in lasts
`dashboard_session_ttl_hours` (a week by default) and survives closing the tab,
so you normally sign in once and not again. It survives restarting the gateway
too, as long as the gateway's database does: sessions are rows in it, so a
container running the default SQLite file with no mounted volume starts every run
signed out. If you are on a fresh install and are not sure where your key is, the
"First run? Where to find your key" hint on the sign-in screen points you back at
the logs.

### 5. Add a provider

Open **Providers** from the sidebar and add a provider (for example OpenAI),
pasting its API key. Stored keys are encrypted at rest with `OTARI_SECRET_KEY`,
and the API only ever echoes the last four characters back to the UI; the
plaintext key is write-only. Providers configured in `config.yml` also appear
here, marked `config` and read-only; keys you add in the UI are marked `stored`
and can be edited, tested, and deleted.

### 6. Test the connection

On the Providers page, use **Test the connection** for the provider you just
added. Otari makes a live call to confirm the credential works before you route
real traffic through it.

The check lists the provider's models, so a backend that does not implement a
`/v1/models` endpoint cannot be verified this way. That case is reported as
"No model discovery" rather than "Unreachable": the key may be perfectly good,
and the provider can still serve requests. Declare the model ids it serves under
that provider's `models:` key in `config.yml` to have them appear in the
catalogue. You can also price them one at a time from the dashboard, with no
config edit or restart: the Models page offers **Price a model** in the warning
it shows for a provider without discovery, and in its empty state when a search
finds nothing.

### 7. Send your first request

The Providers page includes a "Send your first request" snippet you can copy.
Point any OpenAI-compatible client at the gateway using an Otari API key or the
master key, and select a model in `provider:model` form (for example
`openai:gpt-4o`). See the [Quickstart](quickstart.md) for a full end-to-end
example.

### 8. (Optional) Set up keys, users, and budgets

For multi-user or multi-app deployments, use the **Access** section of the
sidebar to hand out scoped API keys, define users, and attach budgets so spend
is enforced before each call. These are optional: a single-operator setup can
run on the master key alone.

## Page-by-page reference

The sidebar groups pages by what they do. This section is filled in as pages
land; the groups below match the current dashboard.

### Overview

The landing page. An at-a-glance view of spend, traffic, and health across the
gateway.

### Observability

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

### Catalog

- **Providers**: add, edit, test, and delete provider credentials at runtime
  (standalone only). Stored keys are encrypted with `OTARI_SECRET_KEY`; config
  providers appear read-only. See the first-run walkthrough above. The add and
  edit forms also take **Client options (JSON)**, the `client_args` passed to the
  provider's client (a request timeout, custom headers); on the known-provider
  form they sit under Advanced. A backend that can take longer than 10 minutes to
  answer a non-streaming request needs an explicit `{"timeout": 1800}` here.
- **Models**: browse the model catalogue and set per-model pricing, with specs
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
- **Routing**: every named model your callers can send, in one place. A simple
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

- **Users**: the principals that keys and budgets attach to, including the
  default model access for a user's keys.
- **API keys**: issue and revoke gateway API keys, optionally restricting the
  models a key may call and setting an expiry (leave blank for a key that never
  expires).
- **Budgets**: spending limits callers are held to, with per-period resets.

For how users, keys, and budgets fit together and the management endpoints behind these pages, see [Access control](access-control.md).

### System

- **Tools & Guardrails**: configure the backends for built-in tools (for
  example the `otari_web_search` search backend) and request-level guardrails.
  Each tool Otari runs itself also carries a **price per call** here: those calls
  cost you money at a search provider or a sandbox, so they are billed onto the
  request that triggered them. An unpriced tool is refused with a 402 while
  `require_pricing` is on. See [Built-in tools](tools.md#pricing-a-gateway-run-tool).
  Each gateway-run tool also shows **how to call it**: the `tools[].type` values
  this deployment accepts and a request you can copy. Turning on
  `web_search_intercept` adds the provider-named keywords (`web_search`,
  `web_search_<date>`) to that list, which is what lets a client like Claude Code
  reach your search backend without knowing Otari's own tool name. See
  [Web-search interception](tools.md#web-search-interception).
  The **Search tools** card on the same page configures what
  [`POST /v1/search`](api-reference.md#search) can run against, which no longer
  needs a config file: add a `searxng` tool with no backend URL and it uses the
  web-search URL set just above it, so the backend you already run answers the
  direct search endpoint too. Tools declared in a config file are listed here as
  read-only; edit those where the file is defined.
- **Settings**: search and toggle runtime settings, review and apply default
  pricing updates, and rotate the generated master key. Rotating the master key
  issues a fresh `otari-mk-…` value and keeps your current session signed in.

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
  change gateway configuration through the management API. Rotate it if you
  suspect it leaked.
- **Use HTTPS for anything but local access.** The `http://localhost:8000/`
  examples here assume you are on the same machine (loopback). The master key
  authorizes every management request and must never travel over cleartext HTTP,
  so put the gateway behind HTTPS or a trusted reverse proxy before signing in
  from another host.
- **A session cookie, not a stored key.** The dashboard trades your master key
  for an HttpOnly cookie (`SameSite=Strict`, and `Secure` whenever the request
  arrives over HTTPS), so the key itself is never kept in the browser and script
  on the page cannot read the cookie. Signing out revokes the session on the
  server, expires the cookie, and clears any cached admin data. Rotating the
  master key revokes every session and re-mints the one you are using, so other
  signed-in browsers are logged out.
- **Sign out on a machine you share.** A session runs for its full
  `dashboard_session_ttl_hours` with no idle timeout, so an unattended browser
  stays signed in until the cookie expires. Use **Sign out** when you are done on
  a shared or public machine, or shorten `dashboard_session_ttl_hours`. Rotating
  the master key is the way to revoke a session you can no longer reach.
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
- [Modes](modes.md): standalone versus hybrid, and why the dashboard is
  standalone-only.
