# Run the dashboard

A self-hosted gateway serves its dashboard at the gateway root. This page covers
what the deployment's operator sets up and sees there: the keys that sign in and
encrypt, the first run, the deployment view, and the settings that change what
every signed-in person is offered. What any signed-in person sees is in
[Admin dashboard](dashboard.md).

## Sign-in and secrets

The master key and `OTARI_SECRET_KEY` are different credentials:

| Credential | Purpose |
| --- | --- |
| Master key | Authorizes deployment management and signs in the first operator |
| `OTARI_SECRET_KEY` | Encrypts provider and search-tool credentials stored in the database |

If `OTARI_MASTER_KEY` is unset on a new database, Otari generates a
master key and prints it once. The dashboard exchanges it for an HttpOnly session
cookie; it does not store the master key in browser storage.

Set `OTARI_SECRET_KEY` before saving provider or search-tool credentials through
the dashboard:

```bash
otari gen-secret-key
```

Keep this key separate from the database. Losing it makes stored credentials
undecryptable. Providers configured in `config.yml` do not require it.

The operator can claim a new deployment by setting an email address and password
from Account settings. After that, the sign-in page uses the password rather than
the master key. The master key remains valid for management API calls and can
reset the operator password. It resets the operator's password only: to help a
member who forgot theirs, send them a password link from Accounts, which emails
it where mail is configured and otherwise gives you the link to share (see
[Deployment-wide account administration](sign-in.md#deployment-wide-account-administration)).

When the sign-in page offers the password form, and when it offers the
master-key box beside it, is described in
[Dashboard sessions and identity](sign-in.md#dashboard-sessions-and-identity).

## First-run walkthrough

1. Start Otari.
2. Open the gateway root, usually `http://localhost:8000/`.
3. Sign in with the configured or generated master key.
4. Add a provider. If storing its credential in the dashboard, configure
   `OTARI_SECRET_KEY` first.
5. Test the provider connection or declare model IDs for a backend that has no
   model-listing endpoint.
6. Create an API key from the setup guide or the Keys page.
7. Send a request and confirm it appears on the Overview, in Requests.

The [Self-hosted quickstart](self-hosted-quickstart.md) includes a complete request example.

## The deployment view

The deployment view, opened from the account menu, holds what belongs to the
gateway process rather than to any organization on it:

| Page | What it is | Rules |
| --- | --- | --- |
| Settings | The running configuration, and the Rate limit rules card | [Settings](#settings) |
| Accounts | Every account on the deployment | [Deployment-wide account administration](sign-in.md#deployment-wide-account-administration) |
| Hosted providers | The provider keys the deployment serves inference on | [The Hosted providers page](hosted-providers.md#the-hosted-providers-page) |
| Subagent router | The decision models that pick the model a coding agent's subagent runs on | [Subagent router](#subagent-router), [Subagent routers](routing.md#subagent-routers) |
| Web search | The search and fetch tools the deployment runs for every workspace, and what each call costs | [Web search and Code execution pricing](#web-search-and-code-execution-pricing) |
| Code execution pricing | What the deployment charges per successful call of `otari:code_execution` | [Web search and Code execution pricing](#web-search-and-code-execution-pricing) |

Accounts and Hosted providers are listed only for a deployment operator, and
answer anyone else that they are not available to them. Web search and Code
execution pricing are listed for one too, and say the same to anyone else who
opens them. Tools > Web search links an operator to the Web search page.

### Settings

Settings shows the effective non-secret configuration. Some values can be changed
at runtime and others require a restart. The server marks that distinction in the
settings response. Its Rate limit rules card adds, edits and removes limits on
requests per minute, tokens per minute and requests in flight; rules from
config.yml are listed there read-only (see
[Rate limit rules](configuration.md#rate-limit-rules)).

### Subagent router

Subagent router lists the decision models that pick the model a coding agent's
subagent runs on when the otari-router plugin for Claude Code, or pi-otari,
asks `POST /api/v1/routing/recommend`. A router is a provider (TypeSafe,
OpenRouter or a llama.cpp server), a model, an optional API base, a key stored
encrypted and shown by its last four characters, and the price per million
input and output tokens the caller is charged. Several may be added and one
is switched on at a time: the first one added is on, and the Serving switch
moves it. Both prices at 0 make a free router, such as a local model. With
none added, or every router switched off, no model is picked. This page is
the only place the router is set. See
[Subagent routers](routing.md#subagent-routers).

### Web search and Code execution pricing

Web search holds the search and fetch tools the deployment runs for every
workspace and what each call costs. Its two tables list the search tools and
the fetch tools, `builtin_fetch` first, each with its key's last four and its
price per call, and end with the `otari:web_search` or `otari:web_fetch` rate,
which prices a call on an organization's own key, and `otari:web_search` a
search on the legacy settings too. A tool is priced at its own rate, or at one
stored under its old `<provider>:<name>` model key, and a tool with neither is
unpriced. A row opens its endpoint, enrichment and options; its star
makes it the default the model uses, and Test runs one search or fetch and
says how many hits or characters came back. Add search tool and Add fetch tool
pick a provider first, then ask for what it needs (its key, an endpoint when it
has none of its own) and a price, with the endpoint, timeout, enrichment and
the provider's options under More settings, typed from the provider catalog or
written as JSON. Edit and Remove change a stored tool; a `config.yml` tool,
`builtin_fetch` and the two `otari:` rates take only a price, and a rate from
`price_per_call` is changed in `config.yml`. Below the tables, The model's web
tools turns the model's search off, and sets the calls a request may make, the
results per search, reading pages for hits without text, and running a
provider's own web search here. The legacy SearXNG backend's URL and engines
show while they are set, marked deprecated, with how to move them to a
`searxng` search tool (see
[The legacy SearXNG settings](configuration.md#the-legacy-searxng-settings)).

Code execution pricing lists `otari:code_execution`.

On both pages, Edit sets a rate in dollars per call, in force from the moment
it is saved, and 0 prices a tool as free. With `require_pricing` on, the page
warns while a tool it lists has no rate, because a request that would run it
is refused. See [Pricing a gateway-run tool](tools.md#pricing-a-gateway-run-tool).

## The setup guide

Set `activation_guide: false` to disable the Overview's
[setup guide](dashboard.md#the-setup-guide) for the deployment.

## Signup and terms

Where the deployment sets `open_signup: true`, a new address can create an
account with its own organization, through the signup form or a Google or
GitHub sign-in, and someone signed in who belongs to no organization is offered
to create one. The gateway refuses both otherwise. The signup form also needs
[mail](configuration.md#mail), since it sends a verification link; a Google or
GitHub sign-in does not. Setup and the rules for each sign-in method are in
[Sign-in and accounts](sign-in.md).

Where the deployment publishes terms (`terms_url`), the signup page asks for
them before any of those, and the invitation page asks before "Accept and sign
in with a provider instead". A provider sign-in from the sign-in screen that
would set up an account without them stops on the page the provider returns to,
which asks for the terms and starts the sign-in again.

Mail is optional: when an invitation email does not go out, the inviter gets
the accept link to share. See [Mail](configuration.md#mail).

## Metrics

Use Prometheus at `/metrics` for process-level monitoring when
`enable_metrics` is enabled. The scrape needs the `metrics` extra
(`pip install gateway[metrics]`); the Docker image installs it, and a
source install that sets `enable_metrics` without it refuses to start rather
than serving an empty scrape.

## Custom documentation

The dashboard's Documentation link opens docs.otari.ai unless `docs_url` names
another site; see [Documentation links](configuration.md#documentation-links).

## Legal pages

`terms_url` and `privacy_url` name where this deployment's terms of service and
privacy notice live. The account menu links each row it has an address for; see
[Configuration](configuration.md#legal-pages).

## Development

The dashboard source is under `web/`. The published Docker image includes the
bundle. A source checkout needs `make dashboard` for the gateway to serve it, or
`pnpm --dir web dev` for frontend development. See [web/README.md](../web/README.md).
