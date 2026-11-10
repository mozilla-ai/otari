# Run the dashboard

A self-hosted gateway serves its dashboard at the gateway root. This page covers
what the operator sets up there: the sign-in and encryption keys, the first run,
the deployment view, and deployment-wide settings.

## Sign-in and secrets

| Credential | Purpose |
| --- | --- |
| Master key (`OTARI_MASTER_KEY`) | Authorizes deployment management and signs in the first operator |
| `OTARI_SECRET_KEY` | Encrypts provider and search-tool credentials stored in the database |

If `OTARI_MASTER_KEY` is unset on a new database, Otari generates a master key
and prints it once, inside a setup link:

```text
Otari 1.0.0 is running at http://localhost:8000

Open this link to choose your password and finish setup:

http://localhost:8000/#/setup?key=otari-mk-...

It is shown only once and carries your master key, so treat it like one.
```

That start prints nothing else unless something needs attention, and holds
its reminders about optional settings left unset until the next start. Later
starts print where Otari runs and where to sign in. `otari serve --log-level
debug` shows every startup line, and the access lines of successful health
probes, which are otherwise left out of the log. A deployment that generated its master key
creates no bootstrap API key (see
[Budgets and API keys in config](configuration.md#budgets-and-api-keys-in-config)); create the
first key in the dashboard.

The key is the part after `key=`. It sits in the link's fragment, which the
browser does not send to the server, and the setup page removes it from the
address bar as soon as it opens. The dashboard exchanges the key for an
HttpOnly session cookie and never stores it in the browser. Save the key if you
will call the management API with it, and treat the link as you would the key.

On SQLite, Otari generates the key that encrypts stored credentials and keeps
it in a file beside the database. On PostgreSQL, set `OTARI_SECRET_KEY` before
saving provider or search-tool credentials in the dashboard:

```bash
otari gen-secret-key
```

Losing the key makes stored credentials undecryptable. See
[Runtime provider management](configuration.md#runtime-provider-management)
for where the key lives and what it protects against. Providers configured in
`config.yml` do not need it.

Opening the setup link and choosing an email address and password claims a new
deployment. A deployment whose operator signed in with the master key instead
is claimed from **Account settings**. Once claimed, the sign-in page asks for
the password instead of the master key, and the setup link leads to the sign-in
page. The master key still works for the management API and can reset the
operator's password, but only the operator's. For a member who forgot theirs,
send a password link from **Accounts** (see
[Deployment-wide account administration](sign-in.md#deployment-wide-account-administration)).

Which sign-in forms the page offers is described in
[Dashboard sessions and identity](sign-in.md#dashboard-sessions-and-identity).

## First-run walkthrough

1. Start Otari. With Docker, keep its data in a named volume, since an
   anonymous volume does not reattach to a new container:

   ```bash
   docker run -v otari:/data -p 127.0.0.1:8000:8000 mzdotai/otari
   ```

2. Open the setup link from the log and choose an email address and password.
   If you set `OTARI_MASTER_KEY` yourself, no link is printed: open the gateway
   root, usually `http://localhost:8000/`, and sign in with that key.
3. Add a provider. On PostgreSQL, storing its credential in the dashboard needs
   `OTARI_SECRET_KEY` set first.
4. Test the provider connection, or declare model IDs for a backend with no
   model-listing endpoint.
5. Create an API key from the setup guide or the **Keys** page.
6. Send a request and check that it appears on the **Overview**, in
   **Requests**.

The [Self-hosted quickstart](self-hosted-quickstart.md) has a complete request
example.

## The deployment view

The deployment view, opened from the account menu, holds what belongs to the
gateway process rather than to an organization:

| Page | What it is | Details |
| --- | --- | --- |
| Settings | The running configuration, and the Rate limit rules card | [Settings](#settings) |
| Accounts | Every account on the deployment | [Deployment-wide account administration](sign-in.md#deployment-wide-account-administration) |
| Hosted providers | The provider keys the deployment serves inference on | [The Hosted providers page](hosted-providers.md#the-hosted-providers-page) |
| Subagent router | The decision models that pick the model a coding agent's subagent runs on | [Subagent router](#subagent-router), [Subagent routers](https://docs.otari.ai/reference/gateway/routing#subagent-routers) |
| Web search | The search and fetch tools the deployment runs for every workspace, and their prices | [Web search and Code execution pricing](#web-search-and-code-execution-pricing) |
| Code execution pricing | The price per successful `otari:code_execution` call | [Web search and Code execution pricing](#web-search-and-code-execution-pricing) |

Only a deployment operator sees Accounts, Hosted providers, Web search and Code
execution pricing; anyone else who opens them is told they are not available.
**Tools > Web search** links an operator to the Web search page.

### Settings

Settings shows the effective non-secret configuration and marks which values
change at runtime and which need a restart. The **Rate limit rules** card adds,
edits and removes limits on requests per minute, tokens per minute and requests
in flight. Rules from `config.yml` are listed read-only (see
[Rate limit rules](configuration.md#rate-limit-rules)).

### Subagent router

This page sets the decision models that pick a coding agent's subagent model
when the otari-router plugin for Claude Code, or pi-otari, calls
`POST /api/v1/routing/recommend`. It is the only place the router is set.

Each router has:

- a provider (TypeSafe, OpenRouter or a llama.cpp server), a model and an
  optional API base
- a key, stored encrypted and shown by its last four characters
- the price per million input and output tokens charged to the caller (both 0
  for a free router, such as a local model)

You can add several; one serves at a time. The first one added serves, and the
**Serving** switch moves it. With no router serving, no model is picked. See
[Subagent routers](https://docs.otari.ai/reference/gateway/routing#subagent-routers).

### Web search and Code execution pricing

**Web search** lists the search tools and the fetch tools (`builtin_fetch`
first), each with its key's last four characters and its price per call. Each
table ends with the `otari:web_search` or `otari:web_fetch` rate, which prices
a call on an organization's own key; `otari:web_search` also prices a search on
the legacy settings. A tool uses its own rate, or one stored under its old
`<provider>:<name>` model key; with neither, it is unpriced.

On a tool's row:

- Open the row for its endpoint, enrichment and options.
- The star makes it the default the model uses.
- **Test** runs one search or fetch and reports how many hits or characters
  came back.
- **Edit** and **Remove** change a stored tool. A `config.yml` tool,
  `builtin_fetch` and the two `otari:` rates take only a price; a rate from
  `price_per_call` is changed in `config.yml`.

**Add search tool** and **Add fetch tool** ask for a provider, then its key, an
endpoint if it has none of its own, and a price. **More settings** holds the
endpoint, timeout, enrichment and the provider's options, typed from the
provider catalog or written as JSON.

Below the tables, **The model's web tools** turns the model's search off and
sets the calls per request, results per search, reading pages for hits without
text, and running a provider's own web search here. The legacy SearXNG URL and
engines show while they are set, marked deprecated, with how to move them to a
`searxng` search tool (see
[The legacy SearXNG settings](configuration.md#the-legacy-searxng-settings)).

**Code execution pricing** lists `otari:code_execution`.

On both pages, **Edit** sets a rate in dollars per call, in force once saved;
0 makes a tool free. With `require_pricing` on, the page warns while a tool has
no rate, because a request that would run it is refused. See
[Pricing a gateway-run tool](https://docs.otari.ai/reference/gateway/tools#pricing-a-gateway-run-tool).

## The setup guide

Set `activation_guide: false` to turn off the Overview's
setup guide for the deployment.

## Signup and terms

With `open_signup: true`, a new address can create an account with its own
organization, through the signup form or a Google or GitHub sign-in. Someone
signed in who belongs to no organization is offered to create one. Otherwise
the gateway refuses both. The signup form also needs [mail](configuration.md#mail)
to send a verification link; Google and GitHub sign-in do not. Setup for each
sign-in method is in [Sign-in and accounts](sign-in.md).

With `terms_url` set, the signup page asks the person to accept the terms
first, and so does the invitation page before "Accept and sign in with a
provider instead". With `privacy_url` set too, the box links the privacy policy
beside the terms. A provider sign-in that would create an account without
them stops on the return page, asks for the terms, and starts the sign-in again.

Mail is optional: when an invitation email does not go out, the inviter gets
the accept link to share. See [Mail](configuration.md#mail).

## Metrics

With `enable_metrics` on, Prometheus metrics are served at `/metrics`. This
needs the `metrics` extra (`pip install gateway[metrics]`). The Docker image
has it; a source install without it refuses to start.

## Custom documentation

The dashboard's Documentation link opens docs.otari.ai unless `docs_url` names
another site; see [Documentation links](configuration.md#documentation-links).

## Legal pages

`terms_url` and `privacy_url` give the addresses of this deployment's terms of
service and privacy notice. The account menu links each one set; see
[Legal pages](configuration.md#legal-pages).

## Development

The dashboard source is under `web/`, and the Docker image includes the
bundle. From a source checkout, run `make dashboard` for the gateway to serve
it, or `pnpm --dir web dev` for frontend development. See
[web/README.md](https://github.com/mozilla-ai/otari/blob/c3d12d127592322bf66b75525926b8415d18ae00/web/README.md).
