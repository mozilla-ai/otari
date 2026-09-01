# Admin dashboard

Otari serves its dashboard at the gateway root. The dashboard uses the same
management API available to scripts, so access and behavior are determined by the
server rather than duplicated in the frontend.

## Deployment modes

- **Standalone** shows the local management UI and serves inference from the same
  process.
- **Hosted** shows the multi-tenant control plane. Inference snippets point to
  `data_plane_url`; the hosted process does not serve inference.
- **Hybrid** shows gateway health and a link to the otari.ai control plane. It has
  no local management UI.

The dashboard reads `GET /v1/bootstrap` before rendering. That response controls
the available sign-in methods and navigation surfaces, so a page that is not
available in the current mode is not offered.

## Sign-in and secrets

The master key and `OTARI_SECRET_KEY` are different credentials:

| Credential | Purpose |
| --- | --- |
| Master key | Authorizes deployment management and signs in the first operator |
| `OTARI_SECRET_KEY` | Encrypts provider and search-tool credentials stored in the database |

If `OTARI_MASTER_KEY` is unset on a new standalone database, Otari generates a
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
reset the operator password.

## First-run walkthrough

1. Start Otari in standalone mode.
2. Open the gateway root, usually `http://localhost:8000/`.
3. Sign in with the configured or generated master key.
4. Add a provider. If storing its credential in the dashboard, configure
   `OTARI_SECRET_KEY` first.
5. Test the provider connection or declare model IDs for a backend that has no
   model-listing endpoint.
6. Create an API key from the setup guide or the Keys page.
7. Send a request and confirm it appears under Activity.

The [Quickstart](quickstart.md) includes a complete request example.

## The setup guide

Overview offers a short setup flow until the selected workspace serves its first
successful gateway request. The key it creates is an ordinary workspace API key.
Skipping the guide hides it for that workspace; it does not revoke a key already
created. Set `activation_guide: false` to disable the flow for the deployment.

## Navigation

The workspace view contains day-to-day gateway operations:

- Overview, Activity, and Usage
- Models and Routing
- Tools
- API keys, providers, and workspace members

The organization view contains tenant-wide administration:

- Organization-wide usage, in hosted mode
- Workspaces and organization members
- Spend and budgets
- Organization pricing
- Organization settings and, in hosted mode, provider keys

Settings shows the effective non-secret configuration. Some values can be changed
at runtime and others require a restart. The server marks that distinction in the
settings response.

What a page shows can also depend on who is signed in, not only on the
deployment. Spend and budgets is the clearest case: an organization owner or
admin manages their own organization's budgets and the spend ceilings holding
them, while a deployment operator gets the deployment-wide budgets and the
gateway users assigned to them. Model pricing splits the same way, with the
default pricing catalog kept to an operator and the organization's own rate
overrides open to its admins.

Exact page names and availability can change with deployment mode and installed
extensions. The running dashboard is the source of truth.

What a page shows can also depend on who is signed in. On API keys, an operator
manages every key in the organization and chooses each key's owner, while a
member sees the same page scoped to their own keys, always billed to themselves
and never budget-exempt.

## Observability

Activity is the per-request log. Usage provides aggregates and time series.
Routed requests share a request-group ID, so Activity can show the attempted
models and the model that served the response. Imported usage is labeled by
source and does not consume a budget.

Use Prometheus at `/metrics` for process-level monitoring when
`enable_metrics` is enabled.

## Organization

Organizations own workspaces. The workspace switcher determines which keys,
members, usage, tools, aliases, and stored routing policies are in view. The
organization section manages resources shared across those workspaces.

What a signed-in person can see or change is enforced by their organization and
workspace roles. Deployment-wide operations require an operator. See
[Access control](access-control.md).

## Authentication options

Password sign-in is always tied to an existing identity. Optional passkeys,
Google OAuth, and GitHub OAuth add ways for that identity to sign in; they do not
make an unknown account a member. OAuth requires `public_base_url` plus the
provider's client ID and secret. Passkeys can instead use `public_base_url`, or
an explicit `webauthn_rp_id` and `webauthn_allowed_origins` pair.

Mail is optional. Without SMTP, invitation links can still be copied and shared
manually. See [Configuration](configuration.md#mail).

## Bundled guide and custom documentation

This guide is bundled into the dashboard at `/#/docs`. Set `docs_url` to make
the dashboard's Documentation link open a different site. The bundled route
remains available.

## Legal pages

`terms_url` and `privacy_url` name where this deployment's terms of service and
privacy notice live. The account menu links each row it has an address for; see
[Configuration](configuration.md#legal-pages).

## Development

The dashboard source is under `web/`. The published Docker image includes the
bundle. A source checkout needs `make dashboard` for the gateway to serve it, or
`pnpm --dir web dev` for frontend development. See [web/README.md](../web/README.md).
