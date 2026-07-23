# Admin dashboard

Otari ships with a web admin dashboard for operators. It browses the model
catalogue, sets model pricing, manages aliases, adds and edits provider API
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
| **Where it lives** | Only its SHA-256 hash is stored; in the browser it is held in the tab's session storage | Supplied out of band at runtime; never written to the database |
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
sign-in screen. Paste your master key and select **Sign in**. The key is held
only in this browser tab's session storage and sent directly to this gateway; it
is never persisted to disk by the browser and is cleared when you sign out. If
you are on a fresh install and are not sure where your key is, the "First run?
Where to find your key" hint on the sign-in screen points you back at the logs.

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
- **Usage**: aggregate usage and analytics, showing spend and volume over time,
  broken down by model and by user.

### Catalog

- **Providers**: add, edit, test, and delete provider credentials at runtime
  (standalone only). Stored keys are encrypted with `OTARI_SECRET_KEY`; config
  providers appear read-only. See the first-run walkthrough above.
- **Models**: browse the model catalogue and set per-model pricing, with specs
  and modality metadata where available (from models.dev).
- **Aliases**: friendly names that resolve to a real provider model. Callers
  use the alias; the underlying model stays private to the gateway.

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
- **Settings**: search and toggle runtime settings, review and apply default
  pricing updates, and rotate the generated master key. Rotating the master key
  issues a fresh `otari-mk-…` value and keeps your current session signed in.

## Security notes

- **The master key is an admin credential.** Anyone who has it can read and
  change gateway configuration through the management API. Rotate it if you
  suspect it leaked.
- **Use HTTPS for anything but local access.** The `http://localhost:8000/`
  examples here assume you are on the same machine (loopback). The master key
  authorizes every management request and must never travel over cleartext HTTP,
  so put the gateway behind HTTPS or a trusted reverse proxy before signing in
  from another host.
- **Session storage, not local storage.** The dashboard keeps the master key in
  the browser tab's session storage, so it does not persist across tabs or
  survive closing the tab, and signing out clears it along with any cached admin
  data.
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
