# Modes

Otari operates in two modes: **standalone** and **connected to otari.ai**
(`hybrid mode`).

## Standalone

This is the default. Otari manages everything locally:

- **Database**: stores API keys, users, budgets, usage logs (SQLite or PostgreSQL).
- **Provider credentials**: configured in `config.yml` or via environment variables.
- **API key management**: create and manage keys through `/v1/keys`.
- **User and budget controls**: manage users and spending limits through `/v1/users` and `/v1/budgets`.
- **Usage tracking**: all requests are logged locally and queryable through `/v1/usage`.

All endpoints are available. On first startup, Otari bootstraps an API key and logs it to the console.

## Connected to otari.ai (Hybrid Mode)

When connected to [otari.ai](https://otari.ai), Otari delegates provider routing, authentication, and usage tracking to the platform.

This mode activates automatically when `OTARI_AI_TOKEN` is set. You can also set
`OTARI_MODE` explicitly to assert the intended mode: `OTARI_MODE=hybrid` requires
a token (startup fails without one), and `OTARI_MODE=standalone` with a token set
is rejected at startup as conflicting configuration (the token would otherwise
select hybrid). Leave `OTARI_MODE` unset to let the token decide.

`OTARI_AI_TOKEN` is the gateway token (`gw_...`) you create in otari.ai for
this Otari instance. In otari.ai, go to `Organisation > Gateways`, create or
open a gateway, then click `Create token`. It is not the per-request user token
(`tk_...`) that clients send in `Authorization: Bearer ...`.

Note the prefix: the platform gateway token uses an underscore (`gw_...`),
whereas a locally issued Otari API key (standalone mode) uses a hyphen
(`gw-...`). They are different credential types that differ by one character,
so take care not to confuse them.

### What otari.ai handles

- **Provider routing**: otari.ai resolves which provider and credentials to use for each request, including multi-provider fallback.
- **Authentication**: requests are authenticated via bearer tokens issued by the platform.
- **Usage reporting**: Otari reports token usage back to otari.ai after each request.
- **MCP server resolution**: workspace-scoped MCP servers are resolved through the platform.

### What changes on your Otari instance

- No local database is used for keys, users, budgets, or usage logs.
- The `providers` block in `config.yml` must be empty (or absent).
- Only these routes are exposed: `/health`, `/health/liveness`, `/health/readiness`, `/v1/chat/completions`, `/v1/messages`, and `/v1/responses`.
- Chat requests use `Authorization: Bearer <otari-user-token>`.
- The `/health` endpoint includes platform reachability status.

### Setup

```bash
export OTARI_AI_TOKEN=gw_your_token_here
```

See [Deployment](deployment.md) for the full Docker setup.

## Managed models vs. your own keys

In hybrid mode, a request can use either your own provider key or a
mozilla.ai-managed model. The practical differences are: whose credential is
used, who pays, what the model string looks like, and whether the request can
run through a self-hosted gateway.

| Option | Credential source | Billing | Model string | Works on a self-hosted gateway? |
|---|---|---|---|---|
| Your own keys (BYO) | Your provider key stored in otari.ai | The upstream provider bills you directly | `provider/model` (or `provider:model`) | Yes |
| Managed models | mozilla.ai-managed upstream key | Your otari.ai wallet | `mzai:...` | No |

### Your own keys (BYO)

Store a provider API key in otari.ai and assign it to a workspace. When a
request arrives, otari.ai returns that workspace key to your Otari instance,
which then calls the provider directly.

- The upstream provider bills you directly.
- This works through any Otari gateway, including one you self-host.
- Use model strings like `openai/gpt-4o` or `anthropic/claude-sonnet-4-6`.
- The `provider:model` form also works. See [Use with Claude Code](use-with-claude-code.md) for the model-string conventions per mode.

### Managed models

Managed models use mozilla.ai-managed upstream credentials. You never supply or
see the provider key yourself.

- Usage is billed to your otari.ai wallet.
- Use the `mzai:` prefix, for example `mzai:moonshotai/Kimi-K2.6`.
- These models are available only through the gateway that mozilla.ai operates as part of otari.ai.
- They are not served to a gateway you self-host, because that would expose mozilla.ai-managed upstream credentials outside infrastructure mozilla.ai controls.

If a self-hosted instance requests a managed model, otari.ai returns
`403 ManagedKeyRequiresDefaultGatewayError`. To use managed models, send the
request through otari.ai's hosted gateway instead.

## Comparison

| | Standalone | Connected to otari.ai |
|---|---|---|
| Database | Local (SQLite/PostgreSQL) | None |
| Provider credentials | In config or env vars | Your keys in the otari.ai vault, or mozilla.ai-managed models |
| API key management | `/v1/keys` endpoints | Through otari.ai |
| User/budget management | `/v1/users`, `/v1/budgets` | Through otari.ai |
| Usage tracking | Local database | Reported to otari.ai |
| Multi-provider fallback | No | Yes |
| Available API routes | Full Otari API surface | Health, chat completions, messages, and responses |

## How Otari talks to otari.ai

When connected, Otari communicates with otari.ai through four internal endpoints:

### Provider resolution

Before each LLM request, Otari asks otari.ai which provider and credentials to use:

```
POST {base_url}/gateway/provider-keys/resolve
```

The platform returns an ordered list of attempts (provider, model, credentials). Otari tries them in order; if one fails with a retryable error, it moves to the next.

### Usage reporting

After each attempt (success or failure), Otari reports usage:

```
POST {base_url}/gateway/usage
```

Reports include token counts, status, and any error information. Transient failures are retried with exponential backoff.

### MCP server resolution

When a request includes MCP server references, Otari resolves them:

```
POST {base_url}/gateway/mcp-servers/resolve
```

The platform returns server configurations (URL, auth token, allowed tools).

### Web search resolution

When a request enables Otari's built-in web search, Otari resolves the workspace's search policy:

```
POST {base_url}/gateway/web-search/resolve
```

The platform returns whether search is enabled and its configuration (provider, result limits, domain filters).

For the full wire-level protocol specification, see [hybrid-mode-protocol.md](hybrid-mode-protocol.md).
