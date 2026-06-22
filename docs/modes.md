# Modes

Otari operates in two modes: **standalone** and **connected to otari.ai**.

## Standalone

This is the default. Otari manages everything locally:

- **Database**: stores API keys, users, budgets, usage logs (SQLite or PostgreSQL).
- **Provider credentials**: configured in `config.yml` or via environment variables.
- **API key management**: create and manage keys through `/v1/keys`.
- **User and budget controls**: manage users and spending limits through `/v1/users` and `/v1/budgets`.
- **Usage tracking**: all requests are logged locally and queryable through `/v1/usage`.

All endpoints are available. On first startup, Otari bootstraps an API key and logs it to the console.

## Connected to otari.ai

When connected to [otari.ai](https://otari.ai), Otari delegates provider routing, authentication, and usage tracking to the platform.

This mode activates automatically when `OTARI_AI_TOKEN` is set.

### What otari.ai handles

- **Provider routing**: otari.ai resolves which provider and credentials to use for each request, including multi-provider fallback.
- **Authentication**: requests are authenticated via bearer tokens issued by the platform.
- **Usage reporting**: Otari reports token usage back to otari.ai after each request.
- **MCP server resolution**: workspace-scoped MCP servers are resolved through the platform.

### What changes

- No local database is used.
- The `providers` block in `config.yml` must be empty (or absent).
- Only these routes are exposed: `/health`, `/health/liveness`, `/health/readiness`, `/v1/chat/completions`, `/v1/messages`, and `/v1/responses`.
- All other `/v1/*` routes are unavailable from this Otari instance.
- Chat requests use `Authorization: Bearer <otari-user-token>`.
- The `/health` endpoint includes platform reachability status.

### Setup

```bash
export OTARI_AI_TOKEN=gw_your_token_here
```

See [Deployment](deployment.md) for the full Docker setup.

## Managed models vs. your own keys

Hybrid mode resolves provider credentials in one of two ways, and the difference decides who pays and where the request is allowed to run.

### Your own keys (BYO)

You store provider API keys in the otari.ai vault and assign them to a workspace. When a request arrives, otari.ai hands the matching key back to your Otari instance, which calls the provider directly. The provider bills you; otari.ai does not charge a wallet for these calls. BYO keys work through any Otari gateway, including one you self-host in hybrid mode.

In a request, a BYO model uses the `provider/model` form, such as `openai/gpt-4o` or `anthropic/claude-sonnet-4-6`, resolved against the key you configured for that provider. (Standalone mode uses the `provider:model` form instead. See [Use with Claude Code](use-with-claude-code.md) for the model-string conventions per mode.)

### Managed models

Managed models are served with mozilla.ai's own upstream credentials and billed to your otari.ai wallet. You reference them with the `mzai:` prefix (for example `mzai:moonshotai/Kimi-K2.6`) and never supply or see the upstream key.

Managed models are available only through the gateway that mozilla.ai operates as part of otari.ai. They are not served to a gateway you self-host. This is a deliberate security boundary: a self-hosted Otari instance is a process mozilla.ai does not control, so returning a platform-owned upstream key in the resolve response would expose that secret to whoever runs the instance. A self-hosted gateway can therefore use your BYO keys but not managed models.

If a self-hosted instance requests a managed model, otari.ai rejects the request with `403 ManagedKeyRequiresDefaultGatewayError`. To use managed models, send the request through otari.ai's hosted gateway instead.

## Comparison

| | Standalone | Connected to otari.ai |
|---|---|---|
| Database | Local (SQLite/PostgreSQL) | None |
| Provider credentials | In config or env vars | Your keys in the otari.ai vault, or mozilla.ai-managed models |
| API key management | `/v1/keys` endpoints | Through otari.ai |
| User/budget management | `/v1/users`, `/v1/budgets` | Through otari.ai |
| Usage tracking | Local database | Reported to otari.ai |
| Multi-provider fallback | No | Yes |
| Available API routes | Full Otari API surface | Health + chat completions only |

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
