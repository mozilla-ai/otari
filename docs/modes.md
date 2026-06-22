# Modes

Otari operates in two modes: **standalone** and **connected to otari.ai**.

## Standalone

This is the default. Otari manages everything locally:

- **Database** -- stores API keys, users, budgets, usage logs (SQLite or PostgreSQL).
- **Provider credentials** -- configured in `config.yml` or via environment variables.
- **API key management** -- create and manage keys through `/v1/keys`.
- **User and budget controls** -- manage users and spending limits through `/v1/users` and `/v1/budgets`.
- **Usage tracking** -- all requests are logged locally and queryable through `/v1/usage`.

All endpoints are available. On first startup, Otari bootstraps an API key and logs it to the console.

## Connected to otari.ai

When connected to [otari.ai](https://otari.ai), Otari delegates provider routing, authentication, and usage tracking to the platform.

This mode activates automatically when `OTARI_AI_TOKEN` is set.

### What otari.ai handles

- **Provider routing** -- otari.ai resolves which provider and credentials to use for each request, including multi-provider fallback.
- **Authentication** -- requests are authenticated via bearer tokens issued by the platform.
- **Usage reporting** -- Otari reports token usage back to otari.ai after each request.
- **MCP server resolution** -- workspace-scoped MCP servers are resolved through the platform.

### What changes

- No local database is used.
- The `providers` block in `config.yml` must be empty (or absent).
- Only these routes are exposed: `/health`, `/health/liveness`, `/health/readiness`, and `/v1/chat/completions`.
- All other `/v1/*` routes are unavailable from this Otari instance.
- Chat requests use `Authorization: Bearer <otari-user-token>`.
- The `/health` endpoint includes platform reachability status.

### Setup

```bash
export OTARI_AI_TOKEN=gw_your_token_here
```

See [Deployment](deployment.md) for the full Docker setup.

## Comparison

| | Standalone | Connected to otari.ai |
|---|---|---|
| Database | Local (SQLite/PostgreSQL) | None |
| Provider credentials | In config or env vars | Managed by otari.ai |
| API key management | `/v1/keys` endpoints | Through otari.ai |
| User/budget management | `/v1/users`, `/v1/budgets` | Through otari.ai |
| Usage tracking | Local database | Reported to otari.ai |
| Multi-provider fallback | No | Yes |
| Available API routes | Full Otari API surface | Health + chat completions only |

## How Otari talks to otari.ai

When connected, Otari communicates with otari.ai through three internal endpoints:

### Provider resolution

Before each LLM request, Otari asks otari.ai which provider and credentials to use:

```
POST {base_url}/gateway/provider-keys/resolve
```

The platform returns an ordered list of attempts (provider, model, credentials). Otari tries them in order -- if one fails with a retryable error, it moves to the next.

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

For the full wire-level protocol specification, see [hybrid-mode-protocol.md](hybrid-mode-protocol.md).
