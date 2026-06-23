# Hybrid-mode protocol

When Otari runs in **hybrid mode** (`OTARI_AI_TOKEN` is set), it
delegates per-request authorization and provider-credential resolution to a
peer platform service over HTTP. This document describes the wire contract
Otari expects from that peer.

The reference peer is the [otari.ai](https://otari.ai) platform, but any service
that implements this contract can stand in.

## Endpoints

Otari calls these endpoints, all rooted at the configured platform base URL:

| Endpoint | Purpose |
|---|---|
| `POST {base}/gateway/provider-keys/resolve` | Authorize a request and return one or more provider credentials to try |
| `POST {base}/gateway/usage`                 | Report the outcome of an attempt back to the platform |
| `POST {base}/gateway/mcp-servers/resolve`   | Swap workspace-scoped MCP server ids for inline server configs (called only when a request references MCP server ids) |
| `POST {base}/gateway/web-search/resolve`    | Resolve the workspace's web-search policy (called only when a request uses the `otari_web_search` tool) |

`{base}` means Otari platform `base_url` setting. Otari concatenates literally. The peer service is responsible for including any API-version prefix it exposes its own routes under. For the reference otari deployment that prefix is `/api/v1`, so the base URL is `http://backend:8000/api/v1` and Otari ends up POSTing to `http://backend:8000/api/v1/gateway/provider-keys/resolve`.

## Authentication

Every endpoint requires `X-Gateway-Token: <gw_...>` in the request headers. This
proves the caller is an Otari instance configured against this platform
deployment. The three resolve endpoints additionally require `X-User-Token:
<tk_...>`, which is the workspace API token forwarded opaquely from the end
user's `Authorization: Bearer ...` header. The usage endpoint sends only the
gateway token.

## Extension policy

This document describes only the fields Otari actually reads. Every response
shape below is **open for extension**: a peer may return additional fields, and
Otari ignores any it does not recognise. Consumers of this contract MUST do the
same (ignore unknown fields) so the platform can add fields without breaking
older gateways.

For example, the otari.ai resolve response also carries `workspace_id`,
`organization_id`, `provider_key_id`, and `allowed_models`. Otari does not read
these today, so they are intentionally absent from the shapes documented here.

When the operator points Otari's web-search backend at the platform
(`GATEWAY_WEB_SEARCH_URL` under `base_url`), Otari also sends `X-Gateway-Token`
on its search queries (`GET {base}/gateway/web-search/search`) so a
platform-hosted search endpoint can authenticate the gateway. The token is sent
only when that URL shares the platform origin (scheme/host/port, under the base
path); it is never sent to a standalone or third-party search backend.

## Resolve

### Request

```http
POST /gateway/provider-keys/resolve
X-Gateway-Token: gw_...
X-User-Token: tk_...
Content-Type: application/json

{
  "model": "gpt-4o-mini",
  "provider": "openai"          // optional; otherwise inferred from model prefix
}
```

### Response — multi-attempt shape (preferred)

```json
{
  "request_id": "01HXY...",
  "fallback_enabled": true,
  "attempts": [
    {
      "attempt_id": "01HX1...",
      "position": 0,
      "provider": "anthropic",
      "model": "claude-sonnet-4-5",
      "api_key": "sk-ant-...",
      "api_base": null,
      "managed": false
    },
    {
      "attempt_id": "01HX2...",
      "position": 1,
      "provider": "openai",
      "model": "gpt-4o",
      "api_key": "sk-...",
      "api_base": "https://api.openai.com/v1",
      "managed": false
    }
  ]
}
```

Otari iterates `attempts` in order. On a retryable failure it moves to the
next entry; on success it stops. The `attempt_id` of the entry that ultimately
succeeded (or the last one tried, on total failure) is what Otari echoes
back via `X-Correlation-ID` and reports through `/gateway/usage`.

`request_id` groups every `attempt_id` from the same resolve call so the
platform can attribute spend, render trace timelines, and emit fallback events.
Otari also surfaces it as the `X-Otari-Request-ID` response header.

`fallback_enabled` is informational — set by the platform when its routing
policy actually allows fallback (i.e. the policy has multiple enabled entries
and `fallback_enabled = true`). Otari uses `len(attempts) > 1` for its
own behaviour.

`attempts` MUST contain at least one entry. An empty list is treated as a
platform bug and surfaced as `502 Bad Gateway`.

### Response — single-attempt shape

Otari also accepts a flat payload:

```json
{
  "provider": "openai",
  "model": "gpt-4o-mini",
  "api_key": "sk-...",
  "api_base": "https://api.openai.com/v1",
  "managed": true,
  "correlation_id": "01HXC..."
}
```

Otari maps this onto a single-attempt route (`attempts = [{...}]`,
`fallback_enabled = false`) and behaves as it always has — no retry loop, errors
propagate to the client. New platform implementations should prefer the
multi-attempt shape.

### Failure

| Status | Behaviour |
|---|---|
| `401`, `402`, `403`, `404`, `429` | Status code is forwarded to the client; `429`'s `Retry-After` header is preserved. The `detail` is the platform's JSON `detail` string when present, otherwise the fallback `"Authorization request rejected"`. |
| `422`, `5xx`                      | Mapped to `502 Bad Gateway` with `detail = "Authorization service unavailable"`. |
| Network/timeout                    | Mapped to `502 Bad Gateway`. |

## MCP server resolution

Called only when a request references one or more workspace-scoped MCP server
ids (a hybrid-only feature). Otari swaps those ids for the inline server
configs it needs to open the connections.

### Request

```http
POST /gateway/mcp-servers/resolve
X-Gateway-Token: gw_...
X-User-Token: tk_...
Content-Type: application/json

{
  "mcp_server_ids": ["01HX1...", "01HX2..."]
}
```

### Response

```json
{
  "servers": [
    {
      "name": "github",
      "url": "https://mcp.example.com/github",
      "authorization_token": "ghp_...",   // optional
      "purpose_hint": "Repo and issue lookups",   // optional
      "allowed_tools": ["list_issues", "get_file"] // optional
    }
  ]
}
```

Otari reads `name`, `url`, `authorization_token`, `purpose_hint`, and
`allowed_tools` off each entry in `servers`; a missing `servers` key is treated
as an empty list. The same URL-safety rules as inline MCP configs apply once the
configs are resolved (SSRF guard, no bearer token over cleartext `http://`).

### Failure

| Status | Behaviour |
|---|---|
| `401`, `402`, `403`, `404`, `429` | Status code is forwarded to the client; `429`'s `Retry-After` header is preserved. The `detail` is the platform's JSON `detail` string when present, otherwise the fallback `"MCP server resolution failed"`. |
| `422`, `5xx`                      | Mapped to `502 Bad Gateway` with `detail = "Authorization service unavailable"`. |
| Network/timeout                    | Mapped to `502 Bad Gateway`. |

## Web search resolution

Called only when a request uses the `otari_web_search` tool. The platform owns
the per-workspace web-search policy: whether it is enabled at all, plus the
workspace-default limits and filters.

### Request

```http
POST /gateway/web-search/resolve
X-Gateway-Token: gw_...
X-User-Token: tk_...
Content-Type: application/json

{}
```

The request body is empty; the workspace is identified by `X-User-Token`.

### Response

```json
{
  "enabled": true,
  "provider": "searxng",
  "max_results": 5,
  "purpose_hint": "Background research",
  "allowed_domains": ["example.com"],
  "blocked_domains": ["spam.example"],
  "provider_options": { "engines": "google,bing" }
}
```

If `enabled` is falsy, Otari rejects the request with `403`. The remaining
fields are workspace defaults that apply only where the request did not supply
its own value: `max_results`, `allowed_domains`, `blocked_domains`, and
`purpose_hint` fill in when the per-request tool entry omits them (an empty list
or empty string reads as "no preference" and does not clear the workspace
value), and `provider_options` is shallow-merged with per-request keys winning.
`provider` is informational: the active web-search backend is configured on the
gateway itself via `WEB_SEARCH_URL`, so Otari does not switch backends based on
this field.

### Failure

| Status | Behaviour |
|---|---|
| `401`, `402`, `403`, `404`, `429` | Status code is forwarded to the client; `429`'s `Retry-After` header is preserved. The `detail` is the platform's JSON `detail` string when present, otherwise the fallback `"Web search resolution failed"`. |
| `422`, `5xx`                      | Mapped to `502 Bad Gateway` with `detail = "Authorization service unavailable"`. |
| Network/timeout                    | Mapped to `502 Bad Gateway`. |

> The resolve endpoints share the timeout (`PLATFORM_RESOLVE_TIMEOUT_MS`) and
> token headers with `provider-keys/resolve`. Their exact response shapes will
> become the contract of record once the consumer-side fixtures land
> ([#146](https://github.com/mozilla-ai/otari/issues/146)); until then this
> document is authoritative.

## Usage report

After every attempt — successful or failed — Otari sends:

```http
POST /gateway/usage
X-Gateway-Token: gw_...
Content-Type: application/json

{
  "correlation_id": "01HX1...",       // = the attempt_id from the resolve response
  "status": "success" | "error",
  "usage": {                           // present on success only
    "prompt_tokens": 13,
    "completion_tokens": 7,
    "total_tokens": 20,
    "cache_read_tokens": 8,            // provider cache-read input tokens
    "cache_write_tokens": 0           // cache-write (creation) input tokens; Anthropic only
  },
  "error_class": "http_401"            // optional on error; omitted when the
                                       // Otari can't classify the failure
                                       // (e.g. mid-stream errors). See below.
}
```

`cache_read_tokens` and `cache_write_tokens` are additive fields carrying the
provider cached-token counts (default `0` when a provider reports none). Their
inclusion convention differs by provider, so the platform must price them with
that in mind:

- OpenAI (chat and Responses) and Gemini report cached tokens as a **subset** of
  `prompt_tokens`. `cache_read_tokens` is informational for re-pricing those
  tokens at the cached rate; there is no cache-write concept, so
  `cache_write_tokens` is always `0`.
- Anthropic reports `prompt_tokens` (mapped from `input_tokens`) **excluding**
  cache. `cache_read_tokens` and `cache_write_tokens` are reported **separately**
  and are not part of `prompt_tokens`. `cache_write_tokens` is a true cache
  creation charge billed at a premium.

The platform must accept these additive keys with lenient parsing; a handler that
rejects unknown fields would 422 the report (a non-retryable status), silently
dropping it. See companion issue mozilla-ai/otari-ai#1168.

A multi-attempt request that iterates two attempts produces two usage reports —
one per attempt — sharing the same `request_id` (recoverable via the original
resolve response). The platform is responsible for correlating them.

`error_class` is a short tag describing why the attempt was abandoned:

| Tag | Cause |
|---|---|
| `timeout` | `httpx.TimeoutException`, `asyncio.TimeoutError`, `TimeoutError` |
| `conn_err` | `httpx.NetworkError` |
| `http_<code>` | Provider returned an HTTP status code (e.g. `http_429`, `http_401`) |
| `unknown` | Any other exception class |

The field is **omitted entirely** when Otari can't classify the failure
back to an exception — this happens with mid-stream errors surfaced via the
SSE channel, where only an error string is available. Treat a missing
`error_class` as "uncategorised error" when aggregating.

### Retry semantics

The usage endpoint is called as a background task on Otari side. It
retries on transient failures (timeout, network error, 5xx) up to
`PLATFORM_USAGE_MAX_RETRIES` times with exponential backoff
(`0.25s`, `0.5s`, `1s`). It does **not** retry on `401`, `404`, `409`, `422` —
those are treated as terminal client errors.

## Streaming

Streaming requests (`stream: true`) iterate `attempts` just like non-streaming
requests, with one structural difference: **Otari can only fall through
before any bytes have been flushed to the client.** Once an attempt yields its
first chunk, Otari commits to that attempt; any further error
propagates to the SSE channel as today.

The mechanism is a per-attempt **first-chunk gate**. For each attempt:

1. Open the upstream stream (`acompletion(stream=True, ...)`). If this raises
   — provider returned `401` / `5xx` / network error before the stream even
   opened — classify the error: retryable failures move to the next attempt;
   non-retryable failures propagate.
2. Wait for the first chunk with a bounded timeout
   (`STREAMING_FALLBACK_FIRST_CHUNK_TIMEOUT_MS`, default 2000 ms). If the
   upstream raises before yielding or the wait times out, move to the next
   attempt.
3. Once a first chunk is in hand, commit. Stitch it back onto the iterator
   and start flushing SSE chunks to the client.

**Latency contract:** zero added latency in the success case — the first
chunk is held only for the microseconds it takes to call the SSE response
builder. In the failure case, each abandoned attempt costs at most
`first_chunk_timeout_seconds`.

**What this catches:** auth errors (`401`/`403`), rate-limits (`429`),
upstream `5xx`, connection failures, hung connections, "stream opens but
errors before yielding."

**What this doesn't catch:** errors that arrive *after* the first chunk has
flushed (mid-stream connection drops, refusal messages embedded in normal
content chunks). These are out of reach without either prefix-buffering
(which would add visible latency on every request) or a client-cooperative
restart event (which would break OpenAI SDK compatibility).

Mid-stream failover is not currently planned. If a future client SDK starts
honouring a custom restart event, it could be added behind that capability
flag.

## Configuration

| Env var | Default | Notes |
|---|---|---|
| `OTARI_AI_TOKEN` | — | Setting this enables hybrid mode. |
| `PLATFORM_RESOLVE_TIMEOUT_MS` | `5000` | Per-resolve timeout. |
| `PLATFORM_USAGE_TIMEOUT_MS` | `5000` | Per-usage-report timeout. |
| `PLATFORM_USAGE_MAX_RETRIES` | `3` | Max retries for transient usage-report failures. |
| `STREAMING_FALLBACK_FIRST_CHUNK_TIMEOUT_MS` | `2000` | Per-attempt budget for the streaming first-chunk gate. |
