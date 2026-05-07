# Platform protocol

When the gateway runs in **platform mode** (`OTARI_PLATFORM_TOKEN` is set), it
delegates per-request authorization and provider-credential resolution to a
peer platform service over HTTP. This document describes the wire contract
the gateway expects from that peer.

The default peer implementation is [otari](https://github.com/mozilla-ai/otari),
but any service that implements this contract can stand in.

## Endpoints

The gateway calls two endpoints, both rooted at `PLATFORM_BASE_URL`:

| Endpoint | Purpose |
|---|---|
| `POST {base}/gateway/provider-keys/resolve` | Authorize a request and return one or more provider credentials to try |
| `POST {base}/gateway/usage`                 | Report the outcome of an attempt back to the platform |

## Authentication

Both endpoints require `X-Gateway-Token: <gw_...>` in the request headers. This
proves the caller is the gateway instance configured against this platform
deployment. The resolve endpoint additionally requires `X-User-Token: <tk_...>`,
which is the workspace API token forwarded opaquely from the end user's
`Authorization: Bearer ...` header.

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

The gateway walks `attempts` in order. On a retryable failure it moves to the
next entry; on success it stops. The `attempt_id` of the entry that ultimately
succeeded (or the last one tried, on total failure) is what the gateway echoes
back via `X-Correlation-ID` and reports through `/gateway/usage`.

`request_id` groups every `attempt_id` from the same resolve call so the
platform can attribute spend, render trace timelines, and emit fallback events.
The gateway also surfaces it as the `X-Otari-Request-ID` response header.

`fallback_enabled` is informational — set by the platform when its routing
policy actually allows fallback (i.e. the policy has multiple enabled entries
and `fallback_enabled = true`). The gateway uses `len(attempts) > 1` for its
own behaviour.

`attempts` MUST contain at least one entry. An empty list is treated as a
platform bug and surfaced as `502 Bad Gateway`.

### Response — legacy single-attempt shape

For backwards compatibility with older platform deployments, the gateway also
accepts a flat payload:

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

The gateway maps this onto a single-attempt route (`attempts = [{...}]`,
`fallback_enabled = false`) and behaves as it always has — no retry loop, errors
propagate to the client. New platform implementations should prefer the
multi-attempt shape.

### Failure

| Status | Behaviour |
|---|---|
| `401`, `402`, `403`, `404`, `429` | Mapped through to the client as-is. `429`'s `Retry-After` header is preserved. |
| `422`, `5xx`                      | Mapped to `502 Bad Gateway` with `detail = "Authorization service unavailable"`. |
| Network/timeout                    | Mapped to `502 Bad Gateway`. |

## Usage report

After every attempt — successful or failed — the gateway sends:

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
    "total_tokens": 20
  },
  "error_class": "http_401"            // present on error only; see below
}
```

A multi-attempt request that walks two attempts produces two usage reports —
one per attempt — sharing the same `request_id` (recoverable via the original
resolve response). The platform is responsible for correlating them.

`error_class` is a short tag describing why the attempt was abandoned:

| Tag | Cause |
|---|---|
| `timeout` | `httpx.TimeoutException`, `asyncio.TimeoutError`, `TimeoutError` |
| `conn_err` | `httpx.NetworkError` |
| `http_<code>` | Provider returned an HTTP status code (e.g. `http_429`, `http_401`) |
| `unknown` | Any other exception class |

### Retry semantics

The usage endpoint is called as a background task on the gateway side. It
retries on transient failures (timeout, network error, 5xx) up to
`PLATFORM_USAGE_MAX_RETRIES` times with exponential backoff
(`0.25s`, `0.5s`, `1s`). It does **not** retry on `401`, `404`, `409`, `422` —
those are treated as terminal client errors.

## Streaming

Streaming requests (`stream: true`) only ever try `attempts[0]`. Mid-stream
failover would require either buffering the prefix (delaying the first byte)
or a custom SSE event signalling the client to discard the partial response.
Both options are out of scope for this version of the protocol. If the first
attempt fails for a streaming request, the error propagates to the client.

## Configuration

| Env var | Default | Notes |
|---|---|---|
| `OTARI_PLATFORM_TOKEN` | — | Setting this enables platform mode. Legacy alias: `ANY_LLM_PLATFORM_TOKEN`. |
| `PLATFORM_BASE_URL` | — | Required in platform mode. The gateway POSTs to `{base}/gateway/...`. |
| `PLATFORM_RESOLVE_TIMEOUT_MS` | `5000` | Per-resolve timeout. |
| `PLATFORM_USAGE_TIMEOUT_MS` | `5000` | Per-usage-report timeout. |
| `PLATFORM_USAGE_MAX_RETRIES` | `3` | Max retries for transient usage-report failures. |
