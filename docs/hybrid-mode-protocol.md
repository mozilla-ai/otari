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
Otari ignores any it does not recognize. Consumers of this contract MUST do the
same (ignore unknown fields) so the platform can add fields without breaking
older gateways.

For example, the otari.ai resolve response also carries `provider_key_id` and
`allowed_models`. Otari does not read these today, so they are intentionally
absent from the shapes documented here. `user_id`, `workspace_id`, and
`organization_id` (all below) are the exceptions: Otari reads each when
present, but a peer omitting any of them is exactly as valid as a peer sending
an unrecognized field.

When the operator points Otari's web-search backend at the platform
(`OTARI_WEB_SEARCH_URL` under `base_url`), Otari also sends `X-Gateway-Token`
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

### Response: multi-attempt shape (preferred)

```json
{
  "request_id": "01HXY...",
  "fallback_enabled": true,
  "user_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "workspace_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "organization_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
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
    },
    {
      "attempt_id": "01HX3...",
      "position": 2,
      "provider": "bedrock",
      "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
      "api_key": "wJalrXUtnFEMI...",
      "api_base": null,
      "managed": false,
      "extra_params": {
        "region_name": "us-east-1",
        "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE"
      }
    }
  ]
}
```

Otari iterates `attempts` in order. On a provider failure before a response is
committed, it moves to the next entry; on success it stops. The `attempt_id` of
the entry that ultimately succeeded (or the last one tried, on total failure) is what Otari echoes
back via `X-Correlation-ID` and reports through `/gateway/usage`.

`extra_params` (optional, omitted for most providers) carries provider-specific
credential/client fields beyond `api_key`/`api_base`: for example AWS
Bedrock's mandatory `region_name` and, for the classic IAM-key-pair shape,
`aws_access_key_id` (the paired secret access key travels in `api_key`, since
that's the only field the wire contract treats as sensitive). This field is
sourced only from the trusted platform peer; Otari never accepts it from the
caller, and it can never be shadowed by a same-named field in the caller's
own request body: the same non-overridable treatment `api_key` and `model`
already get.

Otari does not merge `extra_params` into the completion call's kwargs
directly. any-llm's `acompletion()` only forwards a separate `client_args`
mapping to the provider's client constructor (everything else in its own
`**kwargs` goes to the completion call instead), so `extra_params` is nested
under `client_args` before the call. A flat merge would silently forward
`region_name` to the completion call rather than boto3's client constructor,
which is how an earlier version of this forwarding path still hit boto3's
`NoRegionError` despite carrying the right value end to end.

AWS Bedrock gets additional handling on top of that generic nesting, since
any-llm's Bedrock provider never reads a plain `api_key` when building its
boto3 client and AWS has two distinct credential shapes:

- Classic IAM access-key/secret-key pair (`aws_access_key_id` present in
  `extra_params`): the paired secret (`api_key`) is aliased into
  `client_args["aws_secret_access_key"]`, boto3's own constructor kwarg for it.
- Bearer token ("Bedrock API key", no `aws_access_key_id`): Otari builds a
  boto3 client itself, with signing disabled and the `Authorization: Bearer
  <token>` header injected via a `before-sign` event hook, and passes it as
  `client_args["client"]` (an any-llm-sdk `BedrockProvider` constructor
  parameter it already supports overriding its client with). The pinned
  boto3 version has no native support for `AWS_BEARER_TOKEN_BEDROCK` or an
  `aws_bearer_token` constructor kwarg, so this hook-based approach is the
  only way to authenticate this shape today; it is a per-request, in-process
  client, safe under concurrent load. See
  `src/gateway/services/bedrock_gateway_auth.py`.

`request_id` groups every `attempt_id` from the same resolve call so the
platform can attribute spend, render trace timelines, and emit fallback events.
Otari also surfaces it as the `X-Otari-Request-ID` response header.

`fallback_enabled` is informational, set by the platform when its routing
policy actually allows fallback (i.e. the policy has multiple enabled entries
and `fallback_enabled = true`). Otari uses `len(attempts) > 1` for its
own behavior.

`attempts` MUST contain at least one entry. An empty list is treated as a
platform bug and surfaced as `502 Bad Gateway`.

`user_id` (optional) identifies the platform identity that owns the
`X-User-Token` presented for this call, i.e. the token's creator. It is a
stable, opaque string; Otari does not assume any particular format and never
tries to parse it. Absent on older peers, which is exactly as valid as a peer
omitting any other unrecognized field: nothing that reads this field may
require it. When present, it is Otari's only way to key its own per-caller
gateway-side state (aliases, routing memory, files, batches) in hybrid mode.

`workspace_id` and `organization_id` (both optional) identify the tenant that
owns this resolution: the workspace the presented `X-User-Token` belongs to,
and the organization above it. Both are stable, opaque strings that Otari never
parses. They are the only tenant Otari sees, because `X-Gateway-Token`
authenticates the calling gateway and carries no tenant of its own, so nothing
downstream can recover them after the fact. Both are optional in the same
fail-open sense as `user_id`: a peer that omits them costs whatever record
needed a tenant, never the request. Otari records them and does not route on
them, so neither value can reach a provider call.

### Response: single-attempt shape

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
`fallback_enabled = false`) and behaves as it always has: no retry loop, errors
propagate to the client. New platform implementations should prefer the
multi-attempt shape. `user_id` has no legacy mirror here, the same treatment
as `extra_params`: a peer old enough to still emit this shape predates the
field entirely. `workspace_id` and `organization_id` are read here too, from
the same top level, since a peer that still answers flat may nonetheless know
its own tenant.

### Failure

| Status | Behavior |
|---|---|
| `400`, `401`, `402`, `403`, `404`, `429` | Status code is forwarded to the client; `429`'s `Retry-After` header is preserved. The `detail` is the platform's JSON `detail` string when present, otherwise the fallback `"Authorization request rejected"`. |
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

| Status | Behavior |
|---|---|
| `400`, `401`, `402`, `403`, `404`, `429` | Status code is forwarded to the client; `429`'s `Retry-After` header is preserved. The `detail` is the platform's JSON `detail` string when present, otherwise the fallback `"MCP server resolution failed"`. |
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
gateway itself via `OTARI_WEB_SEARCH_URL`, so Otari does not switch backends based on
this field.

### Failure

| Status | Behavior |
|---|---|
| `400`, `401`, `402`, `403`, `404`, `429` | Status code is forwarded to the client; `429`'s `Retry-After` header is preserved. The `detail` is the platform's JSON `detail` string when present, otherwise the fallback `"Web search resolution failed"`. |
| `422`, `5xx`                      | Mapped to `502 Bad Gateway` with `detail = "Authorization service unavailable"`. |
| Network/timeout                    | Mapped to `502 Bad Gateway`. |

> The resolve endpoints share the timeout (`PLATFORM_RESOLVE_TIMEOUT_MS`) and
> token headers with `provider-keys/resolve`. Their exact response shapes will
> become the contract of record once the consumer-side fixtures land
> ([#146](https://github.com/mozilla-ai/otari/issues/146)); until then this
> document is authoritative.

## Usage report

After every attempt, successful or failed, Otari sends:

```http
POST /gateway/usage
X-Gateway-Token: gw_...
Content-Type: application/json

{
  "correlation_id": "01HX1...",       // = the attempt_id from the resolve response
  "status": "success" | "error",
  "is_final_attempt": true,            // no later planned fallback will run
  "usage": {                           // present on success when usage is available
    "prompt_tokens": 13,
    "completion_tokens": 7,
    "total_tokens": 20,
    "cache_read_tokens": 8,            // provider cache-read input tokens
    "cache_write_tokens": 0           // cache-write (creation) input tokens; Anthropic only
  },
  "error_class": "http_401",           // optional on error; omitted when the
                                       // Otari can't classify the failure
                                       // (e.g. mid-stream errors). See below.
  "session_label": "my-run-personas"   // optional; the caller's cost-attribution
                                       // label (see below). Omitted when absent.
}
```

A successful attempt that completes without provider usage data still sends a
final report, but omits `usage` so the platform can record it as unavailable
rather than as an explicit zero-token result.

| Platform response | Gateway behavior |
|---|---|
| Completed success with `usage_status: "reported"` and non-null `pricing.source` | Validate the correlation id and attach `cost_usd` plus `pricing_source`. |
| Completed response without a price source, with unavailable usage, or for a failed attempt | Return the provider response without inline cost. An unpriced `0.000000` is a placeholder, not a priced zero. |
| `202` or legacy `204` | Return the provider response without inline cost. |
| Terminal 4xx, including `404` and `410` | Do not retry or attach inline cost. |
| Invalid body, correlation mismatch, reporting failure, or expired inline budget | Preserve the provider response and omit inline cost. |

### Inline response fields

| API format | Non-streaming placement | Streaming placement |
|---|---|---|
| Chat Completions | `usage.cost_usd`, `usage.pricing_source` | Terminal usage chunk: `usage.cost_usd`, `usage.pricing_source` |
| Messages | `usage.cost_usd`, `usage.pricing_source` | `message_delta.usage.cost_usd`, `message_delta.usage.pricing_source` |
| Responses | `usage.cost_usd`, `usage.pricing_source` | `response.completed.response.usage.cost_usd`, `response.completed.response.usage.pricing_source` |

The fields always appear together, and `pricing_source` is never null. The cost
is the platform's exact six-decimal string, and its source is `organization`,
`managed`, or `genai_prices`; Otari never calculates or converts it locally,
and settlement failure never fails the model response. A priced zero includes
both fields, while unavailable, pending, legacy, unpriced,
or carrier-less results include neither. Provider token usage is otherwise
unchanged, and standalone responses never include these fields. Use
`GET /request-costs/{request_id}` when a durable value is required.

`session_label` is an optional caller-supplied label for cost attribution (per
run, experiment, or conversation). A caller sets it on the request body
(`session_label` on the chat/messages/responses request); Otari strips it before
the upstream provider call and forwards it here so the platform can attribute the
attempt's spend to that session without the caller standing up OpenTelemetry. It
is trimmed and omitted when blank; Otari caps it at 255 characters at the request
boundary so the platform never has to truncate. All attempts of one request carry
the same label.

> **`user` is not used for cost attribution in hybrid mode.** The OpenAI-standard
> `user` field is stripped before the upstream call and is not forwarded on the
> usage report, so it does not segment spend here. Callers who want per-run cost
> breakdown must use `session_label`.

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

A multi-attempt request that iterates two attempts produces two usage reports,
one per attempt, sharing the same `request_id` (recoverable via the original
resolve response). The platform is responsible for correlating them.

`is_final_attempt` tells the platform that the gateway will not try another
planned fallback. It is `false` for a provider failure followed by another
attempt, and `true` for success, fallback exhaustion, a gateway-side refusal,
or a failure after a tool loop has locked in to one provider. The marker lets
the platform ignore later planned attempts that were never tried instead of
waiting forever for usage reports that will never arrive.

`error_class` is a short tag describing why the attempt was abandoned:

| Tag | Cause |
|---|---|
| `timeout` | `httpx.TimeoutException`, `asyncio.TimeoutError`, `TimeoutError`, or the OpenAI/Anthropic SDKs' own `APITimeoutError` |
| `conn_err` | `httpx.NetworkError`, or the OpenAI/Anthropic SDKs' own `APIConnectionError` |
| `http_<code>` | Provider returned an HTTP status code (e.g. `http_429`, `http_401`) |
| `http_<code>_billing` | Provider returned a 400 or 422 whose message identifies account billing exhaustion (e.g. Anthropic's "credit balance is too low"), or any 402 Payment Required, including one whose SDK discarded the response message. Reported as `http_400_billing` etc., keeping an empty provider wallet separable from a malformed request. Like every provider error before lock-in, it advances to the next attempt. |
| `unknown` | Any other exception class. It still advances to the next candidate. |

Treat `error_class` as an open set of strings, not a closed enum: new tags are
added as Otari learns to separate failure causes, and a handler that rejects an
unrecognized value would 422 the report, which is non-retryable and so silently
drops it.

Otari calls the OpenAI/Anthropic SDKs directly rather than httpx, and both SDKs
catch `httpx.TimeoutException`/network errors internally and re-raise as their
own `APITimeoutError`/`APIConnectionError`; neither is an instance of any
`httpx` exception, and neither carries `status_code`/`response`. Otari
recognizes these wrapped types explicitly (covering the majority of providers,
which reuse the OpenAI/Anthropic base provider classes), plus a conservative
class-name-based fallback (`*TimeoutError` / `*ConnectionError`, only when no
status code is present) for the other any-llm provider SDKs, so a real
"provider unreachable" or provider-side timeout still falls through to the
next attempt instead of being misclassified as `unknown`. If any-llm's unified
exceptions (`ANY_LLM_UNIFIED_EXCEPTIONS=1`) are enabled in the future, the same
detection still applies through `original_exception`: any-llm re-wraps a raw
SDK timeout/connection error into a generic `any_llm.exceptions.ProviderError`
that carries neither a recognizable class name nor a status code, but keeps
the original SDK exception on `.original_exception`, which Otari unwraps and
re-classifies.

The field is **omitted entirely** when Otari can't map the failure to an
exception class; this happens with mid-stream errors surfaced via the
SSE channel, where only an error string is available. Treat a missing
`error_class` as "uncategorised error" when aggregating.

### Retry semantics

Otari awaits the successful report at the response's terminal boundary for up
to `PLATFORM_USAGE_INLINE_TIMEOUT_MS`; expiry omits inline cost but leaves the
report running. Failed-attempt reports remain background work, so they can
arrive after the success report. Reports retry timeouts, network errors, and
5xx responses up to `PLATFORM_USAGE_MAX_RETRIES` with exponential backoff, but
not terminal 4xx responses.

## Streaming

Streaming requests (`stream: true`) iterate `attempts` just like non-streaming
requests, with one structural difference: **Otari can only fall through
before any bytes have been flushed to the client.** Once an attempt yields its
first chunk, Otari commits to that attempt; any further error
propagates to the SSE channel as today.

The mechanism is a per-attempt **first-chunk gate**. For each attempt:

1. Open the upstream stream (`acompletion(stream=True, ...)`). If this raises
   (provider returned `401` / `5xx` / network error before the stream even
   opened), record the error and move to the next attempt.
2. Wait for the first chunk with a bounded timeout. Non-final attempts use the
   per-attempt failover budget (`STREAMING_FALLBACK_FIRST_CHUNK_TIMEOUT_MS`,
   default 2000 ms). The sole/final attempt has no next attempt to fall over to,
   so it additionally gets `STREAMING_FALLBACK_FINAL_ATTEMPT_EXTRA_FIRST_CHUNK_TIMEOUT_MS`
   of grace on top of the budget (default 0). A request that forwards provider
   tools but does not run a gateway-managed tool loop keeps the tight budget on
   non-final attempts, while its final deadline is floored to the tool-loop
   budget (default 30000 ms). This gives tool-heavy agents more time only when
   there is nowhere left to fail over. If the upstream raises before yielding or
   the bounded wait times out, move to the next attempt.
3. Once a first chunk is in hand, commit. Stitch it back onto the iterator
   and start flushing SSE chunks to the client.

**Latency contract:** content streams immediately. In hybrid mode, only the
terminal usage suffix waits for settlement, bounded by
`PLATFORM_USAGE_INLINE_TIMEOUT_MS`, with keepalives during the wait. An early
carrier, timeout, mid-stream failure, or legacy `204` response skips inline cost
without reshaping the stream. Non-final attempts still cost at most the
first-chunk failover budget, and the final attempt at most that budget plus its
grace period.

Hybrid Chat Completions forces `stream_options.include_usage: true` so a
terminal carrier exists; standalone mode preserves the caller's value.

**What this catches:** auth errors (`401`/`403`), rate-limits (`429`),
upstream `5xx`, connection failures, hung connections, "stream opens but
errors before yielding."

**What this doesn't catch:** errors that arrive *after* the first chunk has
flushed (mid-stream connection drops, refusal messages embedded in normal
content chunks). These are out of reach without either prefix-buffering
(which would add visible latency on every request) or a client-cooperative
restart event (which would break OpenAI SDK compatibility).

Mid-stream failover is not currently planned. If a future client SDK starts
honoring a custom restart event, it could be added behind that capability
flag.

## Configuration

| Env var | Default | Notes |
|---|---|---|
| `OTARI_AI_TOKEN` | none | Setting this enables hybrid mode. |
| `PLATFORM_HEALTH_PATH` | `/utils/health-check/` | Path under `base_url` probed to report `platform_reachable` on `GET /health`; `GET /health/readiness` answers `503` when the same probe fails. Only a `2xx` counts as reachable, so point it at a route the peer actually serves: a `404`, a `401`, and a redirect to a login page all report unreachable. The default is an otari.ai route. |
| `PLATFORM_RESOLVE_TIMEOUT_MS` | `5000` | Per-resolve timeout. |
| `PLATFORM_USAGE_TIMEOUT_MS` | `5000` | Per-usage-report timeout. |
| `PLATFORM_USAGE_INLINE_TIMEOUT_MS` | `1500` | Budget for the one usage report the response path waits on to attach inline cost. Expiry ships the response without cost; the report itself continues. |
| `PLATFORM_USAGE_MAX_RETRIES` | `3` | Max retries for transient usage-report failures. |
| `STREAMING_FALLBACK_FIRST_CHUNK_TIMEOUT_MS` | `2000` | Per-attempt budget for the streaming first-chunk gate. Forwarded provider-tool requests retain it on non-final attempts. |
| `STREAMING_FALLBACK_FINAL_ATTEMPT_EXTRA_FIRST_CHUNK_TIMEOUT_MS` | `0` | Extra first-chunk grace for the sole/final attempt. A final request with forwarded provider tools gets at least the tool-loop budget (`30000` ms by default). |
