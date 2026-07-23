# Importing external usage

Otari tracks the requests it routes. Subscription-backed coding agents such as
[Claude Code](use-with-claude-code.md) never hit the gateway, so their usage is
invisible to Otari even though it represents real model consumption. The external
usage API lets you import those events so they show up in your usage analytics
priced at API-equivalent rates, answering "what would my subscription usage have
cost at API prices?" and giving a team admin one dashboard across every channel.

Imported usage is **real cost, attributed to a user, but never enforced**: it is
recorded and shown in cost analytics, but it does not reserve budget, mutate a
user's spend, or gate live traffic (a retrospective event cannot be reserved
before the fact). See [Budget behavior](#budget-behavior) below.

This is a standalone-mode feature; hybrid mode has no local usage database.

## Authentication and attribution

The endpoint accepts **either an API key or the master key**, and usage binds to
the authenticated principal, just like a normal gateway request:

- **An API key** attributes every event to that key's own user, and stamps the
  key's id on the rows. You can omit `user_id` entirely; if you send one, it must
  match the key's user (a different user is rejected, unless
  `reject_user_mismatch: false`). This is the recommended path: hand each importer
  a scoped API key rather than putting the master key in an adapter or collector.
  **The key must be budget-exempt** (`exclude_from_budget: true`); a budgeted key is
  refused with a 403. Imported usage is retrospective and can never be blocked, so
  it must not run through a budget it could silently exceed (see
  [Budget behavior](#budget-behavior)).
- **The master key** is the admin path. It may name any user via the batch
  `user_id` (or a per-event `user_id`), so one importer can attribute a mixed feed
  to many users. Rows imported this way carry no api_key id.

The other usage endpoints (`list`, `count`, `summary`, `csv`) remain master-key
only, because they return every user's usage.

## Hello world (one curl)

Create the user and an importer key (both need the master key once), then import
with the **API key** and read it back. No collector, no infrastructure.

```bash
export OTARI_URL="http://localhost:8000"
export OTARI_MASTER_KEY="your-master-key"

# 1. The user must exist (imported cost is attributed to a real Otari user).
curl -sS "$OTARI_URL/v1/users" \
  -H "Otari-Key: Bearer $OTARI_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"user_id": "alice"}'

# 2. Mint a budget-exempt importer key bound to that user (grab the "key" from the
#    response). Import keys MUST be budget-exempt (imported usage can't be enforced).
curl -sS "$OTARI_URL/v1/keys" \
  -H "Otari-Key: Bearer $OTARI_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"key_name": "claude-code-importer", "user_id": "alice", "exclude_from_budget": true}'
export OTARI_KEY="gw-...."   # the key from the response

# 3. Import one usage event with the API key. No user_id needed: it binds to the
#    key's user.
curl -sS "$OTARI_URL/v1/usage/external-events" \
  -H "Otari-Key: Bearer $OTARI_KEY" -H "Content-Type: application/json" \
  -d '{
    "source": "claude_code",
    "events": [{
      "source_event_id": "req_01ABC",
      "timestamp": "2026-07-22T12:34:56Z",
      "provider": "anthropic",
      "model": "claude-sonnet-4-6",
      "status": "success",
      "input_tokens": 1200,
      "output_tokens": 450,
      "cache_read_tokens": 8000,
      "cache_write_tokens": 1024,
      "session_label": "project:otari"
    }]
  }'
# -> {"accepted":1,"duplicate":0,"rejected":0,"errors":[]}

# 4. Verify it landed (reading usage needs the master key).
curl -sS "$OTARI_URL/v1/usage?source=claude_code" \
  -H "Otari-Key: Bearer $OTARI_MASTER_KEY"
```

Use a recent `timestamp` if you want the row to show under the Activity page's
"24h" filter; a backdated event only appears under wider windows.

If the model has configured pricing at the event's timestamp, the row is priced;
if not, it still lands with `cost: null` (imported usage is budget-exempt, so
pricing is optional here, see [Pricing](#pricing)). Add pricing later to price
future imports.

## The endpoint

```http
POST /v1/usage/external-events
Otari-Key: Bearer <api key or master key>
Content-Type: application/json
```

The body is a batch that shares a `source` and a default `user_id`:

| Field | Required | Notes |
| --- | --- | --- |
| `source` | yes | Provenance slug, e.g. `claude_code`. Generic: add your own sources. |
| `user_id` | with master key | Default attribution. Optional with an API key (binds to the key's user); required with the master key. Must be an existing user. |
| `events` | yes | 1 to 1000 events. |

Each event:

| Field | Required | Notes |
| --- | --- | --- |
| `source_event_id` | yes | Upstream event id. Idempotency key together with `source`; the scope is global per source (not per user), so one collector and per-user importers can share a feed without double-counting. Use real upstream ids, which are unguessable. |
| `timestamp` | yes | ISO-8601. Used to resolve the effective price. |
| `provider`, `model` | yes | Priced as `provider:model` (falls back to `provider/model`). |
| `status` | no | `success` (default) or `error`. |
| `input_tokens`, `output_tokens` | no | Non-negative. Default 0. |
| `cache_read_tokens`, `cache_write_tokens`, `cache_write_1h_tokens` | no | Anthropic-style additive cache counts. `cache_write_1h_tokens` is the subset of `cache_write_tokens` written with a 1-hour TTL; the remainder is billed at the 5-minute rate. |
| `cache_tokens_in_prompt` | no | Token convention. `false` (default): `input_tokens` excludes the cache counts (Anthropic / Claude Code shape). `true`: cached tokens are a subset of `input_tokens` (OpenAI shape), and the price de-includes them instead of double-charging. |
| `duration_ms` | no | Wall-clock, recorded as the row's latency. |
| `session_label` | no | Optional session/project attribution. |
| `user_id` | no | Per-event override of the batch default (one collector feed can serve many users). |

The endpoint accepts only metadata and numeric usage. Any other field (a prompt,
a completion, tool input or output) is rejected with a 422; prompt and completion
text are never accepted or stored.

The response reports what happened:

```json
{ "accepted": 3, "duplicate": 1, "rejected": 1,
  "errors": [ { "index": 4, "source_event_id": "req_bad", "detail": "user_id 'ghost' not found. Create the user via POST /v1/users first." } ] }
```

### Idempotency

Rows are unique on `(source, source_event_id)`. Re-submitting a batch (a retry, an
overlapping poll) counts prior events as `duplicate` and never creates a second
row, so an at-least-once pipeline is safe.

### Errors

Rejected events carry `problem + cause + fix`, for example:

- Unknown user: `user_id 'ghost' not found. Create the user via POST /v1/users first.`
- A content field: `Field 'prompt' is not accepted. This endpoint ingests content-free usage events only...`
- Oversized batch: over 1000 events is rejected; split into chunks.

## OpenTelemetry (any GenAI app)

Any application instrumented for GenAI telemetry can ship usage to Otari over OTLP;
it lands as imported usage, priced at Otari's rates, budget-exempt, idempotent, and
content-free. Two signal endpoints, one mapping:

```text
POST /v1/traces    spans (what most GenAI instrumentation emits)
POST /v1/logs      log events (what Claude Code emits)
```

Point the exporter's endpoint at the Otari root; it appends `/v1/traces` or
`/v1/logs` itself. Both **protobuf and JSON** are accepted (optionally gzip), over an
**http** protocol (`OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf` or `http/json`; gRPC is
not accepted). Authenticate with a budget-exempt API key via
`OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer <key>`. The master key is refused
here: OTLP events carry no user attribution, so usage always binds to the key's
user (to import a mixed feed for many users, use `/v1/usage/external-events`).

Otari reads only the content-free usage attributes, preferring the
**OpenTelemetry GenAI semantic conventions**:

| Attribute | Stored as |
| --- | --- |
| `gen_ai.provider.name` (or `gen_ai.system`) | provider |
| `gen_ai.request.model` (or `gen_ai.response.model`) | model |
| `gen_ai.response.id` (else the span id) | source_event_id (dedup key) |
| `gen_ai.usage.input_tokens` | input |
| `gen_ai.usage.output_tokens` | output |
| `gen_ai.usage.cache_read_tokens` (or `cached_tokens`) | cache read |
| `gen_ai.usage.cache_write_tokens` | cache write |
| `otari.client_name` | `source` (provenance) |
| `otari.user_session_label` / `otari.session_label` | session label |

A record with no model/provider/tokens (a non-LLM span, a prompt log, a metric) is
skipped, so no prompt or response content is ever stored.

Anthropic-shaped emitters report cache reads/writes additively (outside
`input_tokens`); OpenAI-shaped emitters report cached tokens as a subset of
`input_tokens`. Otari treats the `anthropic` provider as additive and everything
else as OpenAI-shaped, so the price de-includes cached tokens rather than charging
them at both the input and cache-read rate.

### Recognized coding agents

Two coding agents ride the **logs** signal with their own attribute names rather than
`gen_ai.*`, so Otari recognizes them directly (no per-app configuration): **Claude
Code** (`api_request` event, `source = claude_code`, Anthropic additive cache counts)
and **Codex** (`codex.sse_event` / `codex.api_request`, `source = codex`, OpenAI-shaped
counts that the price de-includes). Both are re-priced at Otari's rates, deduped, and
never counted toward budget, the same as the generic path above. For client setup,
see the dedicated guides:

- [Use with Claude Code](use-with-claude-code.md)
- [Use with Codex](use-with-codex.md)

## Other sources

`POST /v1/usage/external-events` (above) is the explicit path for any source that is
not OTLP: send normalized, content-free events yourself. The OTLP endpoints map onto
it internally, so all three share the same idempotency, pricing, and
budget-exempt-key rules.

## Pricing

Imported events are priced with Otari's effective configured pricing **at the
event's timestamp**, so a rate change is honored historically. Anthropic
cache-read and cache-write rates apply when configured. The result is an
API-rate estimate, not an invoice or a subscription charge. Configure prices with
`POST /v1/pricing` (set `effective_at` at or before the events you import).

Pricing is **optional** for imported usage. `require_pricing` is a
budget-enforcement safety gate, and imported usage is budget-exempt, so a model
with no configured price is not rejected: the row lands with `cost: null` and you
can add pricing whenever you want to start seeing the cost. The same holds for
gateway requests on an API key flagged `exclude_from_budget`, they are logged
(cost null when unpriced) rather than blocked by the pricing gate.

## Budget behavior

Imported usage is observability with real cost, **not** enforcement, and that is a
hard invariant, not a default you can turn off:

- **Import keys must be budget-exempt.** A budgeted API key is refused with a 403.
  Imported usage is retrospective, it already happened somewhere Otari did not
  route, so Otari can never *block* it. Letting it count toward an enforced budget
  would let a user silently blow through a ceiling Otari had no way to hold. Rather
  than ship that footgun, ingestion requires an `exclude_from_budget` key (or the
  master key, which imports as observability).
- It never calls reservation, reconciliation, refund, or user-spend mutation; rows
  are written `counts_toward_budget = false`.
- It appears in usage analytics (`/v1/usage`, `/v1/usage/summary`, the Usage and
  Activity pages) with its `source`, so gateway and imported cost are
  distinguishable (`/v1/usage/summary` returns a `by_source` breakdown). The Usage
  page labels the total **Tracked cost** and discloses how many requests carry no
  price; the Activity page shows each row's source.
- It does **not** appear in a user's budget-consumption gauge, because that gauge
  reads the enforcement ledger (`User.spend`), which imported usage never touches.

The same `exclude_from_budget` flag also governs a key's **live** traffic: requests
proxied through an exempt key are logged with cost but never reserved, billed to
spend, or blocked (and skip the `require_pricing` gate). So the flag is one switch,
"this key's usage is tracked, never enforced," applied to both proxied and imported
usage.

## See also

- [Use with Claude Code](use-with-claude-code.md)
- [API reference](api-reference.md)
- [Modes](modes.md) for standalone vs hybrid behavior
