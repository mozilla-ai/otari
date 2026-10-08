# API reference

Otari serves an OpenAPI document at `/api/v1/openapi.json` and interactive API
docs at `/api/v1/docs` by default. The repository also commits the generated
[OpenAPI specification](https://github.com/mozilla-ai/otari/blob/v0.18.0/docs/public/openapi.json) and
[Postman collection](https://github.com/mozilla-ai/otari/blob/v0.18.0/docs/public/otari.postman_collection.json). Those generated
artifacts are the source of truth for paths, parameters, and schemas.

The default server address is `http://localhost:8000`. The API is mounted at
`/api/v1`. OpenAI-compatible clients use `http://localhost:8000/api/v1` as their
base URL; Anthropic-compatible clients use `http://localhost:8000/api`, because
their SDK appends `/v1/messages` itself. OTLP ingest is a sibling at `/otlp`:
point `OTEL_EXPORTER_OTLP_ENDPOINT` at `http://localhost:8000/otlp` and the
exporter appends `/v1/traces`, `/v1/logs` and `/v1/metrics`.

## Authentication

Otari accepts a credential in any of these forms, whatever the mode: a local API
key or the master key in standalone and hosted mode, an otari.ai user token in
hybrid mode:

```text
Authorization: Bearer <token>
Otari-Key: <token>
Otari-Key: Bearer <token>
x-api-key: <token>
```

Use API keys for inference. The master key is a deployment-wide administrative
credential. Dashboard sessions authenticate browser requests, but deployment-wide
operations also require operator authority. Where a tenant needs one of those
operations, a separately scoped endpoint serves it to the caller's own
organization: `/api/v1/organizations/me/usage` for usage, and
`/api/v1/organizations/me/keys` for a member's own API keys.

In hybrid mode, the generation APIs and the `/api/v1/mcp` and `/api/v1/hooks`
endpoints accept an otari.ai user token in the same header forms. Local API
keys and management APIs are not used.

## Availability by mode

| Surface | Standalone | Hosted | Hybrid |
| --- | --- | --- | --- |
| Health and `/api/v1/bootstrap` | Yes | Yes | Yes |
| Chat, Messages, and Responses | Yes | No | Yes |
| Caller-orchestrated MCP | Yes | No | Yes |
| Other inference APIs | Yes | No | No |
| `/api/v1/models` | Yes | Yes | Yes |
| Management APIs | Yes | Yes | No |

In hybrid mode, `/api/v1/models` lists the models the platform reports for the
caller's key, so it needs the platform's model-listing endpoint (see the
[Hybrid mode protocol](hybrid-mode-protocol.md#model-listing)).

Hosted mode is a control plane. Its inference paths return a descriptive `404`
and, when configured, the data-plane URL to use instead. See [Modes](modes.md).

## Core inference APIs

Otari implements three completion surfaces:

- `POST /api/v1/chat/completions`, OpenAI Chat Completions
- `POST /api/v1/messages` and `/api/v1/messages/count_tokens`, Anthropic Messages
- `POST /api/v1/responses`, OpenAI Responses

Standalone mode also serves embeddings, images, audio, files, batches,
moderations, rerank, search, and decisions. Provider support differs by endpoint, so use
`GET /api/v1/models` and the OpenAPI document for the deployment you are calling.

### Responses on providers without a Responses API

`/api/v1/responses` also serves providers that only implement Chat Completions
(Mistral, Anthropic, Bedrock, and most others). Otari translates the request into
a chat completion and translates the answer back, streamed or not, so the client
sees an ordinary Responses object or event stream. A provider with its own
Responses API (OpenAI, Azure OpenAI, Gemini, Groq, and others) is called
natively, as before.

The translation covers text, image and file input, `instructions`, function
tools and `tool_choice`, `max_output_tokens`, `reasoning.effort`, JSON output
through `text.format`, and reasoning text a provider returns. Gateway-run tools
(MCP, web search, code execution) work on top of it.

Some requests have no chat equivalent and are refused with a `400` that names
the field:

- `previous_response_id`, `conversation`, `background`, `context_management`
  and `prompt`, which need the provider to keep state. Send the full
  conversation in `input` instead.
- Hosted tools such as `file_search` or `computer_use`, and any tool type other
  than `function`.
- Input items other than messages, function calls and their outputs, such as
  `item_reference`.
- Image or file parts in a `system` or `developer` message or in a function
  call output, where a chat completion only takes text.

Fields that only control what a provider stores or adds to its answer
(`store`, `metadata`, `include`, `truncation` and similar) are ignored, and
reasoning items on an inbound `input` are dropped, because Chat Completions has
no way to send them back.

### Request ID and inline cost

Every Chat, Messages, and Responses response carries an `Otari-Request-ID`
header, streaming or not. In hybrid mode it is the platform's id for the
request; a standalone gateway mints its own.

A priced response also carries its cost on the usage object it already returns,
as `usage.cost_usd` (a six-decimal USD string) and `usage.pricing_source`. On a
stream the fields ride the terminal usage event: the last usage chunk for Chat
Completions, `message_delta` for Messages, and `response.completed` for
Responses. The two fields always appear together, and an unpriced or
unreported request carries neither.

In standalone mode the amount is the one the gateway wrote to its own usage
record, including any gateway-run tool charges, and `pricing_source` names the
rate that priced the model: `organization` (an organization's override),
`deployment` (a rate stored on this gateway), or `defaults` (the bundled
genai-prices dataset). Hybrid mode attaches the platform's settlement instead;
see [Hybrid mode protocol](hybrid-mode-protocol.md#inline-response-fields).

### Provider-specific fields

A field a provider adds to a chat completion's message beyond the OpenAI schema
(Exa's `citations`, for instance) is kept where the provider put it and copied
under `message.provider_specific_fields` (`delta.provider_specific_fields` on a
stream), where clients written against LiteLLM look for it.

### Request tags

A request can tag its spend through `metadata`, OpenAI's field for this, on all
three surfaces:

```json
{"model": "openai:gpt-4o", "messages": [...], "metadata": {"purpose": "chat", "country": "DE"}}
```

A standalone gateway records the tags on every usage row the request writes:
served, failed, streamed or not, refused by the gateway (a disallowed model,
missing pricing, an exhausted budget), and the separate row billing an image
description made for a text-only model. LiteLLM's
nested form, `"metadata": {"spend_logs_metadata": {...}}`, is read as well, so a
client moving off a LiteLLM proxy keeps its attribution unchanged; a nested key
wins over a flat one of the same name. The limits are OpenAI's: up to 16 string
pairs, keys up to 64 characters and values up to 512. A null value is ignored,
and anything else is refused with a 422.

What reaches the provider depends on the surface. Chat Completions never
forwards `metadata`. Messages forwards only `metadata.user_id`, the one key
Anthropic accepts there, which also names the billed user, as it always has, and
is not a tag. Responses forwards `metadata` as before, minus
`spend_logs_metadata`, because the Responses API stores it.

The usage endpoints (`GET /api/v1/usage`, `/count`, `/summary`, `/series`, and
their `/api/v1/organizations/me/usage` counterparts) filter by tag with a
repeatable `tag=key:value`. Values for the same key match any of them, and
different keys must all match. `/summary` also takes `group_by_tag=<key>` and
returns spend by that tag's values as `by_tag`, with untagged rows under a null
key:

```
GET /api/v1/usage/summary?group_by_tag=purpose&dimensions=none
```

Tags are recorded in standalone mode only.

### Cost of a failed or interrupted request

A stream that fails mid-response ends in an error event, and one the client
disconnects from ends with nothing, so neither delivers `usage.cost_usd`. The
provider may still have charged for the tokens it reported before the stream
ended (an Anthropic stream reports its input tokens in `message_start`), and a
standalone gateway records and bills those tokens rather than treating the
request as free.

To recover that amount, look the request up by its `Otari-Request-ID`:

```
GET /api/v1/usage/requests/{request_id}
```

```json
{
  "request_id": "5f0c…",
  "status": "error",
  "cost_usd": "0.012400",
  "prompt_tokens": 4100,
  "completion_tokens": 0,
  "total_tokens": 4100,
  "row_count": 1
}
```

The response sums every usage row the request wrote: a request routed through a
policy writes one per attempt, all sharing that id as their `request_group_id`,
so the total covers the attempts it fell over from as well as the one that
served. `cost_usd` uses the inline format and is `null` when nothing was priced.
An API key sees only its own requests and the master key sees any. The endpoint
answers 404 until the request has settled, since usage rows are written in the
background, and for an id that is unknown or belongs to another key. A stream the
client abandoned before the provider reported any usage, and that ran no gateway
tools, has nothing to bill and writes no row, so its id stays 404.

This lookup is standalone only. In hybrid mode the platform owns settlement, and
a failed stream reports no usage to it; see
[Hybrid mode protocol](hybrid-mode-protocol.md).

### Retrying safely

A request the provider or the gateway refused (a 429, a 529, any other error) is
not billed: its budget hold is refunded, so a client can retry it as it is. A
stream that fails or that the client disconnects from is billed for the tokens
the provider reported before it ended, and nothing more (see
[Cost of a failed or interrupted request](#cost-of-a-failed-or-interrupted-request));
a retry of it is a new request, billed on its own. The case that does bill twice
is a
non-streaming request that succeeded while its response was lost on the way
back, through a dropped connection or a client timeout, because the retry calls
the provider again.

Send an `Idempotency-Key` header on a non-streaming Chat, Messages, or Responses
request to make that retry safe. The value is any unique string of 1 to 255
printable ASCII characters; a UUID is the usual choice. A retry with the same key
and the same body then gets the original response, with its original
`Otari-Request-ID` and `usage.cost_usd`, and an `Otari-Idempotent-Replayed: true`
header, without calling the provider or billing again. While the original is
still running, a retry is answered 409 with `Retry-After`, and if the original
fails the next retry runs in its place.

- A key belongs to the API key that sent it (or, for the master key, to the
  billed user), so two callers never see each other's responses.
- A retry has to be the same request: the same body, and the same
  `Otari-Code-Execution`, `Otari-Web-Search`, `Otari-Router`,
  `Otari-Router-Task`, `Otari-Conversation-Id` and `anthropic-beta` headers,
  since those change what the request does. The same key with a different body
  or different values for those headers is refused with 422, so send a new key
  for a new request. Key order and whitespace in the JSON body do not count as a
  difference.
- A retry that arrives while the original is still running is answered 409
  with `Retry-After` at once, as the IETF `Idempotency-Key` draft and Stripe's
  API do. Retry again later with the same key, backing off exponentially.
- The request holding a key renews its claim while it runs, so a retry does not
  run it a second time however long it takes. If the worker running it dies, its
  key frees up within `idempotency_lease_sec` (a minute by default).
- A response is kept for `idempotency_retention_sec` (a day by default),
  generated content included, and then deleted. It is stored encrypted with
  `OTARI_SECRET_KEY`, so a deployment without that key ignores the header, and
  a response no configured key can decrypt (after the key was rotated away)
  runs again. Responses larger than 8 MiB are not kept, so a retry of one runs
  again. Expired responses are still deleted after the header is turned off.
- Only a successful response is kept. On a retry the request is still
  authenticated and checked against the key's model access, and a user who has
  since been blocked is refused rather than given the stored response.
- Streaming requests ignore the header, and so does hybrid mode, which has no
  local database to keep the response in.

A retry runs again, and is billed again, whenever the original's response was
not stored or can no longer be read. The cases above are the ones a deployment
chooses: the response was larger than 8 MiB, its retention passed, the header was
turned off, or `OTARI_SECRET_KEY` was rotated away. Two more come from failures:
the gateway stops after the provider answers and before the response is stored,
or the database stays unreachable for about `idempotency_lease_sec` while the
original runs, so its claim lapses and a retry takes it over.

## Search

`POST /api/v1/search` and `POST /api/v1/search/{search_tool_name}` run a configured
search tool directly. This is separate from `otari_web_search`, which lets a
model request searches during a completion. Both are described in
[Built-in tools](tools.md).

A service key's `user` field names one of its end users here as it does on chat
completions: the search is billed to that end user, under the key's end-user
budget, and counted by `rate_limits`.

Search-tool management lives under `/api/v1/search-tools`. The generated OpenAPI
document describes the supported providers, filters, and management schemas.

## Decisions

`POST /api/v1/decisions` answers typed questions about a piece of content with
probabilities rather than generated text: `noul` (the probability of yes),
`choice` (one of the named options) and `score` (a level on an ordered scale).
The body is TypeSafe's System One shape, which OpenRouter's alpha Decisions API
and llama-server also take:

```json
{
  "model": "typesafe:jev-latest",
  "state": "I've been trying to connect Stripe for 3 days and I'm losing sales.",
  "questions": {
    "urgency": {"type": "noul", "instructions": "Does this message express urgency?"},
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {"billing": null, "technical": "Integrations and outages"}
    }
  }
}
```

`model` is `<provider>:<model>`, where the provider is an entry under
[`decision_providers`](configuration.md#decision-providers). The answer comes back
as the provider returned it, keyed like the questions, with `probabilities`,
`confidence` and `usage`. `images` takes data URLs for a vision decision model
served by llama-server; other providers refuse it.

`POST /api/v1/systemone` serves the same request at the path TypeSafe's SDK and
llama-server use, so a client written for either works against Otari with its
base URL set to `https://<otari>/api`.

A decision is billed like a completion: the `<provider>:<model>` rate prices the
reported tokens, the provider's own reported cost is used when no rate is set
(only OpenRouter reports one), and it is subject to budgets, key allow-lists and
`require_pricing`. A provider's refusal of the request (400, 422, or llama-server's
501 for a model that cannot answer it) returns 400, its rate limit returns 429, and
any other failure returns 502.

## Routing policies

Routing-policy management lives under `/api/v1/routing/policies`; learned-routing
examples and status live under `/api/v1/routing/preferences` and `/api/v1/routing/status`.
See [Routing policies](routing.md) for configuration and behavior, and OpenAPI for
the request schemas.

## Provider error details

Otari may return a short, sanitized provider diagnostic when the upstream
provider rejects something the caller can fix, such as a model name or request
parameter. Credentials, URLs, account identifiers, and reflected payloads are
removed.

Gateway-side failures use fixed public messages. Diagnose them with protected
logs and safe metadata such as request ID, provider, model, and status. Do not
log provider keys, prompts, responses, or raw upstream bodies.

## Error codes

A refusal a caller is expected to act on carries a stable code, both as an
`Otari-Error-Code` header and as `code` in the body beside the human-readable
`detail`: `{"detail": "...", "code": "budget_exceeded"}`. Map refusals by the
code: it keeps its meaning across releases, while the `detail` text may be
reworded.

| `Otari-Error-Code` | Status | Meaning | Also sent |
|---|---|---|---|
| `budget_exceeded` | 403 | A budget refused the request | `Otari-Budget-Scope`: `user` for the billed user's own budget, otherwise the ceiling's scope: `organization`, `workspace`, `workspace_member`, `org_member` or `api_token` |
| `user_blocked` | 403 | The billed user is blocked | |
| `user_not_found` | 404 | The billed user does not exist | |
| `rate_limited` | 429 | A gateway rate limit is full | `Otari-Rate-Limit-Rule` for a `rate_limits` rule; `Retry-After` when waiting helps |
| `upstream_rate_limited` | 429 | The provider rate limited the gateway | `Retry-After` when the provider sent one |
| `invalid_model` | 400 | The model selector names no configured provider | |
| `model_not_allowed` | 403 | The key may not use the model | |
| `context_length_exceeded` | 400 | The prompt is too long for the model | |
| `all_candidates_rejected` | 400 | Every model a routing policy, a catalog ID or the control plane tried rejected the request as invalid | |
| `pricing_required` | 402 | `require_pricing` is on and the model has no price | |
| `end_user_budget_not_allowed` | 403 | A service key named an end-user budget that is not on its `end_user_budget_ids` | |

A failure after a stream has started arrives as an error event, which carries
the code as `error.code` on Chat Completions and Responses:
`{"error": {"message": "...", "type": "server_error", "code": "upstream_rate_limited"}}`.

## Caller-orchestrated MCP

Two stored-server endpoints let an application own its own MCP tool loop, as an
alternative to sending `mcp_servers` with a completion and letting Otari own it.
`GET /api/v1/mcp/servers/{mcp_server_id}/tools` returns the tool definitions a stored
MCP server exposes to the authenticated workspace, and `POST /api/v1/mcp/execute`
runs one exact caller-authorized call and returns the remote server's native MCP
result.

Otari executes a caller-authorized call; it does not verify a user approval and
does not claim to. The calling application is the authorization boundary and owns
any human approval, argument editing, cancellation, and action history. Otari
enforces authentication, stored-server access, the stored tool allowlist, URL
safety, and its own execution bounds.

`POST /api/v1/mcp/execute` must never be retried automatically, including by a
reverse proxy or service mesh: an `outcome_unknown` response means the tool may
already have run. See [MCP](mcp.md#caller-orchestrated-mcp) for the request
shapes, the error and execution-state contract, and the limits.

## Keeping generated clients current

API changes must regenerate both committed artifacts:

```bash
uv run python scripts/generate_openapi.py
make postman
make openapi-check
make postman-check
```
