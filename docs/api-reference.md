# API reference

Otari serves an OpenAPI document at `/api/v1/openapi.json` and interactive API
docs at `/api/v1/docs` by default. The repository also commits the generated
[OpenAPI specification](public/openapi.json), the source of truth for paths,
parameters, and schemas.

## Base URL

| Where Otari runs | Base URL |
| --- | --- |
| otari.ai, an account in the United States | `https://api.otari.ai/api` |
| otari.ai, an account in the European Union | `https://eu.api.otari.ai/api` |
| Your own gateway | `https://<your-otari-host>/api`, or `http://localhost:8000/api` on your own machine |

The dashboard reaches the API at `https://otari.ai/api` as well, but programs
use the hosts above: the CDN in front of the dashboard's host ends a response
it has waited about 30 seconds for with an HTML `504` and no `Otari-Request-ID`,
while the gateway finishes the request and bills it.

Every endpoint lives under the base URL, at `/v1/...`: an OpenAI-style request
goes to `<base URL>/v1/chat/completions`, an Anthropic-style one to
`<base URL>/v1/messages`. SDKs append their own paths, so each takes the base
URL in the form it expects:

| Client | Base URL setting |
| --- | --- |
| OpenAI SDKs and OpenAI-compatible tools | the base URL plus `/v1`, for example `https://api.otari.ai/api/v1` |
| Anthropic SDKs and Claude Code | the base URL itself, for example `https://api.otari.ai/api`, since they append `/v1/messages` |

Leaving `/api` off is the most common mistake:

- On otari.ai, a request to `https://api.otari.ai/v1/...` never reaches Otari.
  A small one answers `404 Not Found`; a large one, such as any Claude Code
  request, answers `403 Forbidden`, which Claude Code reports as "Failed to
  authenticate".
- Your own gateway answers a request under `/v1` with a 404 whose detail names
  the same path under `/api/v1`.

An otari.ai key belongs to one region. Sent to the other region's URL, it
answers `421` naming the host it belongs to.

OTLP ingest is a sibling of the API at `/otlp`: point
`OTEL_EXPORTER_OTLP_ENDPOINT` at `http://localhost:8000/otlp` and the exporter
appends `/v1/traces`, `/v1/logs` and `/v1/metrics`.

## Authentication

Otari accepts an API key or the master key in any of these forms:

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

An API key reaches:

- the inference routes;
- `GET /api/v1/usage/requests/{request_id}` for its own requests;
- [`/api/v1/key`](#a-keys-own-budget-and-usage) for its own budget and usage;
- the catalog reads: `GET /api/v1/models`, `/pricing`, `/tools`,
  `/providers/catalog`, `/tool-settings/guardrails/catalog`,
  `/tool-settings/guardrails/profiles`, `/search-tools/providers` and
  `/search-tools/catalog`;
- for a [service key](end-users.md), the end-user routes.

The rest of the management API, including `/api/v1/usage`, `/api/v1/budgets`
and everything under `/api/v1/organizations/me`, takes the master key or a
dashboard session cookie. A request that carries an API key is never read as a
session, so those routes answer it `401` with
`{"detail": "Invalid master key"}`, on otari.ai too (where a tenant has no
master key and manages that part in the dashboard).

## Core inference APIs

Otari implements three completion surfaces:

- `POST /api/v1/chat/completions`, OpenAI Chat Completions
- `POST /api/v1/messages` and `/api/v1/messages/count_tokens`, Anthropic Messages
- `POST /api/v1/responses`, OpenAI Responses

Otari also serves embeddings, images, audio, files, batches, videos,
moderations, rerank, search, fetch, and decisions. Provider support differs by endpoint, so use
`GET /api/v1/models` and the OpenAPI document for the deployment you are calling.

### Embedding input type

Some embedding models embed a query differently from a document, so
`POST /api/v1/embeddings` takes an optional `input_type` beside the OpenAI
fields and passes it to the provider: Cohere (`search_document`,
`search_query`, `classification`, `clustering`, `image`) and Voyage (`query`,
`document`). The provider checks the value, and its own default applies when the
field is omitted. A
request that sets `input_type` for any other provider is refused with a `400`
rather than embedded as if the field were absent. With the OpenAI SDK, send it
through `extra_body={"input_type": "search_query"}`.

### Responses on providers without a Responses API

`/api/v1/responses` also serves providers that only implement Chat Completions
(Mistral, Anthropic, Bedrock, and most others). Otari translates the request into
a chat completion and translates the answer back, streamed or not, so the client
sees an ordinary Responses object or event stream. A provider with its own
Responses API (OpenAI, Azure OpenAI, Gemini, Groq, and others) is called
natively.

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

Chat, Messages, and Responses responses carry an `Otari-Request-ID` header,
streaming or not, and so do the responses of the embeddings, rerank, images,
audio, moderations, decisions and search endpoints. A request rejected before
the gateway handles it (a failed authentication `401`, or a body that does not
validate, `422`) carries none, and so does an answer from a proxy or load
balancer in front of the gateway, such as a `502` or `504` page on a timeout,
since the gateway did not send it. The gateway mints the id, sends it on its own
refusals as well, and records it on the request's usage rows, so
[`GET /api/v1/usage/requests/{request_id}`](#cost-of-a-failed-or-interrupted-request)
finds the request's cost by it.

The `Otari-Provider`, `Otari-Model`, `Otari-Attempt-Count`, `Otari-Fallback`,
`Otari-Response-Duration-Ms` and `Server-Timing` headers say what served the
request and how long it took; see
[What is billed and what the caller sees](routing.md#what-is-billed-and-what-the-caller-sees).

A priced response also carries its cost on the usage object it already returns,
as `usage.cost_usd` (a six-decimal USD string) and `usage.pricing_source`. The
two fields always appear together, and an unpriced or unreported request
carries neither.

- The [rerank](#rerank), embeddings and images (`/images/generations` and
  `/images/edits`) endpoints carry the same two fields on their `usage`. An
  image model whose provider sends no `usage` (OpenAI's DALL-E models) gets one
  holding just the two fields; one that sends token counts keeps them beside
  the fields.
- On a stream the fields ride the terminal usage event:
  - Chat Completions: the last chunk that reports usage and either has no
    choices (OpenAI's `stream_options.include_usage` chunk) or finishes a
    choice (providers such as Cohere and Gemini report usage on the chunk
    carrying `finish_reason`). Otari asks the provider for that chunk even when
    a request sets `include_usage` to `false`, since it is what the request is
    billed on, and leaves it out of the stream that request receives;
  - Messages: `message_delta`;
  - Responses: `response.completed`.

The amount is the one the gateway wrote to its own usage
record, including any gateway-run tool charges, and `pricing_source` names the
rate that priced the model: `organization` (an organization's override),
`deployment` (a rate stored on this gateway), `defaults` (the bundled or
accepted models.dev snapshot), or `reported` (the charge the provider stated,
which settles a completion in place of a rate nobody set; see
[Configuration](configuration.md#keeping-the-defaults-current)).

A completion (Chat Completions, Messages, Responses) also carries
`usage.tool_cost_usd` beside them, in the same format: the part of `cost_usd`
that is per-call tool charges. That is every gateway-run tool call
(`otari_web_search`, `otari_web_fetch`, `otari_code_execution`, MCP tools) plus
the searches the model's own provider ran and billed per search, so
`cost_usd` less `tool_cost_usd` is what the model's tokens cost. It is
`"0.000000"` when no tool was charged (a tool with no price is not charged), and
left out when looking up the tools' prices failed during settlement, since the
gateway then cannot say how much of the cost they are.

Each request's cost is settled to the micro-dollar, half-up, so a request that
costs less than half a micro-dollar, such as a few tokens of a cheap embedding
model, is recorded, debited and reported as `"0.000000"`. The usage row and
the inline field always agree, and `pricing_source` still says the request was
priced.

Speech (`/audio/speech`) answers with audio bytes, so its cost rides the
response headers instead:

| Header | Value |
| --- | --- |
| `Otari-Cost-USD`, `Otari-Pricing-Source` | The same two values, under the same rule (both when the request was priced, neither when it was not). |
| `Otari-Usage-Unit`, `Otari-Usage-Units` | The meter the request was billed on and the count billed: the characters of `input` at a rate priced per character, or one request at a rate priced per request. Absent when nothing was billed, as with a zero rate. |

A speech model with no rate is refused with `402` `pricing_required` under
`require_pricing`; see
[Each endpoint in its unit](configuration.md#each-endpoint-in-its-unit) for
which rates price it.

The direct [search](#search), [fetch](#fetch) and
[code execution](#code-execution) endpoints return no `usage` object, so a
served call carries its settled cost in `Otari-Cost-USD` and
`Otari-Pricing-Source` too, under the same rule. `Otari-Pricing-Source` is
`reported` when the provider stated what the call cost, which is what it
settled at, and otherwise names the rate. A refused or failed call is not
charged and carries neither.

### Who runs a built-in tool

The `Otari-Code-Execution`, `Otari-Web-Search` and `Otari-Web-Fetch` request
headers say who runs a tool declared in a provider's own words
(`code_execution_<date>`, `code_interpreter`, `web_search_<date>`,
`web_fetch_<date>`). Each takes `auto`, `otari` or `provider`, decided for each
attempt of a fallback chain on its own. A workspace may pin the same choice for
every request billed to it, and then a header that disagrees is refused with a
403. [Who runs a tool](tools.md#who-runs-a-tool) defines the values and their
limits.

### Gateway tool calls on Chat Completions

Chat Completions has no shape for a tool the server runs, so a chat completion
reports the calls Otari ran itself (web searches and fetches, code execution and
MCP tool calls) in `otari_tool_calls`, at the top level of the response, beside
`choices`. A caller can read there what a tool really returned rather than rely
on the model's account of it. Each entry is one call, and a field that does not
apply to its type is left out:

| Field | Value |
|---|---|
| `id` | The call's ID: `otari_ws_…` for a search, `otari_wf_…` for a fetch, `otari_ci_…` for code execution, `otari_mcp_…` for an MCP call. |
| `type` | `web_search`, `web_fetch`, `code_execution` or `mcp`. |
| `status` | `in_progress`, `completed` or `failed`. A run that exits non-zero, a code execution called with no code (which never reaches the sandbox), and an MCP call that returned a tool error, are `failed`. |
| `query` | A search's query. |
| `url` | The URL the model asked a fetch for, without its query string and fragment. A redirect does not change it. |
| `sources` | A completed search's results, each with its `url`, `title` and `published_date` (an ISO date, or date and time, for a search on a search instance; the provider's own date string on the legacy search settings; or `null`); empty when it found none. Never page text. |
| `code` | The code the model sent a code execution. A `failed` one may never have run. |
| `stdout`, `stderr`, `exit_code` | What an ended code execution printed and its exit status, absent when the run never reached the sandbox. |
| `file_ids` | The [Files API](files.md) IDs of the files a code execution produced. |
| `server`, `name` | The MCP server and the tool the model called on it. |
| `output` | The text an ended MCP call gave the model. A failed call's is Otari's own `[tool error] …` text, never the server's exception. |
| `truncated` | `true` when `stdout`, `stderr` or `output` was cut to 64 KiB. The model read it whole. |
| `error` | Why a failed call failed, when Otari can say: `provider`, `status`, `tag` and `retryable`, the same four fields the [Search](#search) endpoint's `Otari-Error-*` headers carry. A call refused past its declaration's `max_uses`, or past the request's `web_search_max_calls`, has `"tag": "call_limit"` and a `null` provider. Absent when Otari can say no more than that the call failed. |

A response that is not streamed carries the ended entry of every call, across
all of the tool loop's rounds:

```json
{
  "id": "chatcmpl-…",
  "object": "chat.completion",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "Python 3.14 is…"}, "finish_reason": "stop"}],
  "otari_tool_calls": [
    {
      "id": "otari_ws_5f0c…",
      "type": "web_search",
      "status": "completed",
      "query": "latest python release",
      "sources": [{"url": "https://www.python.org/downloads/", "title": "Download Python", "published_date": null}]
    }
  ]
}
```

A stream sends a chunk when each call starts and another when it ends, before
the answer's text; a call refused past `max_uses` never started, so it is sent
once, ended. Each chunk carries whole entries at its top level and one choice
whose `delta` has no content or tool calls, so a client that reads only `choices`
passes over it. Match entries by `id`: the ended one replaces the started one.

```text
data: {"id":"chatcmpl-…","choices":[{"delta":{},"index":0,"finish_reason":null}],…,"otari_tool_calls":[{"id":"otari_ws_5f0c…","type":"web_search","status":"in_progress","query":"latest python release"}]}

data: {"id":"chatcmpl-…","choices":[{"delta":{},"index":0,"finish_reason":null}],…,"otari_tool_calls":[{"id":"otari_ws_5f0c…","type":"web_search","status":"completed","query":"latest python release","sources":[…]}]}
```

The entries are not in the `delta`, where the OpenAI Python SDK's stream helper
(`client.chat.completions.stream()`) would fail on them. That helper ignores
them, so read the raw chunks to see them. A stream that has sent an entry has
sent response bytes, so a provider failure after it ends the stream with an
error event rather than falling over to another candidate (see [Failure
behavior](routing.md#failure-behavior)).

A client may keep the entries on the assistant message it sends back: Otari
removes `otari_tool_calls` from every assistant message before calling the
provider.

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

The gateway records the tags on every usage row the request writes:
served, failed, streamed or not, refused by the gateway (a disallowed model,
missing pricing, an exhausted budget), and the separate row billing an image
description made for a text-only model. LiteLLM's
nested form, `"metadata": {"spend_logs_metadata": {...}}`, is read as well, so a
client moving off a LiteLLM proxy keeps its attribution unchanged; a nested key
wins over a flat one of the same name. The limits are OpenAI's: up to 16 string
pairs, keys up to 64 characters and values up to 512. A null value is ignored,
and anything else is refused with a 422. Tags are stored without redaction, so
never put a secret in one.

A client that can add a header but not a body field, such as Claude Code (see
[its guide](claude-code.md#tag-claude-code-traffic)), sends
the same tags in an `Otari-Tags` header, on the same three surfaces:

```
Otari-Tags: agent=bug-fix,run=1234
```

The value is comma-separated `key=value` pairs:

- Whitespace around a pair is ignored, and everything after a pair's first `=`
  is its value.
- Keys and values are percent-decoded, so `note=a%2Cb` records `a,b`. A
  non-ASCII character must be percent-encoded as UTF-8 (`caf%C3%A9`).
- A header sent more than once is read as one comma-joined list.
- Header tags are the base and the body's tags override them on the same key;
  the bounds apply to the merged set.
- A raw non-ASCII character, a pair without `=`, a `%` that does not start a
  two-digit hex escape, a key given twice, a value that does not decode to
  UTF-8, or tags past the bounds are refused with a 422 whose `detail` locates
  the error at `["header", "Otari-Tags"]`.
- The header is never forwarded to the provider.

What reaches the provider depends on the surface. Chat Completions and Messages
never forward the caller's `metadata`. On Messages, `metadata.user_id` names the
billed user and is not a tag; the provider gets an
[end-user identifier](configuration.md#end-user-identifiers) derived from it in
its place. Responses forwards `metadata`, minus
`spend_logs_metadata`, because the Responses API stores it.

The usage endpoints (`GET /api/v1/usage`, `/count`, `/summary`, `/series`, their
`/api/v1/organizations/me/usage` counterparts, and an API key's own
`/api/v1/key/usage/summary` and `/series`) filter by tag with a
repeatable `tag=key:value`. Values for the same key match any of them, and
different keys must all match. `/summary` also takes `group_by_tag=<key>` and
returns spend by that tag's values as `by_tag`, with untagged rows under a null
key:

```
GET /api/v1/usage/summary?group_by_tag=purpose&dimensions=none
```

### The `user` field

OpenAI's `user` field is a free-form end-user identifier, and OpenAI examples
often fill it with one. In Otari it names the user a request is billed to, so it
is not free-form. A request made with an API key may name only that key's own
user: any other value is refused with
`403 'user' field does not match the authenticated API key's user`. A key's
`reject_user_mismatch` (see [Mismatched `user` fields](api-keys.md#mismatched-user-fields)), or the
deployment-wide setting of the same name, can accept other values instead, in
which case the value is ignored and spend stays with the key's user. A
[service key](end-users.md) may name its own end
users, and a master-key request must name the user to bill. Chat Completions
and Responses never send `user` to the provider.

To tag a request with a free-form end-user ID, put it in `metadata`
(see [Request tags](#request-tags)) rather than in `user`:

```json
{"model": "openai:gpt-4o", "messages": [...], "metadata": {"end_user": "customer-42"}}
```

### Cost of a failed or interrupted request

- A stream that fails mid-response ends in an error event, and one the client
  disconnects from ends with nothing, so neither delivers `usage.cost_usd`.
- The gateway still records and bills the tokens the provider reported
  before the stream ended (an Anthropic stream reports its input tokens in
  `message_start`). The request is not treated as free.
- A stream the client abandoned before the provider reported any usage, and
  that ran no gateway tools, writes no usage row under the `fail` and
  `allow_free` policies. Under `estimate` (the default of
  `stream_missing_usage_policy`) it writes an `incomplete` row billed on an
  estimate from the request and the output streamed so far (text, tool-call
  arguments and reasoning): tokens only, with no per-request charge.
- A stream the provider cut after it started answering, with no usage
  reported or only the input tokens (an Anthropic stream), is billed the same
  way under `estimate`: the reported input tokens, or an estimate of the
  request, plus an estimate of the output already streamed, never a projected
  full answer. The row stays an `error` with its error message, so it counts
  in the error totals, and its estimated cost counts in spend. A stream the
  client abandoned after the provider reported only its input tokens is
  topped up the same way, so leaving early never costs less than the output
  already received.
- When a usage row is written for a stream cut before it completed,
  `stream_cut_by` records who ended it: `provider` when the upstream dropped
  it or failed after it started answering, `client` when the caller
  disconnected. A provider failure before the first chunk is an ordinary
  error, and a client disconnect that writes no row (above) records nothing.
  The field is null on any row whose stream was not cut.

`GET /api/v1/usage/requests/{request_id}` returns what a request was billed,
looked up by its `Otari-Request-ID`:

```json
{
  "request_id": "5f0c…",
  "status": "error",
  "cost_usd": "0.012400",
  "tool_cost_usd": "0.000000",
  "prompt_tokens": 4100,
  "completion_tokens": 0,
  "total_tokens": 4100,
  "row_count": 1
}
```

| Aspect | Behavior |
|---|---|
| Totals | The sum of every usage row the request wrote. A request routed through a policy writes one per attempt, all sharing the id as their `request_group_id`, so the total covers the attempts it fell over from as well as the one that served. |
| `cost_usd` | The [inline cost](#request-id-and-inline-cost) format; `null` when nothing was priced. |
| `tool_cost_usd` | The part of `cost_usd` that is tool charges, as `usage.tool_cost_usd` reports it, summed over the rows. `null` when `cost_usd` is, for a request that is not a completion, and when any priced row does not record it: one written before the gateway recorded it, or one where looking up the tools' prices failed. |
| `status` | `error` if any row is an error, `incomplete` for a stream the client abandoned before the provider reported any usage (billed on an estimate), otherwise `success`. |
| Access | An API key sees only its own requests; the master key sees any. |
| `404` | Until the request has settled (usage rows are written in the background), for an unknown id, for another key's id, and for a request that wrote no row. |

### A key's own budget and usage

These routes answer for the API key a request authenticates with, and take no
parameter that names a user, a key or a workspace. The master key and a dashboard
session get `401`: neither has a key of its own.

| Route | Returns |
|---|---|
| `GET /api/v1/key/budget` | What the key may still spend, and every cap that decides it. |
| `GET /api/v1/key/usage/summary` | The `/api/v1/usage/summary` response over the key's own requests, with the same filters, `dimensions` and `group_by_tag`. |
| `GET /api/v1/key/usage/series` | The `/api/v1/usage/series` response over the key's own requests. |

```json
{
  "budgets_enforced": true,
  "remaining_usd": 0.19,
  "limits": [
    {
      "scope": "user",
      "provider": null,
      "resets_at": "2026-12-01T00:00:00Z",
      "limit_usd": 0.5,
      "used_usd": 0.31,
      "remaining_usd": 0.19,
      "limit_tokens": null,
      "used_tokens": null,
      "remaining_tokens": null,
      "limit_requests": null,
      "used_requests": null,
      "remaining_requests": null
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `limits` | The budget the key's user is on (`scope: user`), then each spend ceiling on a scope the key bills to: `workspace_member`, `org_member`, `api_token`, `workspace`, `organization`, the most specific first. A request the key bills to its own user has to pass all of them. A [service key](end-users.md) request that names an end user checks that end user's budget in place of the `scope: user` entry, and that budget is not in this report; the key and member ceilings still apply. |
| `used_*` | What the budget gate counts: recorded use plus the holds of requests still in flight. A period that has ended reads as the fresh one the next request starts. |
| `workspace` and `organization` caps | Shared with the rest of the workspace or organization, so `limit_*` and `used_*` are `null` and only `remaining_*` is reported. `remaining_*` still falls as the others spend, so a key that reads it over time can tell how fast the team is spending. |
| `remaining_usd` (top level) | The fewest dollars any dollar cap on every provider leaves: how much the key can spend before one of those refuses a request with `403 budget_exceeded`. A token cap, a request cap or a cap narrowed to one provider (each listed in `limits`) can refuse sooner. `null` when no dollar cap on every provider applies. |
| `budgets_enforced` | `false`, with no limits, when the key is budget-exempt or the deployment runs with `budget_strategy: disabled`; `true` otherwise, even when no cap applies. |

To read what each kind of request cost, tag requests with
[`metadata`](#request-tags) and group the summary by that tag:

```
GET /api/v1/key/usage/summary?group_by_tag=role&dimensions=none
```

### Retrying safely

What a retry costs depends on how the first attempt ended:

| First attempt | Billed | A retry |
|---|---|---|
| Refused by the provider or the gateway (a 429, a 529, any other error) | No: its budget hold is refunded | Can be sent as it is. |
| A stream that failed or that the client disconnected from | The tokens the provider reported before it ended, and nothing more (see [Cost of a failed or interrupted request](#cost-of-a-failed-or-interrupted-request)) | Is a new request, billed on its own. |
| A non-streaming request that succeeded while its response was lost (a dropped connection, a client timeout) | Yes | Calls the provider again and is billed again, unless both carry the same `Idempotency-Key`. |

#### `Idempotency-Key`

A non-streaming Chat, Messages, or Responses request may carry an
`Idempotency-Key` header: any unique string of 1 to 255 printable ASCII
characters, usually a UUID. A retry with the same key and the same request gets
the original response, with its original `Otari-Request-ID`, serving headers
and `usage.cost_usd`, plus `Otari-Idempotent-Replayed: true` (and no
`Otari-Response-Duration-Ms` or `Server-Timing`, which described the original), without calling the
provider or billing again.

| Aspect | Behavior |
|---|---|
| Scope | A key belongs to the API key that sent it (or, for the master key, to the billed user), so two callers never see each other's responses. |
| Same request | The same body, and the same `Otari-Code-Execution`, `Otari-Web-Search`, `Otari-Web-Fetch`, `Otari-Router`, `Otari-Conversation-Id`, `Otari-Tags` and `anthropic-beta` headers, since those change what the request does. Key order and whitespace in the JSON body do not count as a difference. |
| Different request, same key | `422`. |
| Original still running | `409` with `Retry-After`, at once, as the IETF `Idempotency-Key` draft and Stripe's API answer, for as long as the original runs (a slow model writing a long answer can take minutes), unless its claim lapses, as the list below says. |
| Original failed | The next retry runs in its place. Only a successful response is kept. |
| Long-running original | The request holding a key renews its claim while it runs, so a retry never runs it a second time. If the worker running it dies, its key frees up within `idempotency_lease_sec` (a minute by default). |
| Retention | A response, generated content included, is kept for `idempotency_retention_sec` (a day by default; `0` ignores the header) and then deleted. Expired responses are still deleted after the header is turned off. |
| Storage | Encrypted with `OTARI_SECRET_KEY`; a deployment without that key ignores the header. Responses larger than 8 MiB are not kept. |
| Checks on a replay | The request is still authenticated and checked against the key's model access, and a user who has since been blocked is refused rather than given the stored response. |
| Not applied | Streaming requests ignore the header. |

A keyed retry runs again, and is billed again, whenever the original's response
was not stored or can no longer be read:

- the response was larger than 8 MiB;
- its retention passed, or the header was turned off;
- `OTARI_SECRET_KEY` was rotated away, so no configured key can decrypt it;
- the gateway stopped after the provider answered and before the response was
  stored;
- the database stayed unreachable for about `idempotency_lease_sec` while the
  original ran, so its claim lapsed and a retry took it over.

#### Procedure: retry a non-streaming request

1. Generate a UUID for the request and send it as `Idempotency-Key`.
2. On a lost response, resend the identical request with the same key.
3. On a `409`, wait for `Retry-After` and retry with the same key, backing off
   exponentially. Keep the same key until the original settles: a new key runs
   the request again and bills it again.
4. For a new or changed request, generate a new key; reusing one is a `422`.

The same retry recovers what a lost response would have said: the replay
carries the original's `Otari-Request-ID`, and its `usage.cost_usd` where the
original had one. An answer from a proxy in front of the gateway carries
neither (see [Request ID and inline cost](#request-id-and-inline-cost)).

## Rerank

`POST /api/v1/rerank` orders a list of documents by relevance to a query. The
response keeps the provider's own billing report in `meta`, unchanged, and says
in `usage` what the gateway metered and what it charged:

```json
{
  "id": "…",
  "model": "rerank-v3.5",
  "results": [{"index": 0, "relevance_score": 0.86}],
  "meta": {"billed_units": {"search_units": 1.0}, "tokens": null},
  "usage": {"total_tokens": null, "search_units": 1, "cost_usd": "0.002000", "pricing_source": "deployment"}
}
```

- `total_tokens` is the token count the provider reported, or null when it reported none.
- `search_units` is the number of search units the request was billed for, and
  appears only when the model is priced per request: the provider's count
  rounded up, or one when it reported none. A model priced per token is billed
  on `total_tokens` instead, and its usage has no `search_units`.
- `cost_usd` and `pricing_source` are the inline cost described under
  [Request ID and inline cost](#request-id-and-inline-cost), and are absent when
  the request was not priced.

How a rerank model is priced, per search unit or per input token, is described in
[Each endpoint in its unit](configuration.md#each-endpoint-in-its-unit).

## Videos

A video takes minutes to render, so the Videos API is asynchronous, in the
shape of OpenAI's. `POST /api/v1/videos` starts a job and returns it at once,
`GET /api/v1/videos/{video_id}` reads its status from the provider that runs
it, `GET /api/v1/videos/{video_id}/content` serves the finished video's bytes
in the provider's media type, and `GET /api/v1/videos` lists the caller's own
jobs from Otari's records, newest first. There is no delete and no remix.

```json
{"model": "gemini:veo-3.0-generate-001", "prompt": "a cat on a skateboard", "seconds": 8}
```

The request takes `model`, `prompt` and the optional `seconds`, `size`,
`resolution`, `aspect_ratio`, `seed`, `negative_prompt`, `reference_images`,
`reference_audios`, `first_frame` and `last_frame`, which reach the provider
as any-llm names them. `user` is Otari's own and is never forwarded: it names
the billed user under the rules of [The `user` field](#the-user-field). The
job comes back as:

```json
{
  "id": "…",
  "object": "video",
  "model": "veo-3.0-generate-001",
  "provider": "gemini",
  "status": "queued",
  "created_at": 1700000000,
  "completed_at": null,
  "progress": null,
  "seconds": null,
  "size": null,
  "error": null
}
```

- `status` is `queued`, `in_progress`, `completed` or `failed`. `error` carries
  the provider's message and code when the job failed.
- `seconds` is the length the provider reports, once it does.
- `provider` is the instance that runs the job, which every later call on the
  job uses.

How a video is billed:

- The model must have a per-second video price, from the catalog or a `rates`
  block set through the pricing API
  ([Every price in its unit](configuration.md#every-price-in-its-unit-rates)).
  A model with none is refused with 402 and the `pricing_required` error code,
  whatever `require_pricing` says: a video is never started without a
  per-second price.
- Starting a job reserves the length asked for at that price, or 8 seconds
  when the request names none, so a caller over budget is refused before the
  job starts. The hold is released once the provider accepts the job, and the
  start writes a usage row with no cost under the `/v1/videos` label.
- A provider that renders inside the start call, as DeepInfra does, returns
  the job already `completed`: it is billed then, and what it reported is
  kept on Otari's record, so a later read of the job or its content answers
  from the record without asking the provider.
- The first `GET` that finds the job `completed`, on the job or on its content,
  prices the seconds the provider reports, else the seconds asked for, and
  writes one usage row under the `/v1/videos/completed` label with the cost,
  its `seconds` meter and the charge line. The cost folds into the owner's
  spend in dollars, except for a job started on a key exempt from budgets,
  whose row carries the cost and whose spend stays as it was.
- A job whose length neither the provider nor the request gives is billed
  what the provider charged for it, where the provider says (DeepInfra does).
  That row has the cost and no meter, charge line or pricing source, because
  no rate priced it. Where the provider gives no charge either, the job is
  billed 8 seconds, the length the start reserved, at the per-second price.
- A job the per-second price bills is still checked against what the
  provider charged, where it says. When the two differ by more than 10% of
  the provider's charge, the server log warns with both amounts, because the
  stored price is then likely stale or wrong. The job is still billed at the
  stored price.
- A completed job whose model has lost its per-second price stays
  unbilled, and the first read after the price is back bills it. Later reads
  bill nothing, a failed job is never billed, and a job nobody reads is never
  billed.
- A job the gateway could not record after the provider accepted it answers
  500, and the server log names the provider's job id.
- A key reads only the jobs its user started in its workspace. Another user's
  job, and an unknown id, are 404. The master key reads any job, and is the
  only key that reads a job whose workspace was deleted since.

Which providers serve video is any-llm's to say: Gemini, Together and DeepInfra
in the release Otari pins. A provider without it is refused with 422 before it
is called.

## Search

`POST /api/v1/search` and `POST /api/v1/search/{search_tool_name}` run a configured
search tool directly. This is separate from `otari_web_search`, which lets a
model request searches during a completion. Both are described in
[Built-in tools](tools.md).

The request and the response follow LiteLLM's `/v1/search`, so a client written
for it works against Otari with its base URL changed. `include_raw` and
`provider_options` are Otari's own.

**Which tool.**

- The path names the tool. On `POST /api/v1/search`, the body's
  `search_tool_name` does, else the deployment's `web_search_default_tool`,
  else the only configured tool.
- A call that names no tool runs on the workspace's
  [organization search key](tools.md#organization-search-keys) when it has a
  usable one; a named call runs on that tool with its own key. Either way the
  tool decides the access check, the `rate_limits` rules, the timeout and the
  usage row's model and provider.
- Where there is no tool to pick (none configured, or several and no default),
  a call that names none still runs on the organization key, as the tool
  `web_search` on the key's provider, with a 30 second timeout.
- A deployment with no search tool, whose legacy settings describe a backend
  (`web_search_provider` with its key, else `web_search_url`), serves that
  backend as a tool named after its provider, such as `tavily` or `searxng`.
  A call that names none runs on it when the workspace has no organization key;
  a call that names it runs on it in any case; any other name is a 400 listing
  it. The deployment pays: it is priced under `otari:web_search`, and an
  `allowed_tools` grants it by that name. Once any search tool exists, the
  legacy backend is neither served nor listed here.
- The access check reads the key's `allowed_tools`, else its user's, for the
  tool's name or `*`. With neither set, it follows the key's model access, or
  its user's when the key has none: unrestricted model access runs every search
  tool, and restricted model access runs only a tool named by its deprecated
  `<provider>:<tool>` entry in `allowed_models`
  ([Model and search tool access](api-keys.md#model-and-search-tool-access)).

**Request.**

| Field | Type | Meaning |
|---|---|---|
| `query` | string, required | What to search for. |
| `search_tool_name` | string | The tool, when the path does not name one. |
| `max_results` | integer, at least 1 | How many results. 10 when unset. No upper bound of Otari's own: lowered to the workspace's ceiling when its web-access policy sets one, and to the provider's own maximum (100 for Exa). |
| `search_domain_filter` | up to 20 domains | Only results from these domains and their subdomains; a leading `-` excludes a domain instead. Exa filters natively; for other providers the results are filtered after the search. |
| `country` | two-letter ISO code | Localizes results where the provider can (Exa's `userLocation`); ignored where it cannot. |
| `max_tokens_per_page` | integer, at least 1 | The page budget, at four characters a token: it cuts every result's `snippet`, and bounds the page text Exa returns, at most 10,000 characters. Unset, a budget of 1,024 tokens cuts every provider's snippet but Exa's, whose page text that budget already bounds. |
| `include_raw` | boolean | Also return `raw`, the provider's whole response. Default `false`. |
| `provider_options` | object | Native options of the provider serving the call, under its own names, as `GET /api/v1/search-tools/providers` lists them. They win over the tool's own options; the fields above win over them. An option the provider does not take, or takes only from the deployment, is a 400 naming it. A call that names no tool may be served by an organization key on another provider, so name the tool when the options are meant for one provider. |
| `user` | string | Who the search is billed to; see below. |

Until any-search has their adapters, `include_raw` and `provider_options` are a
400 on a call a Tavily or Brave organization key serves; `false` and `{}` pass.

**Response.**

```json
{
  "object": "search",
  "search_tool": "exa-search",
  "results": [
    {
      "title": "Python 3.14.0 (final) is here!",
      "url": "https://blog.python.org/2025/10/python-3140-final-is-here.html",
      "snippet": "Python 3.14.0 is the newest major release ...",
      "date": "2025-10-01T00:00:00.000Z",
      "last_updated": null
    }
  ]
}
```

`title` and `snippet` are always strings, empty when the provider gave none.
`snippet` is the page text when the provider returned it, else its excerpt. `date`
is the page's date as the provider gave it, else `null`, and `last_updated` is
`null`. `raw` appears only when the request set `include_raw`.

A service key's `user` field names one of its end users here as it does on chat
completions: the search is billed to that end user, under the key's end-user
budget, and counted by `rate_limits`.

**Price.** A search is billed at the provider's reported cost when it reports
one, else at the tool's own per-call rate: its `price_per_call`, or a price row
of kind `search` keyed by the tool's name (`POST /api/v1/pricing` takes a
`kind`, as do the pricing reads and deletes, as a `?kind=` query, and
`/api/v1/organizations/me/pricing`, where only a deployment operator sets a
`search` or `fetch` override). A row written under the old
`<provider>:<tool>` key is read after it, until a later release. See
[Pricing a gateway-run tool](tools.md#pricing-a-gateway-run-tool) for the order and the organization key's rate.
A served search carries what it settled at in `Otari-Cost-USD` and
`Otari-Pricing-Source` ([inline cost](#request-id-and-inline-cost)).

**Errors.** Where a refusal carries a stable code, it is in the
`Otari-Error-Code` header and the body's `code`; see [Error codes](#error-codes).

| Situation | Status | `Otari-Error-Code` |
|---|---|---|
| No credentials, or a key that is not valid | 401 | |
| The tool named is not configured, or the call names none and there is no tool to pick (none configured, or several with no default) nor an organization search key to stand in | 400 | |
| The tool cannot run as configured, such as a `searxng` tool with no address | 400 | |
| A `provider_options` key the provider does not take from a request | 400 | |
| `include_raw` or `provider_options` on a call that does not run on any-search yet | 400 | |
| The master key without a `user` | 400 | |
| The body fails schema validation, such as an `include_raw` that is not a boolean | 422 | `invalid_request` |
| The workspace turned web search off | 403 | `web_access_disabled` |
| The key's `allowed_tools`, or the deprecated `<provider>:<tool>` entry in `allowed_models`, does not grant the tool | 403 | |
| A `user` naming another user, where the key rejects a mismatch | 403 | |
| A service key's `Otari-End-User-Budget` is not one it may use | 403 | `end_user_budget_not_allowed` |
| A budget refuses the search | 403 | `budget_exceeded` |
| `require_pricing` is on and the search has no price (a row of kind `search` keyed by the tool's name, or its `price_per_call`; `otari:web_search` for a search the workspace's organization key or the legacy settings serve); the detail names it | 402 | `pricing_required` |
| The tool runs on the deployment's account and the organization has no funds paid in | 402 | `insufficient_funds` |
| A rate limit is full | 429 | `rate_limited` |
| The workspace's organization search key cannot be used as it is | 502 | |
| The provider failed the search, or could not be reached | 502 | |
| The workspace's stored web-access policy cannot be read | 503 | |

Every provider failure is a 502 whose `detail` names no upstream detail (the
completion routes, by contrast, classify a provider's failure). The 502 carries
what is safe to show of the failure as headers, never a body, a URL or the
query:

| Header | Value |
|---|---|
| `Otari-Error-Provider` | The provider that failed, such as `exa`. |
| `Otari-Error-Status` | The provider's own HTTP error status; absent when it gave none, as for a timeout, an error inside a successful answer, or an answer that could not be read. |
| `Otari-Error-Tag` | What failed, in the provider's terms or the library's: `timeout`, `network`, `invalid_header` (a key holding a character that is not ASCII), `http_error`, `invalid_response`, `unexpected_response`, or the provider's own code, such as Exa's `INVALID_API_KEY`. |
| `Otari-Error-Retryable` | `true` for a timeout, a 429 or a 5xx from the provider, when a retry may help; otherwise `false`. |

On a search the workspace's organization key serves, they name the key's
provider. A 502 for a failure that is not the provider's, such as an organization
key this gateway cannot call, carries none of them.
The completion routes send the same headers when a provider fails a completion;
see [Provider failures](#provider-failures).

**Listing the tools.** `GET /api/v1/search-tools/catalog` lists the tools the
calling key may name, as `GET /api/v1/models` lists the models it may call: a
tool its `allowed_tools` does not grant is left out. Each entry carries the
tool's name as `id`, its `provider`, a display `label`, its `price_per_call`,
and `default`, true for the deployment's default tool. `price_per_call` is the
configured rate in USD for the caller's organization (its override, else the
deployment's price list), not what a call is charged: a provider that reports a
search's cost, such as Exa, is charged that cost, and the rate prices only a
call whose provider reports none. It is `null` when no rate row prices the
tool, which says nothing of whether its provider reports a cost. A call that
names no tool runs on the default, unless the workspace has an organization
search key, which then serves it; a call that names the default tool runs on that tool's own key and price. A tool a direct
search cannot run on is left out, and no key, endpoint or option is returned.
On a deployment with no search tool, the backend the legacy settings describe
is the one entry, named after its provider, the default, and priced at the
`otari:web_search` rate.
An API key whose workspace has web access off gets an empty list;
`GET /api/v1/tools` says why. The list takes `skip` and `limit` (at most 1000,
100 by default).

```json
{
  "object": "list",
  "data": [
    {"id": "exa", "object": "search_tool", "provider": "exa", "label": "Exa", "price_per_call": 0.008, "default": true}
  ]
}
```

Search-tool management lives under `/api/v1/search-tools`. The generated OpenAPI
document describes the supported providers, filters, and management schemas.

## Fetch

`POST /api/v1/fetch` and `POST /api/v1/fetch/{fetch_tool_name}` read one page
through a configured fetch tool, as [Search](#search) runs a search tool. This is
separate from `otari_web_fetch`, which lets a model fetch pages during a
completion; both are described in [Built-in tools](tools.md#direct-fetch).

`web_fetch_enabled` governs this endpoint as it governs the `otari_web_fetch`
tool, read on every request: while it is off, a call is refused with `400`.
Search result enrichment follows `web_search_extract` instead.

**Which tool.** The path names the tool. On `POST /api/v1/fetch`, the body's
`fetch_tool_name` does, else the deployment's `web_fetch_default_tool`, else
`builtin_fetch`, the gateway's own fetcher, which always exists. A call that
names no tool runs on the workspace's
[organization fetch key](tools.md#organization-search-keys) when it has a usable
one; a named call runs on that tool with its own key. Either way the tool decides
the access check, the `rate_limits` rules, the timeout and the usage row's model
and provider. The access check reads the key's `allowed_tools`, else its user's,
for the tool's name or `*`, as a search does
([Model and search tool access](api-keys.md#model-and-search-tool-access)).

**Which URLs.** The workspace's web-access policy applies in full: its switch, and
its allowed and blocked domain lists, which bind the URL asked for and, on the
built-in fetcher, every redirect. A provider tool is held to the URL it reports
finding the page at; DeepSeek, Exa and Tabstack report none, so a redirect they
follow escapes the lists (see [Fetch providers](tools.md#fetch-providers)).
Whatever the lists say, a URL that names a private, loopback or link-local
address is refused. The built-in fetcher also refuses a host that resolves to
one, redirects included; a provider tool resolves hosts on its own network,
where the gateway checks only the URL it reports finding the page at.

**Request.**

| Field | Type | Meaning |
|---|---|---|
| `url` | string, required | The page: one public `http` or `https` URL. |
| `fetch_tool_name` | string | The tool, when the path does not name one; `builtin_fetch` is a valid name. |
| `max_chars` | integer, at least 1 | Cut the page text to this many characters. Unset, the tool's own `max_chars` option applies, else the provider's own limits. |
| `include_raw` | boolean | Also return `raw`, the provider's whole response. `builtin_fetch` calls no provider, so its `raw` holds only `content_kind`, how the page's body was read, and `redirect_count`. Default `false`. |
| `provider_options` | object | Native options of the provider serving the call, under its own names, as `GET /api/v1/search-tools/providers?kind=fetch` lists them. They win over the tool's own options. An option the provider does not take, or takes only from the deployment, is a 400 naming it. A call that names no tool may be served by an organization key on another provider, so name the tool when the options are meant for one provider. |
| `user` | string | Who the fetch is billed to; see [Search](#search). |

**Response.**

```json
{
  "object": "fetch",
  "fetch_tool": "builtin_fetch",
  "url": "https://blog.python.org/2025/10/python-3140-final-is-here.html",
  "final_url": "https://blog.python.org/2025/10/python-3140-final-is-here.html",
  "title": "Python 3.14.0 (final) is here!",
  "content_type": "text/html",
  "text": "# Python 3.14.0 (final) is here! ...",
  "published": null,
  "truncated": false
}
```

`url` is the URL asked for, in canonical form, and `final_url` where the page was
found after redirects, the same as `url` when the provider does not report where
a redirect led. `title` and `content_type` are empty strings when the provider
gives none. `text` is the page's text as the provider extracted it: the built-in
fetcher keeps an HTML page's content as light Markdown, without its navigation,
banners, footers and sidebars. `published` is when the page was published, where
the provider reports it, else `null`. `truncated` is `true` when `text` is less
than the whole page: the body or the extracted text hit a limit of the
provider's, or `max_chars` cut it. `raw` appears only when the request set
`include_raw`.

**Price.** A fetch is billed at the provider's reported cost when it reports one
(Exa does), else at the tool's own per-call rate: its `price_per_call`, or a
price row of kind `fetch` keyed by the tool's name, `builtin_fetch` included.
A fetch the workspace's organization fetch key served is priced under
`otari:web_fetch` instead. See
[Pricing a gateway-run tool](tools.md#pricing-a-gateway-run-tool). A served
fetch carries what it settled at in `Otari-Cost-USD` and `Otari-Pricing-Source`
([inline cost](#request-id-and-inline-cost)).

**Errors.** Where a refusal carries a stable code, it is in the
`Otari-Error-Code` header and the body's `code`; see [Error codes](#error-codes).

| Situation | Status | `Otari-Error-Code` |
|---|---|---|
| No credentials, or a key that is not valid | 401 | |
| `web_fetch_enabled` is off | 400 | |
| The tool named is not configured, or cannot run as configured | 400 | |
| The URL is not a valid public `http` or `https` URL or names a private address, asked for or reported by a provider after a redirect | 400 | |
| A `provider_options` key the provider does not take from a request | 400 | |
| The master key without a `user` | 400 | |
| The body fails schema validation, such as a `max_chars` below 1 | 422 | `invalid_request` |
| The workspace turned web access off | 403 | `web_access_disabled` |
| The workspace's domain lists disallow the URL, asked for or found after a redirect | 403 | |
| The key's `allowed_tools`, or the deprecated `<provider>:<tool>` entry in `allowed_models`, does not grant the tool | 403 | |
| A `user` naming another user, where the key rejects a mismatch | 403 | |
| A service key's `Otari-End-User-Budget` is not one it may use | 403 | `end_user_budget_not_allowed` |
| A budget refuses the fetch | 403 | `budget_exceeded` |
| `require_pricing` is on and the fetch has no price (a row of kind `fetch` keyed by the tool's name, or its `price_per_call`; `otari:web_fetch` for a fetch the workspace's organization key serves); the detail names it | 402 | `pricing_required` |
| The tool runs on the deployment's account and the organization has no funds paid in | 402 | `insufficient_funds` |
| A rate limit is full | 429 | `rate_limited` |
| The workspace's organization fetch key cannot be used as it is | 502 | |
| The provider, or the built-in fetcher, failed the fetch: the page answered an error status, timed out, could not be reached, has a content type the fetcher does not read, or could not be extracted; on the built-in fetcher, a host that cannot be resolved or resolves to a private address (`url_invalid`) | 502 | |
| The workspace's stored web-access policy cannot be read | 503 | |

A failed fetch is a 502 whose `detail` names no upstream detail; the failure's
safe fields ride as the same headers a [search](#search)'s 502 carries:
`Otari-Error-Provider` (`builtin` for the gateway's own fetcher),
`Otari-Error-Status` (the page's own HTTP status, when the fetcher reports one),
`Otari-Error-Tag` (the provider's own tag, or the built-in fetcher's: `timeout`,
`network`, `http_status`, `redirect`, `url_invalid`, `content_type`,
`extraction`, `content_decoding`) and `Otari-Error-Retryable`. The body never
carries the URL.

## Code execution

`POST /api/v1/code/execute` runs code in the gateway's sandbox and returns its
output, with no model involved. It is admitted, priced and budgeted as an
`otari_code_execution` call; see
[Running code without a model](tools.md#running-code-without-a-model).

**Request.**

| Field | Type | Description |
|---|---|---|
| `code` | string | Python source to run. Required, and sent to the sandbox unchanged. Code that is empty or only whitespace is a `422`. |
| `container` | string | `auto` to hold the sandbox after the run, or an `otari_cntr_…` id from an earlier `Otari-Container-Id` to run in that sandbox again, with its files and interpreter state. Omitted, the sandbox is released when the run ends. |
| `timeout_seconds` | integer | How long the code may run. Capped at the workspace's `exec_timeout_s`, or 60 seconds when it sets none. |
| `environment` | string | The sandbox environment to run in, a `name` from `GET /api/v1/code/environments`. Omitted, the workspace's pinned environment, else the deployment's default. A held `container` keeps the environment it was created in. |
| `file_ids` | array of strings | Uploads from `POST /api/v1/files` to put in the working directory before the code runs, each under its own filename. At most 32. |
| `user` | string | The billed user, as on the completion routes ([the `user` field](#the-user-field)). Required with the master key. |

**Response.**

```json
{
  "object": "code_execution.result",
  "stdout": "2\n",
  "stderr": "",
  "return_code": 0,
  "files": [{"file_id": "file-…", "filename": "chart.png"}]
}
```

Code that fails is still a `200`, and is billed: its `return_code` is nonzero,
or its `stderr` carries the error. `files` lists each file the code wrote,
stored in the Files API, so `GET /api/v1/files/{file_id}/content` downloads it;
it is empty where the Files API is off. A held sandbox is named in
`Otari-Container-Id` and `Otari-Container-Expires-At`. A priced run carries what
it settled at in `Otari-Cost-USD` and `Otari-Pricing-Source`
([inline cost](#request-id-and-inline-cost)); an unpriced one carries neither.

**Errors.**

| Status | When |
|---|---|
| `400` | No sandbox is configured, `container` names a sandbox that has expired or does not exist, a `file_ids` entry is not an upload the caller can read (or the Files API is off), `environment` is not one the gateway offers, or `container` names a sandbox created in another environment, or in one it did not record. |
| `402` | `require_pricing` is on and neither the environment's `otari:code_execution:<image>` nor `otari:code_execution` has a price (`pricing_required`), or the organization's funds are exhausted. |
| `403` | The workspace's code execution policy is off or pins another environment, or the billed user's budget is spent. |
| `409` | The named container is running another request. |
| `502` | The sandbox could not be reached. |
| `503` | The sandbox is temporarily out of capacity, with `Retry-After` when it gave one. |

### Environments

`GET /api/v1/code/environments` lists the sandbox environments the caller's
workspace may run code in, for any API key; a master key or a dashboard
session gets the deployment's whole list. See
[Sandbox environments](tools.md#sandbox-environments).

```json
{
  "object": "list",
  "data": [
    {
      "object": "code_execution.environment",
      "name": "python-data",
      "label": "Python data",
      "description": "pandas, numpy and matplotlib, 2 vCPU, 4 GB",
      "price_per_run": 0.05,
      "default": false
    }
  ]
}
```

`price_per_run` is USD per run for the caller's organization, the
environment's own price else `otari:code_execution`'s, and null when neither is
set. The list is empty where no sandbox is configured, the deployment names no
image, or the workspace's code execution policy is off.

`skip` (default `0`) and `limit` (default `100`, at most `1000`) page the list
in the operator's order; a value out of range is a `422`.

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
501 for a model that cannot answer it) returns 400, its rate limit returns 429, its
refusal of the gateway's credential (401 or 403) returns 424 `provider_credential_rejected`,
and any other failure returns 502.

## Routing policies

Routing-policy management lives under `/api/v1/routing/policies`.
A subagent model recommendation is asked at `POST /api/v1/routing/recommend`, and the
operator manages the subagent routers that answer it under `/api/v1/routing/recommenders`
(see [Routing recommendations](#routing-recommendations)).
See [Routing policies](routing.md) for configuration and behavior, and OpenAPI for
the request schemas.

## Routing recommendations

`POST /api/v1/routing/recommend` recommends the model a coding agent's
subagent should start on, asked by the harness at spawn time. One decision call
is made and nothing else is dispatched: the harness runs the subagent on the
answer with its own provider credentials. How the recommendation is reached is
in [Agent model recommendations](routing.md#agent-model-recommendations).

Any active API key may ask; the decision is billed to the key's own user. A
master-key request names the billed user in `user`. The call is metered,
budgeted and recorded in usage like a [decision](#decisions).

```bash
curl "$OTARI_URL/api/v1/routing/recommend" \
  -H "Authorization: Bearer $OTARI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "harness": "claude-code",
    "session_id": "bf05abe2-5ff2-4eb0-8459-ab578d5c9468",
    "tool_use_id": "toolu_01Ab3dEfGh",
    "agent_type": "Explore",
    "description": "Find the budget reservation code",
    "prompt": "Find where budgets are reserved before dispatch.",
    "parent_model": "claude-opus-5",
    "requested_model": null
  }'
```

**Request.** Unknown fields are refused with a `422`.

| Field | Type | Meaning |
|---|---|---|
| `harness` | `claude-code` or `pi`, required | The harness asking. It decides the candidates (below). |
| `candidates` | object | The models the subagent may be sent to, as Otari model ids (`<provider>:<model>`, up to 256 characters) mapped to what each is for (up to 1,024 characters, or `null` when the name says enough). Two to 255 entries. Required for `pi`, refused for `claude-code`. |
| `session_id` | string, 1 to 256, required | The harness's own id for the session the spawn happens in. |
| `tool_use_id` | string, 1 to 256, required | The harness's own id for the tool call that spawns the subagent. |
| `agent_type` | string, 1 to 128, required | The subagent type: a built-in such as `Explore` or `Plan`, or a custom agent's name. |
| `description` | string, up to 1,024 | The caller's short description of the task. Default empty. |
| `prompt` | string, 1 to 65,536, required | The task the subagent is given. Its first 32,000 characters reach the decision model. Never stored or logged. |
| `parent_model` | string, 1 to 256, required | The model the parent conversation runs on, as the harness names it. |
| `requested_model` | string, up to 256 | The model the caller asked for, if any. Received as a fact and kept out of the question, so the orchestrator's guess cannot pull the answer toward itself. |
| `user` | string | The user billed for a master-key request; not sent upstream. |

**Candidates.** A `claude-code` request chooses among the deployment's
`agent_recommender_candidates`, Claude Code's `haiku`, `sonnet` and `opus`
aliases by default, because Claude Code sends the subagent's requests to
Anthropic itself and resolves the alias on its side; it may not send
`candidates`. A `pi` request must send its own. Either mismatch is a `422`.

**Response.**

```json
{
  "model": "haiku",
  "reason": "haiku with 81%",
  "probabilities": { "haiku": 0.81, "sonnet": 0.16, "opus": 0.03 }
}
```

| Field | Meaning |
|---|---|
| `model` | One of the candidates: a model alias or id the harness can start the subagent on. |
| `reason` | One short sentence the harness may show, or `null`. |
| `probabilities` | The decision model's probability for each candidate, or `null` when it reports none. |

**Errors.** No [subagent router](routing.md#subagent-routers) stored, or every
stored one switched off, is a `400`. The decision model's refusal of the request (400, 422, or 501) is a
`400`, its rate limit a `429`, its refusal of the gateway's credential (401 or
403) a `424` with `provider_credential_rejected`, and any other failure, or an
answer that names no candidate, a `502`.

## Provider error details

Otari may return a short, sanitized provider diagnostic when the upstream
provider rejects something the caller can fix, such as a model name or request
parameter. Credentials, URLs, account identifiers, and reflected payloads are
removed.

The diagnostic is the provider's own sentence, taken from the error body it
sent, not the provider SDK's rendering of that error. An OpenAI 400 for a
model that only accepts its default temperature answers:

```json
{"detail": "Unsupported value: 'temperature' does not support 0.5 with this model. Only the default (1) value is supported."}
```

rather than a detail that begins `Error code: 400 - {'error': {...}}`.

- The same holds for Anthropic, Cohere and the other providers whose SDKs keep
  the parsed body: only the body's message can reach the caller, never a status
  prefix, the body's other fields, or the response headers.
- When that body has no message, or its message is redacted away, the caller
  gets the fixed detail for the status instead (for a 400,
  `The provider rejected the request as invalid (check the model name and parameters)`).
- The provider's `param` and `code` fields are not copied into the response.
- When the request named a model alias, a provider message that names the
  alias's target names the alias instead.

The completion routes and the pass-through routes (embeddings, images, audio,
moderations, rerank) classify a provider failure the same way:

| Provider outcome | Otari answers | `Otari-Error-Code` and headers | `detail` | Client retry |
|---|---|---|---|---|
| Rejects the request (its own 400 or 422) | 400 | | The provider's sentence | No |
| OpenAI's 403 with error type `invalid_request_error` (logprobs requested from a model that does not return them, an embedding model sent to chat) | 400 | | OpenAI's sentence | No |
| The model's output did not match the requested JSON schema (Groq's `json_validate_failed`) | 400 | `structured_output_failed` | The provider's sentence, without its pointer to `failed_generation`, which Otari does not relay | Yes: the failure is stochastic, and the same request usually succeeds |
| Unknown model | 404 | | | No |
| The provider has no such operation for the model (rerank on a chat model), or its SDK has no parameter for a field the caller sent | 400 | | Names the operation or the field | No |
| Refuses the request for its size (its own 413, such as a prompt over a tokens-per-minute limit) | 413 | `request_too_large` | The provider's sentence | No |
| Rate limit | 429 | `upstream_rate_limited`; the provider's `Retry-After` when it sent one | | After waiting |
| Refuses the gateway's own credential (its 401, or a 403 that is not a rejection of the request) | 424 | `provider_credential_rejected` | | No |
| Timeout | 504 | | | Yes |
| A non-streamed Chat Completions response with no choices | 502 | `provider_error` | `The provider returned a completion with no choices` | |
| Any other failure | 502 (500 on Messages) | | What happened upstream and nothing more: `The provider failed the request with HTTP 500` (the provider's own status), `The provider closed the connection before answering`, or `The gateway could not connect to the provider`. A failure Otari cannot place keeps the fixed `LLM provider error` | Yes, unless `x-should-retry: false` (below) |

An OpenAI-compatible client does not retry a 400 or a 404 on its own. On Chat
Completions, Messages and Responses, a timeout, a dropped connection or a
provider 5xx also carries the [provider failure headers](#provider-failures).

A chat completion with no choices is a provider failure, not an answer: it is
recorded on the usage row as an error, and a routing policy moves on to its next
candidate. Inside a gateway tool loop this applies to the model's first
response; a later one is returned as the answer, since the rounds before it
were already billed. Messages and Responses are not checked, since an empty
`content` or `output` can be a legitimate answer there.

A 5xx that resending the same request cannot fix also carries
`x-should-retry: false`, the header the OpenAI and Anthropic SDKs read before
their own retry rule, so such a client stops after one attempt:

- Two failures qualify: a provider account out of credit or over its billing
  limit, and a fault in the code making the provider call (such as a provider
  adapter that cannot convert the request), which raises before anything is
  sent.
- A timeout, a dropped connection, a provider's own 5xx or 429, and any
  failure Otari cannot place carry no such header.
- A streamed request that fails before its first chunk carries it too, since
  that is answered with an ordinary HTTP error (see [Error codes](#error-codes)).
- A routing policy still moves on to its next candidate on either failure. A
  policy that tried several candidates and saw every one fail answers without
  the header.

Gateway-side failures use fixed public messages. Diagnose them with protected
logs and safe metadata such as request ID, provider, model, and status. Do not
log provider keys, prompts, responses, or raw upstream bodies.

## Error codes

A refusal a caller is expected to act on carries a stable code, both as an
`Otari-Error-Code` header and as `code` in the body beside the human-readable
`detail`: `{"detail": "...", "code": "budget_exceeded"}`. Map refusals by the
code: it keeps its meaning across releases, while the `detail` text may be
reworded.

The Messages routes (`/api/v1/messages` and `/api/v1/messages/count_tokens`)
answer every error in Anthropic's envelope instead, the body Anthropic's SDKs
read: `{"type": "error", "error": {"type": "rate_limit_error", "message": "..."}, "code": "rate_limited"}`.
A body that fails validation there is a 400 `invalid_request_error`, as
Anthropic answers it, whose message names each failing field.

A field the table lists as "in the body" sits beside `code` in either shape,
at the top level of the body.

| `Otari-Error-Code` | Status | Meaning | Also sent |
|---|---|---|---|
| `budget_exceeded` | 403 | A budget refused the request | `Otari-Budget-Scope`: `user` for the billed user's own budget, otherwise the ceiling's scope: `organization`, `workspace`, `workspace_member`, `org_member` or `api_token`; `Retry-After`: seconds until the refusing budget's window ends, when it resets and the request would fit a fresh one; `end_user` in the body and `Otari-End-User-Budget` naming its budget, when the scope is `user` and the billed user is a service key's [end user](end-users.md#how-end-users-are-billed-and-capped) |
| `user_blocked` | 403 | The billed user is blocked | `end_user` in the body, when the billed user is a service key's end user |
| `web_access_disabled` | 403 | The workspace turned web access off, refusing a web search or web fetch | |
| `user_not_found` | 404 | The billed user does not exist | |
| `rate_limited` | 429 | A gateway rate limit is full | `Otari-Rate-Limit-Rule` for a `rate_limits` rule, or `organization_concurrency` for an organization's concurrency limit; `Retry-After` when waiting helps |
| `upstream_rate_limited` | 429 | The provider rate limited the gateway | `Retry-After` when the provider sent one |
| `invalid_model` | 400 | The model selector names no configured provider, alias, policy or catalog model id; checked before the key's allow-list, so a restricted key gets it too, and before `require_pricing`, so it is never answered as `pricing_required`. A `provider:model` selector whose provider resolves is not judged here, since only the provider knows its models | |
| `model_not_allowed` | 403 | The key may not use the model | |
| `model_not_serving` | 403 | The organization offers the model on a provider key, with its serving switch off | |
| `model_not_found` | 404 | A provider key of the organization serves the provider, and none offers the model | |
| `country_not_supported` | 403 | `provider_supported_countries` closes the model's provider to the country the request comes from, or the country is unknown. See [Supported countries](configuration.md#supported-countries) | |
| `context_length_exceeded` | 400 | The prompt is too long for the model | |
| `request_too_large` | 413 | The provider refused the request for its size, such as a prompt over a tokens-per-minute limit | |
| `unsupported_content` | 400 | The provider refused content the request carries because the model does not accept it, such as an image sent to a text-only model. Sent only when Otari recognizes the provider's wording for that refusal; a refusal worded otherwise stays a `400` carrying the provider's message and no code. `detail` names the model where it can. A routing policy moves on to its next candidate first, and answers with this code when every candidate gave a refusal Otari recognizes | |
| `unsupported_parameter` | 400 | The provider cannot express a parameter the request carries for the model, such as `response_format` on a model with no JSON mode. Sent only when Otari's provider library refuses the parameter and names it; a provider's own 400 for it carries the provider's message and no code. A routing policy moves on to its next candidate first, and answers with this code, `detail` naming the parameters, when every candidate refused one | |
| `structured_output_failed` | 400 | The model failed to produce output matching the request's JSON schema (`response_format` or `text.format`), which the provider reported with its own code (Groq's `json_validate_failed`). Unlike other 400s, resending the same request may succeed, so a client can retry it or send it to another model. The partial output the provider returns is not relayed. A routing policy moves on to its next candidate first, and answers with this code when every candidate failed this way | |
| `wrong_model_type` | 400 | The model is a kind the endpoint cannot serve. A chat completion, a message or a response is refused for an embedding model, which the provider's model listing or models.dev may name, or for a rerank model, a transcription model or one that writes no text, which only the listing names; for a routing policy or a catalog ID, only when every candidate is such a kind. `/embeddings` is refused only for a model its provider's listing names as another kind; a listed `other` model, and one only models.dev describes, still reach the provider. `detail` names the kind, except for a model the listing says writes no text, which it calls not a chat model. GET /models carries the kind as `type` | |
| `all_candidates_rejected` | 400 | Every model a routing policy or a catalog ID tried rejected the request as invalid | |
| `insufficient_funds` | 402 | The organization has no funds paid in to cover what the deployment's account would run: the model, served on a credential the deployment pays for, or a gateway-run tool ([Tools](tools.md#pricing-a-gateway-run-tool)). A tool's refusal names the tool in `detail`; a model's does not name the model. Nothing was sent upstream | |
| `pricing_required` | 402 | `require_pricing` is on and the model resolves but has no price, a direct search has no price (see [Search](#search)), or a gateway-run tool the request declares has no row to price its calls ([Tools](tools.md#pricing-a-gateway-run-tool)). An unpriced model whose provider has no credential answers `provider_not_configured` instead | |
| `provider_not_configured` | 424 | Nothing declares the model's provider, or its declaration names no key the provider needs, so nothing was sent upstream. Not a 5xx, so a client does not retry it: an operator adds the provider as a [hosted provider](hosted-providers.md) (switching on the model), or the organization adds its own provider key. A provider's environment variable alone serves nothing | |
| `provider_credential_rejected` | 424 | The provider refused the credential the gateway holds for it (an upstream 401, or a 403 other than one rejecting the request, which is a 400). It is the gateway's key, not the caller's, so it is never a 401 or 403; and not a 5xx, because no retry makes the key valid. Replace or fix the provider credential. The usage row keeps the provider's own status. A routing policy moves on to its next candidate first, and answers with this code when every candidate either had no credential or had it rejected, and at least one was rejected | |
| `end_user_budget_not_allowed` | 403 | A service key named an end-user budget that is not on its `end_user_budget_ids` | |
| `end_user_budget_without_end_user` | 400 | The request sent `Otari-End-User-Budget` but its `user` field (`metadata.user_id` on Messages) names no end user of a service key (the key is not a service key, or the request names no one or the key's own user), so the budget would apply to no one | |
| `end_user_limit_reached` | 403 | A service key already has as many end users as its `max_end_users` allows, so a request that would create a new end user or revive a deleted one is refused. End users it already has are unaffected | |
| `terms_not_accepted` | 400 | The deployment publishes terms (`terms_url`), and creating or first claiming the account needs them accepted: send `terms_accepted` with the signup, the invitation's first password, the password reset, or the OAuth callback | |
| `invalid_request` | 422, or 400 on the Messages routes | The request body or a parameter failed schema validation. `detail` lists the failing fields, each with `loc`, `msg` and `type`; on the Messages routes, `error.message` names them instead | |
| `provider_error` | 502, or 200 in a stream's error event | As a 502, the provider answered a non-streamed chat completion with no choices. In a stream's error event, the provider failed after the stream started, for a reason no other code names | |
| `provider_stream_interrupted` | 200, in a stream's error event only | The provider's stream ended before it finished: it closed without reporting a `finish_reason` (Chat Completions, and Responses translated from a chat completion) or without the event that ends a response (Responses, from a provider with its own Responses API), or the connection carrying it was cut. The output received so far is truncated | |

A `budget_exceeded` refusal's `detail` tells a spent budget (`... has exceeded
budget limit`) from a request whose estimated cost is larger than the budget it
has left (`... has $X of budget remaining, less than this request's estimated
cost of $Y`); see [Refusals](budget-enforcement.md#refusals). Its `Retry-After` is
when the budget resets, which may be days away for a monthly budget; see
[`Retry-After`](budget-enforcement.md#retry-after).

### Provider failures

When the provider itself fails a completion (Chat Completions, Messages or
Responses), by timing out, dropping the connection or answering with a 5xx, the
`5xx` Otari answers with carries what is safe to show of it, the same headers a
[direct search](#search)'s `502` carries, and never a body or a URL:

| Header | Value |
|---|---|
| `Otari-Error-Provider` | The provider that failed, as config names it, such as `exa`. |
| `Otari-Error-Status` | The provider's own HTTP status; absent for a timeout or a lost connection. |
| `Otari-Error-Tag` | `timeout`, `network`, `http_error`, or the provider's own code when its error body names one, such as Exa's `SERVICE_OVERLOADED`. |
| `Otari-Error-Retryable` | `true`: each of these may succeed on a retry. |

When a routing policy tried several candidates, they describe the last one tried.
A `4xx` from the provider carries none of them: a billing, quota or credential
failure describes the gateway's own provider account, and a provider's `429`
already carries `upstream_rate_limited`.

A streamed request whose provider fails before producing any output is answered
with an ordinary HTTP error, with the same status, `detail` and code the request
would get unstreamed: a `400` for a parameter the provider does not accept, a
`502` for a provider outage. To make that possible, Otari holds the response
until the provider's first chunk, for at most `streaming_first_chunk_hold_ms`
(15 seconds by default; `0` turns the hold off and commits the `200` as soon as
the stream opens).

- For a stream running the gateway's tools, opening their backends (connecting
  to MCP servers, leasing a sandbox) counts toward that window.
- Such a stream (MCP servers, sandbox, web search) locks onto its candidate
  once the first round of the tool loop ends, which can happen before anything
  is sent; a failure past that point cannot fall over either.
- A provider slower than the window to start still streams, and a failure
  after that point is reported as below.

A failure after a stream has started arrives as an error event; the `200`
status line has already gone out. On Chat Completions and Responses it carries
the code as `error.code`:
`{"error": {"message": "...", "type": "server_error", "code": "upstream_rate_limited"}}`.

- **Code**: the code from the table above when one applies
  (`upstream_rate_limited`, `context_length_exceeded`),
  `provider_stream_interrupted` when the provider's stream ended before it
  finished, otherwise `provider_error` when the provider failed the stream. An
  event with no `code` is a failure inside the gateway.
- **Provider failure fields**: on Chat Completions and Responses, a failure on
  the provider's own side also carries the
  [provider failure](#provider-failures)'s fields as `error.provider_failure`:
  `{"provider": "exa", "status": 503, "tag": "SERVICE_OVERLOADED", "retryable": true}`,
  with `status` `null` when the provider gave none.
- **Message**: the `detail` the same failure would carry as an HTTP error where
  that names a cause (a rejected parameter, the provider's own explanation of a
  rejected request, redacted as described above, or what happened upstream,
  such as the provider's error status), and a fixed
  `An error occurred during streaming` otherwise.
- The gateway records the request's usage row with status `error`.

When a stream counts as finished:

- **Chat Completions**: once a choice reports its `finish_reason`. One that
  closes without any, even after sending content, ends in the error event
  rather than in a bare `data: [DONE]`.
- **Responses, from a provider with its own Responses API**: once the provider
  sends `response.completed`, `response.incomplete` or `response.failed`. One
  that closes before them ends in the error event.
- **Responses
  [translated from a chat completion](#responses-on-providers-without-a-responses-api)**:
  once a chunk reports its `finish_reason`, ending in `response.completed`, or
  `response.incomplete` when the provider stopped for length or a content
  filter. One that closes without any ends in the error event.
- **Messages**: a stream the provider ends without `message_stop`, including
  one that sent no events at all, ends in an `api_error` event, and the request
  is settled as failed. A Messages error event's `error.type` says whether a
  retry makes sense: `overloaded_error` and `rate_limit_error` when the
  provider failed the stream that way, `timeout_error` when it stopped
  responding (a read timeout, such as Bedrock's, or an upstream `504`), and
  `api_error` otherwise.

## Caller-orchestrated MCP

`GET /api/v1/mcp/servers` lists the enabled stored servers the key can reach,
`GET /api/v1/mcp/servers/{mcp_server_id}/tools` lists the tools a stored MCP
server exposes to the authenticated workspace, and `POST /api/v1/mcp/execute`
runs one exact caller-authorized call and returns the server's native MCP
result. Never retry `POST /api/v1/mcp/execute` automatically, including from a
reverse proxy or service mesh: an `outcome_unknown` response means the tool may
already have run. The request shapes, the error and execution-state contract,
and the limits are in [Caller-orchestrated MCP](mcp.md#caller-orchestrated-mcp).

## Agent traces

The sessions [Sessions](sessions.md) describes are read on these routes.

| Endpoint | Who |
| --- | --- |
| `GET /api/v1/organizations/me/traces`, `/count`, `/series`, `/{trace_id}`, `/requests/{request_id}` | a signed-in member: an owner or admin reads every workspace of their active organization, anyone else the workspaces they belong to; also the master key in the `Otari-Key` header, as the deployment operator in their active organization |
| `GET /api/v1/traces`, and the same four | the deployment operator, every workspace of their active organization |

`/series` counts the sessions active in each bucket of a window: a session is
active from its start to its last activity, so a long one counts in every bucket
it ran through. `bucket` is `1m`, `5m`, `30m`, `hour`, `6h` or `day`.
The list, count and series reads take the window the same way: `start` and
`end` keep the sessions active between them (last active at or after `start`,
started before `end`), so a session still running past `end` is listed in the
window it was counted in.

The list, count and series reads narrow to one workspace with `workspace_id`,
inside the caller's scope and never beyond it. `q` matches any part of a session's trace id, name,
agent or user, in any case, inside the same scope. A trace is keyed by its
workspace and its id, so a detail read takes the `workspace_id` its list entry
names; without it, an id in more than one of the caller's workspaces answers 409
rather than a guess.
`/requests/{request_id}` answers the session a gateway request was traced into,
by the `Otari-Request-ID` it was sent (a usage row's `request_group_id`), and 404
where the scope holds no trace of it. A trace outside the caller's scope answers
404.
