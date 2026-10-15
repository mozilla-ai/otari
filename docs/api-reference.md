# API reference

Otari serves an OpenAPI document at `/api/v1/openapi.json` and interactive API
docs at `/api/v1/docs` by default. The repository also commits the generated
[OpenAPI specification](public/openapi.json) and
[Postman collection](public/otari.postman_collection.json). Those generated
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
request; a standalone gateway mints its own. A standalone gateway also names the
provider instance and model that served the request, whether a fallback did, and
how long it took, in `Otari-Provider`, `Otari-Model`, `Otari-Attempt-Count`,
`Otari-Fallback` and `Otari-Response-Duration-Ms`; see
[What is billed and what the caller sees](routing.md#what-is-billed-and-what-the-caller-sees).

A priced response also carries its cost on the usage object it already returns,
as `usage.cost_usd` (a six-decimal USD string) and `usage.pricing_source`. The
standalone [rerank](#rerank) and embeddings endpoints carry the same two fields
on their `usage`. On a stream the fields ride the terminal usage event: for Chat
Completions, the last chunk that reports usage and either has no choices (OpenAI's
`stream_options.include_usage` chunk) or finishes a choice (providers such as Cohere
and Gemini report usage on the chunk carrying `finish_reason`); `message_delta` for
Messages; and `response.completed` for Responses. The two fields always appear
together, and an unpriced or unreported request carries neither.

In standalone mode the amount is the one the gateway wrote to its own usage
record, including any gateway-run tool charges, and `pricing_source` names the
rate that priced the model: `organization` (an organization's override),
`deployment` (a rate stored on this gateway), or `defaults` (the bundled or
accepted models.dev snapshot). Hybrid mode attaches the platform's settlement instead;
see [Hybrid mode protocol](hybrid-mode-protocol.md#inline-response-fields).

Each request's cost is settled to the micro-dollar, half-up, so a request that
costs less than half a micro-dollar, such as a few tokens of a cheap embedding
model, is recorded, debited and reported as `"0.000000"`. The usage row and
the inline field always agree, and `pricing_source` still says the request was
priced.

### Who runs a built-in tool

A request that declares code execution, web search or Anthropic's web fetch in a
provider's own words (`code_execution_<date>`, `code_interpreter`,
`web_search_<date>`, `web_fetch_<date>`) can say who runs it with the
`Otari-Code-Execution`, `Otari-Web-Search` and `Otari-Web-Fetch` headers. Each
takes `auto`, `otari` or `provider`, and each attempt of a fallback chain is
decided on its own. `auto` keeps the provider's own tool on an attempt whose
model runs it natively and runs Otari's on one that does not. `otari` always
runs Otari's. `provider` always forwards the provider's own tool and never uses
Otari's, so on a fallback to a provider that cannot run it, that attempt fails;
use `auto` for fallback cover. See [Who runs a tool](tools.md#who-runs-a-tool).

### Web tool calls on Chat Completions

Chat Completions has no shape for a tool the server runs, so a chat completion
reports Otari's own web searches and fetches in `otari_tool_calls`, at the top
level of the response, beside `choices`. Each entry is one call:

| Field | Value |
|---|---|
| `id` | The call's ID: `otari_ws_…` for a search, `otari_wf_…` for a fetch. |
| `type` | `web_search` or `web_fetch`. |
| `status` | `in_progress`, `completed` or `failed`. |
| `query` | A search's query. |
| `url` | The URL the model asked a fetch for, without its query string and fragment. A redirect does not change it. |
| `sources` | A completed search's results, each with its `url`, `title` and `published_date` (the provider's own date string, or `null`); empty when it found none. Never page text. |
| `error` | Why a failed call failed, when Otari can say: `provider`, `status`, `tag` and `retryable`, the same four fields the [Search](#search) endpoint's `Otari-Error-*` headers carry. A call refused past its declaration's `max_uses` has `"tag": "call_limit"` and a `null` provider. Absent when Otari can say no more than that the call failed. |

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
data: {"id":"chatcmpl-…","choices":[{"delta":{"content":null,…},"finish_reason":null,"index":0}],…,"otari_tool_calls":[{"id":"otari_ws_5f0c…","type":"web_search","status":"in_progress","query":"latest python release"}]}

data: {"id":"chatcmpl-…","choices":[{"delta":{"content":null,…},"finish_reason":null,"index":0}],…,"otari_tool_calls":[{"id":"otari_ws_5f0c…","type":"web_search","status":"completed","query":"latest python release","sources":[…]}]}
```

The entries are not in the `delta`, where the OpenAI Python SDK's stream helper
(`client.chat.completions.stream()`) would fail on them. That helper ignores
them, so read the raw chunks to see them. A stream that has sent an entry has
sent response bytes, so a provider failure after it ends the stream with an
error event rather than falling over to another candidate (see [Failure
behavior](routing.md#failure-behavior)).

A client may keep the entries on the assistant message it sends back: Otari
removes `otari_tool_calls` from every assistant message before calling the
provider. Code execution and MCP tool calls are not reported in the field.

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

What reaches the provider depends on the surface. Chat Completions and Messages
never forward the caller's `metadata`. On Messages, `metadata.user_id` names the
billed user, as it always has, and is not a tag; the provider gets an
[end-user identifier](configuration.md#end-user-identifiers) derived from it in
its place. Responses forwards `metadata` as before, minus
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

### The `user` field

OpenAI's `user` field is a free-form end-user identifier, and OpenAI examples
often fill it with one. In Otari it names the user a request is billed to, so it
is not free-form. A request made with an API key may name only that key's own
user: any other value is refused with
`403 'user' field does not match the authenticated API key's user`. A key's
`reject_user_mismatch` (see [API keys](access-control.md#api-keys)), or the
deployment-wide setting of the same name, can accept other values instead, in
which case the value is ignored and spend stays with the key's user. A
[service key](access-control.md#service-keys-and-end-users) may name its own end
users, and a master-key request must name the user to bill. Chat Completions
and Responses never send `user` to the provider.

To tag a request with a free-form end-user ID, put it in `metadata`
(see [Request tags](#request-tags)) rather than in `user`:

```json
{"model": "openai:gpt-4o", "messages": [...], "metadata": {"end_user": "customer-42"}}
```

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
`Otari-Request-ID`, serving headers and `usage.cost_usd`, and an `Otari-Idempotent-Replayed: true`
header (but no `Otari-Response-Duration-Ms`, which described the original), without calling the provider or billing again. While the original is
still running, a retry is answered 409 with `Retry-After`, and if the original
fails the next retry runs in its place.

- A key belongs to the API key that sent it (or, for the master key, to the
  billed user), so two callers never see each other's responses.
- A retry has to be the same request: the same body, and the same
  `Otari-Code-Execution`, `Otari-Web-Search`, `Otari-Web-Fetch`, `Otari-Router`,
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
[Per-request pricing](configuration.md#per-request-pricing-audio-moderations-rerank-and-completion-models).

## Search

`POST /api/v1/search` and `POST /api/v1/search/{search_tool_name}` run a configured
search tool directly. This is separate from `otari_web_search`, which lets a
model request searches during a completion. Both are described in
[Built-in tools](tools.md).

The request and the response follow LiteLLM's `/v1/search`, so a client written
for it works against Otari with its base URL changed. `include_raw` and
`provider_options` are Otari's own.

**Which tool.** The path names the tool. On `POST /api/v1/search`, the body's
`search_tool_name` does, else the deployment's `web_search_default_tool`, else
the only configured tool. A call that names no tool runs on the workspace's
[organization search key](tools.md#organization-search-keys) when it has a usable
one; a named call runs on that tool with its own key. Either way the tool decides
the key allow-list entry (`<provider>:<tool>`), the `rate_limits` rules, the
timeout and the usage row's model and provider.

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
400 on a `searxng` tool and on a call a Tavily or Brave organization key serves;
`false` and `{}` pass.

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

**Errors.** Where a refusal carries a stable code, it is in the
`Otari-Error-Code` header and the body's `code`; see [Error codes](#error-codes).

| Situation | Status | `Otari-Error-Code` |
|---|---|---|
| No credentials, or a key that is not valid | 401 | |
| No tool is configured, the tool named is not, or several are and the call names none with no default | 400 | |
| The tool cannot run as configured, such as a `searxng` tool with no address | 400 | |
| A `provider_options` key the provider does not take from a request | 400 | |
| `include_raw` or `provider_options` on a call that does not run on any-search yet | 400 | |
| The master key without a `user` | 400 | |
| The body fails schema validation, such as an `include_raw` that is not a boolean | 422 | `invalid_request` |
| The workspace turned web search off | 403 | |
| The key's allow-list does not name `<provider>:<tool>` | 403 | |
| A `user` naming another user, where the key rejects a mismatch | 403 | |
| A service key's `Otari-End-User-Budget` is not one it may use | 403 | `end_user_budget_not_allowed` |
| A budget refuses the search | 403 | `budget_exceeded` |
| A rate limit is full | 429 | `rate_limited` |
| The workspace's organization search key cannot be used as it is | 502 | |
| The provider failed the search, or could not be reached | 502 | |
| The workspace's stored web-access policy cannot be read | 503 | |

Every provider failure is a 502 whose `detail` names no upstream detail, whatever
the provider answered, unlike the completion routes, which classify a provider's
failure. The 502 carries what is safe to show of the failure as headers instead,
never a body, a URL or the query:

| Header | Value |
|---|---|
| `Otari-Error-Provider` | The provider that failed, such as `exa`. |
| `Otari-Error-Status` | The provider's own HTTP error status; absent when it gave none, as for a timeout, an error inside a successful answer, or an answer that could not be read. |
| `Otari-Error-Tag` | What failed, in the provider's terms or the library's: `timeout`, `network`, `invalid_header` (a key holding a character that is not ASCII), `http_error`, `invalid_response`, `unexpected_response`, or the provider's own code, such as Exa's `INVALID_API_KEY`. |
| `Otari-Error-Retryable` | `true` for a timeout, a 429 or a 5xx from the provider, when a retry may help; otherwise `false`. |

On a search the workspace's organization key serves, they name the key's
provider. A 502 for a failure that is not the provider's, such as an organization
key this gateway cannot call, carries none of them.

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
501 for a model that cannot answer it) returns 400, its rate limit returns 429, its
refusal of the gateway's credential (401 or 403) returns 424 `provider_credential_rejected`,
and any other failure returns 502.

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

The diagnostic is the provider's own sentence, taken from the error body it
sent, and not the provider SDK's rendering of that error. An OpenAI 400 for a
model that only accepts its default temperature answers:

```json
{"detail": "Unsupported value: 'temperature' does not support 0.5 with this model. Only the default (1) value is supported."}
```

rather than a detail that begins `Error code: 400 - {'error': {...}}`. The same
holds for Anthropic, Cohere and the other providers whose SDKs keep the parsed
body: only the body's message can reach the caller, never a status prefix, the
body's other fields, or the response headers. When that body has no message, or
its message is redacted away, the caller gets the fixed detail for the status
instead (for a 400,
`The provider rejected the request as invalid (check the model name and parameters)`).
The provider's `param` and `code` fields are not copied into the response.

The completion routes and the pass-through routes (embeddings, images, audio,
moderations, rerank) classify a provider failure the same way. A provider that
rejects the request (its own 400 or 422) answers 400, and an unknown model 404.
OpenAI answers some request faults with a 403 instead, with the error type
`invalid_request_error` (logprobs requested from a model that does not return
them, an embedding model sent to chat); that is a rejection of the request too,
and answers 400 with OpenAI's sentence.
A model whose provider has no such operation (rerank on a chat model, for
example) or whose SDK has no parameter for a field the caller sent answers 400,
naming the operation or the field. An OpenAI-compatible client does not retry a
400 or a 404. The provider's rate limit answers 429 with
`Otari-Error-Code: upstream_rate_limited` and the provider's `Retry-After` when
it sent one, which a client may retry after waiting. A provider that refuses the
gateway's own credential (its 401, or a 403 that is not a rejection of the request) answers 424 with
`Otari-Error-Code: provider_credential_rejected`, which a client does not retry
either. A provider timeout answers 504, and any other provider failure 502
(500 on Messages), whose detail says what happened upstream and nothing more:
`The provider failed the request with HTTP 500` (the provider's own status),
`The provider closed the connection before answering`, or
`The gateway could not connect to the provider`. A failure Otari cannot place
keeps the fixed `LLM provider error`.
A non-streamed Chat Completions response with no choices is a provider failure
too, not an answer: it is refused with a 502,
`The provider returned a completion with no choices` and the code
`provider_error`, recorded on the usage row as an error, and a routing policy
moves on to its next candidate. Inside a gateway tool loop this applies to the
model's first response; a later one is returned as the answer, since the rounds
before it were already billed. Messages and Responses are not checked: an empty
`content` or `output` can be a legitimate answer there.
When the request named a model alias,
a provider message that names the alias's target names the alias instead.

A 5xx that resending the same request cannot fix also carries
`x-should-retry: false`, the header the OpenAI and Anthropic SDKs read before
their own retry rule, so such a client stops after one attempt instead of
retrying with backoff. Two failures qualify: a provider account out of credit
or over its billing limit, and a fault in the code making the provider call,
such as a provider adapter that cannot convert the request, which raises before
anything is sent and has no upstream status. A timeout, a dropped connection, a
provider's own 5xx or 429, and any failure Otari cannot place carry no such
header, so a client keeps retrying those as before. A streamed request that
fails before its first chunk carries it too, since that is answered with an
ordinary HTTP error (see [Error codes](#error-codes)). A routing policy still
moves on to its next candidate on either failure: the header describes only the
answer the caller receives, and a policy that tried several candidates and saw
every one fail answers without it.

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
| `budget_exceeded` | 403 | A budget refused the request | `Otari-Budget-Scope`: `user` for the billed user's own budget, otherwise the ceiling's scope: `organization`, `workspace`, `workspace_member`, `org_member` or `api_token`; `Retry-After`: seconds until the refusing budget's window ends, when it resets and the request would fit a fresh one |
| `user_blocked` | 403 | The billed user is blocked | |
| `user_not_found` | 404 | The billed user does not exist | |
| `rate_limited` | 429 | A gateway rate limit is full | `Otari-Rate-Limit-Rule` for a `rate_limits` rule; `Retry-After` when waiting helps |
| `upstream_rate_limited` | 429 | The provider rate limited the gateway | `Retry-After` when the provider sent one |
| `invalid_model` | 400 | The model selector names no configured provider, alias, policy or catalog model id; checked before the key's allow-list, so a restricted key gets it too, and before `require_pricing`, so it is never answered as `pricing_required`. A `provider:model` selector whose provider resolves is not judged here, since only the provider knows its models | |
| `model_not_allowed` | 403 | The key may not use the model | |
| `model_not_serving` | 403 | The organization offers the model on a provider key, with its serving switch off | |
| `model_not_found` | 404 | A provider key of the organization serves the provider, and none offers the model | |
| `context_length_exceeded` | 400 | The prompt is too long for the model | |
| `unsupported_content` | 400 | The provider refused content the request carries because the model does not accept it, such as an image sent to a text-only model. Sent only when Otari recognizes the provider's wording for that refusal; a refusal worded otherwise stays a `400` carrying the provider's message and no code. `detail` names the model where it can. A routing policy moves on to its next candidate first, and answers with this code when every candidate gave a refusal Otari recognizes | |
| `all_candidates_rejected` | 400 | Every model a routing policy, a catalog ID or the control plane tried rejected the request as invalid | |
| `insufficient_funds` | 402 | The model is served on a credential the deployment pays for, and the organization has no funds paid in to cover it. Nothing was sent upstream | |
| `pricing_required` | 402 | `require_pricing` is on and the model resolves but has no price. An unpriced model whose provider has no credential answers `provider_not_configured` instead | |
| `provider_not_configured` | 424 | The gateway holds no credential for the model's provider, so nothing was sent upstream. Not a 5xx, so a client does not retry it: add the credential in `config.yml`, through the dashboard, or in the environment variable the detail names | |
| `provider_credential_rejected` | 424 | The provider refused the credential the gateway holds for it (an upstream 401, or a 403 other than one rejecting the request, which is a 400). It is the gateway's key, not the caller's, so it is never a 401 or 403; and not a 5xx, because no retry makes the key valid. Replace or fix the provider credential. The usage row keeps the provider's own status. A routing policy moves on to its next candidate first, and answers with this code when every candidate either had no credential or had it rejected, and at least one was rejected | |
| `end_user_budget_not_allowed` | 403 | A service key named an end-user budget that is not on its `end_user_budget_ids` | |
| `terms_not_accepted` | 400 | The deployment publishes terms (`terms_url`), and creating or first claiming the account needs them accepted: send `terms_accepted` with the signup, the invitation's first password, the password reset, or the OAuth callback | |
| `invalid_request` | 422 | The request body or a parameter failed schema validation. `detail` lists the failing fields, each with `loc`, `msg` and `type` | |
| `provider_error` | 502, or 200 in a stream's error event | As a 502, the provider answered a non-streamed chat completion with no choices. In a stream's error event, the provider failed after the stream started, for a reason no other code names | |
| `provider_stream_interrupted` | 200, in a stream's error event only | The provider's stream ended before it finished: it closed without reporting a `finish_reason`, or the connection carrying it was cut. The output received so far is truncated | |

A `budget_exceeded` refusal's `detail` tells a spent budget (`... has exceeded
budget limit`) from a request whose estimated cost is larger than the budget it
has left (`... has $X of budget remaining, less than this request's estimated
cost of $Y`); see [Budgets](access-control.md#budgets). Its `Retry-After` is
when the budget resets, which may be days away for a monthly budget; see
[Refusals and `Retry-After`](access-control.md#refusals-and-retry-after).

A streamed request whose provider fails before producing any output is answered
with an ordinary HTTP error, with the same status, `detail` and code the request
would get unstreamed: a `400` for a parameter the provider does not accept, a
`502` for a provider outage. Otari holds the response until the provider's first
chunk to make that possible. A standalone gateway holds it for at most
`streaming_first_chunk_hold_ms` (15 seconds by default; `0` turns the hold off
and commits the `200` as soon as the stream opens), and does not hold a stream
that runs gateway tools (MCP, code execution, web search). A provider slower than
that to start still streams, and a failure after that point is reported as
below. A hybrid gateway holds every stream for its first-chunk failover timeout.

A failure after a stream has started arrives as an error event, which carries
the code as `error.code` on Chat Completions and Responses:
`{"error": {"message": "...", "type": "server_error", "code": "upstream_rate_limited"}}`.
The status line has already gone out as `200` by then, so the event is the only
place the failure is named. It carries the code from the table above when one
applies (`upstream_rate_limited`, `context_length_exceeded`),
`provider_stream_interrupted` when the provider's stream ended before it
finished, otherwise `provider_error` when the provider failed the stream. On
Chat Completions a stream is finished once a choice reports its
`finish_reason`; one that closes without any, even after sending content, ends
in this error event rather than in a bare `data: [DONE]`. Its message is the `detail` the same failure would carry as an HTTP
error where that names a cause (a rejected parameter, the provider's own
explanation of a rejected request, redacted as described above, or what
happened upstream, such as the provider's error status), and a fixed
`An error occurred during streaming` otherwise. An event with no `code` is a
failure inside the gateway. Either way, a standalone gateway records the
request's usage row with status `error`.
On Messages, a stream the provider ends without `message_stop`, including one
that sent no events at all, also ends in an `api_error` event rather than
closing as if it had succeeded, and the request is settled as failed.

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
