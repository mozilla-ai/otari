# API Reference

All endpoints are under `http://localhost:8000` by default.

For full request/response schemas, see the [OpenAPI spec](public/openapi.json) or the interactive docs at `/docs` when Otari is running.

## Provider error details

When an upstream provider fails, what the `detail` field contains depends on whose problem it is.

If the provider returned 400, 422 or 404 because it rejected **your request**, Otari responds with 400 or 404 and a sanitized provider diagnostic that names the parameter, model or limit at fault. Credential-shaped tokens, URLs, account identifiers, and reflected request or response payloads are stripped, and the diagnostic is capped at 400 characters.

If the failure is the **gateway's** (its provider credentials were rejected, the provider account is out of credit, the provider returned a 5xx, or the error could not be classified), the detail is a fixed string such as `The provider rejected the gateway's credentials`. There is no remedy you could apply, and the upstream text in those cases tends to name the operator's account rather than anything about your request. Operators should diagnose these with safe metadata (request ID, provider, model, and status) or a protected incident process, never by logging provider keys, internal URLs, prompts, responses, request bodies, or other user payloads.

## Endpoint availability

| Endpoint group | Standalone | Connected to otari.ai |
|---|---|---|
| Health (`/health*`) | Yes | Yes |
| Chat completions (`/v1/chat/completions`) | Yes | Yes |
| Messages (`/v1/messages`, `/v1/messages/count_tokens`) | Yes | Yes |
| Responses (`/v1/responses`) | Yes | Yes |
| Management (keys, users, budgets, aliases, routing policies, pricing, usage, tool discovery) | Yes | No |
| OpenAI-compatible (embeddings, models, files, batches, images, audio, moderations, rerank) | Yes | No |

## Authentication

### Standalone

- Preferred header: `Otari-Key: <token>` (a `Bearer` prefix is also accepted)
- `Authorization: Bearer <token>` is also accepted
- `x-api-key: <token>` is also accepted (for Anthropic-native clients)

Regular API endpoints use an API key. Management endpoints use the master key.

### Connected to otari.ai

The three generation endpoints (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`) expect `Authorization: Bearer <user-token>`. `Otari-Key` and local API keys are not used in this mode.

## Available in both deployment types

### Health

No authentication required.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | General health check. Includes otari.ai reachability fields when connected. |
| `GET` | `/health/liveness` | Kubernetes liveness probe. |
| `GET` | `/health/readiness` | Kubernetes readiness probe. Checks DB (standalone) or otari.ai reachability. Returns 503 on failure. |
| `GET` | `/metrics` | Prometheus metrics. Disabled by default; enable with `enable_metrics: true` in config. |

### Chat completions

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions. Supports streaming and tool use (`otari_code_execution`, `otari_web_search`, MCP). | Standalone: API key or master key. Connected: `Authorization` bearer token from otari.ai. |

For a full client setup example, see [Use with opencode](use-with-opencode.md).

### Messages

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/messages` | Anthropic Messages API-compatible endpoint. Supports streaming, tool use, extended thinking, and Anthropic context management (`context_management` and `betas`, including compaction). Routes to any provider in the catalog (non-Anthropic models are translated to/from the Messages format automatically). | Standalone: API key or master key. Connected: `Authorization` bearer token from otari.ai. |
| `POST` | `/v1/messages/count_tokens` | Anthropic-compatible input-token count for a Messages request. Returns `{"input_tokens": N}`. Counts locally (no provider call, no budget debit); the count is an approximation. `context_management` and `betas` are accepted for wire compatibility, but the local estimate does not apply provider-side context edits. Used by clients such as Claude Code for context-window management. | Standalone: API key or master key. Connected: `Authorization` bearer token from otari.ai. |

> `/v1/messages` uses the Anthropic Messages request shape regardless of which upstream provider serves the model. For example, `max_tokens` is still required even when `model` is `openai:...`.

For a full client setup example, see [Use with Claude Code](use-with-claude-code.md).

### Responses

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/responses` | OpenAI Responses API-compatible endpoint. Supports streaming. | Standalone: API key or master key. Connected: `Authorization` bearer token from otari.ai. |

Not every provider implements the Responses API; one that does not is rejected with `400 Provider '<name>' does not support the Responses API`. For a full client setup example, see [Use with Codex](use-with-codex.md).

## Standalone-only endpoints

### Embeddings

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/embeddings` | Generate embeddings for text input. | API key or master key |

### Models

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `GET` | `/v1/models` | List available models: auto-discovered from configured providers (when `model_discovery` is on, the default), plus configured pricing entries and aliases. | API key or master key |
| `GET` | `/v1/models/{model_id}` | Get a specific model. | API key or master key |

### Tools

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `GET` | `/v1/tools` | List the tools Otari runs itself, with the `tools[].type` values this deployment accepts, each tool's argument schema, and a runnable example. | API key or master key |

A tool with no backend configured is listed with `"available": false` rather than omitted, so a client can distinguish an unknown tool from an unconfigured one. `accepted_types` reflects the current configuration, so it grows when [web-search interception](tools.md#web-search-interception) is enabled.

Standalone-only. Connected to otari.ai the platform owns the per-workspace tool policy, so this gateway's own configuration is not the answer to "what can I call".

### Moderations

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/moderations` | OpenAI-compatible content moderation. | API key or master key |

### Rerank

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/rerank` | Reorder documents by relevance to a query. | API key or master key |

### Search

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/search` | Run a search against a configured search tool, named in `search_tool_name`. | API key or master key |
| `POST` | `/v1/search/{search_tool_name}` | Same, with the tool named in the path. | API key or master key |

Search tools are declared under [`search_tools`](configuration.md#search-tools)
in `config.yml`, or added at runtime through the dashboard's Tools & Guardrails
page and the `/v1/search-tools` endpoints below, which need no config file. This
is the direct counterpart to the `otari_web_search` tool:
the tool answers a model's search call mid-completion, while this endpoint takes
a query from the caller and returns results. Both forms log
`endpoint="/v1/search"`, so one Activity filter covers every search.

The request and response follow LiteLLM's `/v1/search` (itself shaped after
Perplexity's Search API), so a client moving off the LiteLLM proxy keeps its
request shape:

```bash
curl http://localhost:8000/v1/search/exa-search \
  -H "Otari-Key: Bearer $OTARI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "post-training quantization for small models",
    "max_results": 5,
    "search_domain_filter": ["arxiv.org"]
  }'
```

```json
{
  "object": "search",
  "search_tool": "exa-search",
  "results": [
    {
      "title": "…",
      "url": "https://arxiv.org/abs/…",
      "snippet": "…",
      "date": "2026-01-02T00:00:00.000Z"
    }
  ]
}
```

The accepted request fields are `query`, `search_tool_name`, `max_results`
(1 to 20), `search_domain_filter` (up to 20 entries; prefix a domain with `-` to
exclude it rather than restrict to it), `country`, `max_tokens_per_page`, and
`user`. Two differences from Perplexity are worth knowing before you migrate:
`query` must be a single string, so the multi-query array form is rejected with
a 422; and the filters Otari does not model (`search_recency_filter`,
`search_context_size`, the published-date filters) are ignored rather than
rejected, so check that your client does not depend on one. Provider-native
knobs with no request field, such as Exa's `type` or `category`, belong in the
tool's `options`.

The supported providers are `exa` and `searxng`, the latter being the same
SearXNG-shaped backend the `otari_web_search` tool uses, so a deployment that
already runs one can serve this endpoint from it without a commercial key. Which
request fields a `searxng` tool honors differs slightly; see
[serving the web-search backend you already run](configuration.md#serving-the-web-search-backend-you-already-run).

Search bills per request rather than per token, so a usage
row carries zero tokens and a cost taken from the provider's own reported charge
when it reports one (Exa does); otherwise it uses the flat per-request rate
configured for `<provider>:<tool>`, under the same convention as
[moderations](#moderations). Like moderations, search is exempt from
`require_pricing`. Configuring the flat rate is still
[recommended](configuration.md#search-tools): it is what gets reserved against
the caller's budget before the search runs.

A search the gateway itself refuses, an unknown or ambiguous `search_tool_name`
(400) or a tool the key's allowed-models list does not name (403), is written to
the usage log too, with a null cost, so refused searches are visible in Activity
and counted as failures rather than only in the caller's own logs.

#### Search tool management

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `GET` | `/v1/search-tools` | List search tools: the stored ones plus the read-only config-file ones. | Master key |
| `GET` | `/v1/search-tools/providers` | List the search providers this build supports, and what each one requires. | Master key |
| `POST` | `/v1/search-tools` | Add a search tool at runtime. | Master key |
| `PATCH` | `/v1/search-tools/{name}` | Update a stored search tool. Omitted fields are unchanged. | Master key |
| `DELETE` | `/v1/search-tools/{name}` | Delete a stored search tool. | Master key |
| `POST` | `/v1/search-tools/reencrypt` | Re-encrypt stored search-tool keys with the primary `OTARI_SECRET_KEY`. | Master key |

These are the runtime counterpart of the `search_tools` config block, and behave
the same way [runtime provider management](configuration.md#runtime-provider-management)
does: a stored tool wins over a config-file tool of the same name, API keys are
encrypted at rest and never returned (only `last4`), and a config-file tool
cannot be edited or deleted here. Standalone mode only.

### Images

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/images/generations` | Generate images from text prompts. | API key or master key |

Image generation bills per generated image, not per token, so a usage row carries
zero tokens and an `images` meter. Unlike audio and moderations it is subject to
`require_pricing`, so an unpriced image model is rejected with 402 under the
default configuration. See
[per-image pricing](configuration.md#per-image-pricing-image-generation) for how
to set the rate.

### Audio

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/audio/transcriptions` | Transcribe audio to text (multipart upload). | API key or master key |
| `POST` | `/v1/audio/speech` | Generate speech from text (TTS). | API key or master key |

Audio bills per request rather than per token, under the same convention as
[moderations](#moderations) and search, so a usage row carries zero tokens and a
cost taken from the flat rate configured for the model. Like moderations, audio
is exempt from `require_pricing`: with no rate configured the request is served
and logged at $0 with no charge line. See
[per-request pricing](configuration.md#per-request-pricing-audio-and-moderations)
for how to set the rate.

### Files

OpenAI-compatible file storage. Upload a file, then reference it from a chat
request by `file_id`. See [files.md](files.md) for how uploaded files are turned
into something a text-only local model can read.

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/files` | Upload a file (multipart: `file`, `purpose`). Returns a file object with an `id`. | API key or master key |
| `GET` | `/v1/files` | List the caller's files. Query params: `purpose`. | API key or master key |
| `GET` | `/v1/files/{file_id}` | Get file metadata. | API key or master key |
| `GET` | `/v1/files/{file_id}/content` | Download the raw file bytes. | API key or master key |
| `DELETE` | `/v1/files/{file_id}` | Delete a file. | API key or master key |

### Batches

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/batches` | Create an async batch of LLM requests. | API key or master key |
| `GET` | `/v1/batches` | List batches. Query param: `provider`. | API key or master key |
| `GET` | `/v1/batches/{batch_id}` | Get batch status. Query param: `provider`. | API key or master key |
| `POST` | `/v1/batches/{batch_id}/cancel` | Cancel a batch. Query param: `provider`. | API key or master key |
| `GET` | `/v1/batches/{batch_id}/results` | Get batch results. Returns 409 if not complete. Query param: `provider`. | API key or master key |

### Key management

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/keys` | Create an API key. Takes an optional `workspace_id`; the deployment default is used when it is omitted, and the key bills its workspace for every request. | Master key |
| `GET` | `/v1/keys` | List all API keys. | Master key |
| `GET` | `/v1/keys/{key_id}` | Get a specific key. | Master key |
| `PATCH` | `/v1/keys/{key_id}` | Update a key (name, active status, expiration, allowed models, `exclude_from_budget`, `reject_user_mismatch`, `capture_agent_telemetry`, metadata). | Master key |
| `POST` | `/v1/keys/{key_id}/rotate` | Replace a key's secret in place (id, user, name, expiry, and metadata preserved); returns the new key once. The previous secret stops working immediately. | Master key |
| `DELETE` | `/v1/keys/{key_id}` | Revoke a key. | Master key |

### User management

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/users` | Create a user. | Master key |
| `GET` | `/v1/users` | List users. | Master key |
| `GET` | `/v1/users/{user_id}` | Get a specific user. | Master key |
| `PATCH` | `/v1/users/{user_id}` | Update a user. | Master key |
| `DELETE` | `/v1/users/{user_id}` | Soft-delete a user and deactivate their keys. | Master key |
| `GET` | `/v1/users/{user_id}/usage` | Get usage history for a user. | Master key |

### Organizations

Tenancy above users and keys; see [Access control](access-control.md#organizations-and-workspaces). A standalone deployment has one organization, provisioned on the first master-key request, so every path is scoped to it and none names it. Creating, switching between and deleting organizations is what makes a deployment multi-tenant, and belongs to a hosted control plane rather than here.

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `GET` | `/v1/organizations/me` | Get the active organization and the caller's role in it. | Master key |
| `PATCH` | `/v1/organizations/me` | Rename the active organization. | Master key |
| `GET` | `/v1/organizations/me/members` | List the active organization's members. | Master key |
| `POST` | `/v1/organizations/me/members` | Add a member by email, optionally into workspaces at the same time. | Master key |
| `PATCH` | `/v1/organizations/me/members/{organization_member_id}` | Change a member's role or status. | Master key |
| `DELETE` | `/v1/organizations/me/members/{organization_member_id}` | Remove a member (suspends the membership). | Master key |

### Workspaces

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/workspaces` | Create a workspace in the active organization. | Master key |
| `GET` | `/v1/workspaces` | List the workspaces the caller can see. | Master key |
| `GET` | `/v1/workspaces/{workspace_id}` | Get a specific workspace. | Master key |
| `PATCH` | `/v1/workspaces/{workspace_id}` | Rename a workspace or change its description. | Master key |
| `DELETE` | `/v1/workspaces/{workspace_id}` | Delete a workspace and its memberships. Refused with 409 while it holds API keys, usage, aliases or routing policies, and refused for the last workspace. | Master key |
| `GET` | `/v1/workspaces/{workspace_id}/members` | List a workspace's members. | Master key |
| `POST` | `/v1/workspaces/{workspace_id}/members/{user_id}` | Add an organization member to a workspace. | Master key |
| `PATCH` | `/v1/workspaces/{workspace_id}/members/{user_id}` | Change a workspace member's role. | Master key |
| `DELETE` | `/v1/workspaces/{workspace_id}/members/{user_id}` | Remove a workspace member. | Master key |

### Budget management

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/budgets` | Create a budget. | Master key |
| `GET` | `/v1/budgets` | List budgets. | Master key |
| `GET` | `/v1/budgets/{budget_id}` | Get a specific budget. | Master key |
| `PATCH` | `/v1/budgets/{budget_id}` | Update a budget. | Master key |
| `DELETE` | `/v1/budgets/{budget_id}` | Delete a budget. | Master key |

The rows above cap one user. The family below caps a tenancy scope instead, and
both are enforced: a request must pass every ceiling that applies to it. See
[Access control](access-control.md#workspace-scoped-spend).

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/scoped-budgets` | Create a ceiling on an organization, workspace, membership, or API key, optionally narrowed to one provider. Its period is a rolling `budget_duration_sec` or a UTC-aligned `reset_alignment`, never both. Refused with 409 if that scope already has one. | Master key |
| `GET` | `/v1/scoped-budgets` | List scoped budgets, optionally filtered by `scope_type` and `scope_id`. | Master key |
| `GET` | `/v1/scoped-budgets/{budget_id}` | Get a specific scoped budget. | Master key |
| `PATCH` | `/v1/scoped-budgets/{budget_id}` | Update a scoped budget's label, limit, or period. The scope and the provider narrowing are fixed. | Master key |
| `DELETE` | `/v1/scoped-budgets/{budget_id}` | Delete a scoped budget. | Master key |

### Aliases

An alias is a display name that resolves to one real model selector. See
[Model aliases](models.md#model-aliases).

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `GET` | `/v1/aliases` | List every alias in force, from `config.yml` and from storage, in every scope. | Master key |
| `POST` | `/v1/aliases` | Create or update a stored alias. Omit `user_id` for a global one. | Master key |
| `DELETE` | `/v1/aliases/{name}` | Delete a stored alias. `user_id` query param selects the scope; omit it for the global one. Aliases from `config.yml` cannot be deleted here. | Master key |

### Routing policies

A policy is the general form of an alias: it decides which real model serves a
request, what is tried after a retryable failure, and which guardrails always run.
See [Routing policies](routing.md).

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `GET` | `/v1/routing/policies` | List every policy in force, from `config.yml` and from storage, in every scope. | Master key |
| `POST` | `/v1/routing/policies` | Create or update a stored policy. Body is `{name, spec, user_id?, rename_from?}`; `spec` is the same document a `routing.policies` entry takes. Omit `user_id` for a global one. `rename_from` renames an existing policy in the same scope to `name` on the same write: 404 if it does not exist, 409 if `name` is already taken. | Master key |
| `POST` | `/v1/routing/policies/explain` | Compile a policy and return the plan without dispatching anything. Takes a saved `name`, an unsaved draft `spec`, or both (the draft wins). Optional `user_id`, `key_id`, `allowed_models`, `budget_used_pct`, `budget_remaining_usd` simulate the request. Returns the ordered candidates **and** the ones that were dropped, with reasons. For a weighted policy it also returns `router_weights`, the share each candidate takes once filtering is applied. | Master key |
| `DELETE` | `/v1/routing/policies/{name}` | Delete a stored policy. `user_id` query param selects the scope. Policies from `config.yml` cannot be deleted here. | Master key |

Master key on every verb, including `explain`: the response enumerates a policy's
targets, which is what a policy exists to keep off the wire.

### Learned routing

Teaching the router a policy can name with `select: [{router: knn, candidates: [...]}]`.
Routing memory is per user, so `user_id` names whose it is. There is deliberately no
endpoint that fans a prompt out to the candidates for you: seeing what each answers is
what `POST /v1/chat/completions` already does, and going through it means those calls
are budget-checked and logged. See
[learned routing](routing.md#let-a-router-choose-learned-routing).

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/routing/preferences/rank` | Record a batch of scored `examples` under `user_id`: each has a `prompt`, `scores` (selector to quality in `[0, 1]`), an optional `task_id` partition, and an optional `label_source`. Writes the examples the router votes over and returns each touched pool's progress toward the seed count. Up to 100 per call. A score key that no learned policy could route to is refused, because such records are unmatchable and cannot be deleted. | Master key |
| `GET` | `/v1/routing/status` | For `user_id`: records and warmth per pool (the default pool plus each task partition), the router's tuning, and which policies depend on it. | Master key |

### Pricing

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/pricing` | Set or update model pricing. | Master key |
| `GET` | `/v1/pricing` | List all model pricing. | API key or master key |
| `GET` | `/v1/pricing/{model_key}` | Get effective pricing for a model. Optional `as_of` query param. | API key or master key |
| `GET` | `/v1/pricing/{model_key}/history` | Get full pricing history for a model. | API key or master key |
| `DELETE` | `/v1/pricing/{model_key}` | Delete a pricing entry. | Master key |

### Usage

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `GET` | `/v1/usage` | List usage logs. Filters: `start_date`, `end_date`, `user_id`, `status`, `status_code`, `model`, `endpoint`, `provider`, `source`, `source_label`, `api_key_id`, `request_group_id`. `user_id`, `model`, and `api_key_id` are repeatable (up to 50 values each) and match any of the values given. `status_code` is the HTTP status classifying a failure (e.g. 429 provider rate limit, 402 missing pricing); only error rows carry one, so filtering by it also restricts to `status=error` unless `status` is passed explicitly. `request_group_id` is repeatable and returns a routed request's whole attempt plan (see [Routing](routing.md)). | Master key |
| `GET` | `/v1/usage/count` | Total rows matching the filters (paginator total). Same filters as `GET /v1/usage`, so a multi-value filter counts the same rows the list returns. | Master key |
| `GET` | `/v1/usage/summary` | Aggregated spend/volume: totals, breakdowns by model/user/key/source/session/endpoint/provider, the failure taxonomy in `errors_by_status_code` (failures grouped by `status_code` with a coarse `error_class`), and a time series. `dimensions` narrows which breakdowns are computed (each one is a separate `GROUP BY`, including `status_code` for the taxonomy); `dimensions=none` returns totals and series only. `model`, `user_id`, and `api_key_id` are repeatable (up to 50 values each) and match any of the values given, so one call can compare a set of models, users, or keys. | Master key |
| `GET` | `/v1/usage/summary.csv` | Every breakdown as a CSV download. Same filters as `/v1/usage/summary`, including the repeatable `model` / `user_id` / `api_key_id`. | Master key |
| `GET` | `/v1/usage/series` | One time series per group, for stacked charts. `group_by` is required (`model`, `user_id`, `api_key_id`, or `source`). Same filters and window bounds as `/v1/usage/summary`, including the repeatable `model` / `user_id` / `api_key_id` (up to 50 values each, matching any of them). Returns the window's top eight groups by spend, with everything past them folded into one `other` series per bucket, so the stack reconciles with the summary totals. Points are sparse (populated cells only), and an `hour` bucket over a window of more than 1000 buckets is rejected with a 422 rather than returning an oversized payload. | Master key |
| `POST` | `/v1/usage/external-events` | Import externally-observed usage (e.g. Claude Code) as source-tagged rows, priced at API rates, never counted toward budget. An API key (must be budget-exempt) attributes to its own user; the master key may name any user. Idempotent by `(source, source_event_id)`. See [Importing external usage](external-usage.md). | API key (budget-exempt) or master key |
| `POST` | `/v1/traces` | OTLP receiver for GenAI usage **spans** (protobuf or JSON). Maps the OpenTelemetry GenAI conventions (`gen_ai.*`, `otari.*`) onto external usage ingestion. Any instrumented app can ship here. See [Importing external usage](external-usage.md). | API key (budget-exempt); master key refused |
| `POST` | `/v1/logs` | OTLP receiver for GenAI usage **log events** (protobuf or JSON), including Claude Code's `api_request` and Codex's `codex.sse_event` / `codex.api_request`. Same mapping as `/v1/traces`. See [Importing external usage](external-usage.md). | API key (budget-exempt); master key refused |
| `POST` | `/v1/metrics` | OTLP receiver for coding-agent outcome **metrics** (protobuf or JSON). Records the content-free counters Otari has no other source for (lines of code changed, commits, pull requests, active time); token/cost metrics are skipped as already billed, and edit decisions as already captured. Never billable. See [Use with Claude Code](use-with-claude-code.md). | API key (budget-exempt); master key refused |

### Agent telemetry

Content-free coding-agent telemetry: behavioral events from `POST /v1/logs` and
outcome metrics from `POST /v1/metrics`, in one table. Never billable, and gated by
the `capture_agent_telemetry` setting (per key, or deployment-wide).

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `GET` | `/v1/agent-telemetry/summary` | Outcomes (commits, pull requests, lines changed, active time), behavior (tool calls and their mix, accept/reject, turns, API errors), the recorded spend over the same scope, and the derived measures: cost per commit / pull request / line, spend per active hour, acceptance rate, turns per session, error rate. Each measure is `null` when its denominator is zero. Range-bounded like `/v1/usage/summary` (default last 30 days, capped at 366). Filters: `start_date`, `end_date`, `user_id`, `api_key_id` (the last two repeatable, up to 50 values each), `session_label` (scopes both the telemetry rows and the usage they are divided by), `bucket`. Cumulative counters are converted to window increments at read time, per series generation, so a re-exported total is never double counted and a counter reset never reads as negative. | Master key |
| `GET` | `/v1/agent-telemetry/count` | Total rows matching the filters (`start_date`, `end_date`, `user_id`, `api_key_id`, `name`), mirroring `/v1/usage/count`. Same filter set as the purge below, so it sizes exactly what a "delete all N matching" would remove. | Master key |
| `GET` | `/v1/agent-telemetry/series` | Row volume over time split by one dimension, for stacked charts. `group_by` is required (`user_id` or `api_key_id`). Top eight groups plus a reconciling `other` fold; sparse points; an `hour` bucket over more than 1000 buckets is rejected with a 422. | Master key |
| `DELETE` | `/v1/agent-telemetry` | Purge rows by explicit `ids` or, with `by_filter: true`, by `user_id` / `api_key_id` / `name` / date range. Covers behavioral and metric rows alike. A selection matching zero rows succeeds with `deleted: 0`. | Master key |
