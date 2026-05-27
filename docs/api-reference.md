# API Reference

All endpoints are under `http://localhost:8000` by default.

For full request/response schemas, see the [OpenAPI spec](public/openapi.json) or the interactive docs at `/docs` when the gateway is running.

## Endpoint availability

| Endpoint group | Standalone | Connected to otari.ai |
|---|---|---|
| Health (`/health*`) | Yes | Yes |
| Chat completions (`/v1/chat/completions`) | Yes | Yes |
| All other `/v1/*` endpoints in this doc | Yes | No |

## Authentication

### Standalone

- Preferred header: `Otari-Key: Bearer <token>`
- Back-compat headers: `AnyLLM-Key`, `X-AnyLLM-Key`
- `Authorization: Bearer <token>` is also accepted

Regular API endpoints use an API key. Management endpoints use the master key.

### Connected to otari.ai

- `POST /v1/chat/completions` expects `Authorization: Bearer <user-token>`
- `Otari-Key` and local API keys are not used for this path

## Available in both deployment types

### Health

No authentication required.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | General health check. Includes otari.ai reachability fields when connected. |
| `GET` | `/health/liveness` | Kubernetes liveness probe. |
| `GET` | `/health/readiness` | Kubernetes readiness probe. Checks DB (standalone) or otari.ai reachability. Returns 503 on failure. |

### Chat completions

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions. Supports streaming and tool use (`code_execution`, `web_search`, MCP). In standalone mode, omit `model`, set it to `null`, or set it to `default_routing` to use routing policies. | Standalone: API key or master key. Connected: `Authorization` bearer token from otari.ai. |

## Standalone-only endpoints

For a review-oriented map from these endpoints to Merge Gateway-style routing,
governance, attribution, and observability concepts, see
[Merge Gateway Compatibility](merge-gateway-compatibility.md).

### Messages

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/messages` | Anthropic Messages API-compatible endpoint. Supports streaming and extended thinking. | API key or master key |

### Responses

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/responses` | OpenAI Responses API-compatible endpoint. Supports streaming for provider-native calls and non-streaming `default_routing` when `model` is omitted, `null`, or set to `default_routing`. Non-streaming responses include canonical `model` (`provider/model`) and top-level `vendor`; all Responses calls include `X-Response-Model` and `X-Response-Vendor` headers. | API key or master key |

### Embeddings

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/embeddings` | Generate embeddings for text input. | API key or master key |

### Models

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `GET` | `/v1/models` | List available models from discovery/pricing. Defaults to OpenAI-compatible shape; use `format=gateway` for Merge-style model/vendor catalog metadata, or `model=provider/model` for one catalog object. | API key or master key |
| `GET` | `/v1/models/{model_id}` | Get a specific model. | API key or master key |
| `GET` | `/v1/vendors` | List Merge-style execution vendors and the models they can serve. | API key or master key |
| `GET` | `/v1/vendors/{vendor_id}` | Get one Merge-style execution vendor. | API key or master key |

### Moderations

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/moderations` | OpenAI-compatible content moderation. | API key or master key |

### Rerank

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/rerank` | Reorder documents by relevance to a query. | API key or master key |

### Images

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/images/generations` | Generate images from text prompts. | API key or master key |

### Audio

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/audio/transcriptions` | Transcribe audio to text (multipart upload). | API key or master key |
| `POST` | `/v1/audio/speech` | Generate speech from text (TTS). | API key or master key |

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
| `POST` | `/v1/keys` | Create an API key. | Master key |
| `GET` | `/v1/keys` | List all API keys. | Master key |
| `GET` | `/v1/keys/{key_id}` | Get a specific key. | Master key |
| `PATCH` | `/v1/keys/{key_id}` | Update a key (name, active status, expiration, metadata). | Master key |
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

### Budget management

Budgets can be attached to users and projects, or configured as tag-scoped budget groups. A tag-scoped budget uses
`scope_type: "tag"` and exact `match_tags` such as `{ "team": "platform" }` or
`{ "customer_tier": "enterprise" }`. Matching request tags are validated before provider dispatch, and matching usage
cost increments the budget group's `spend`. Budgets can also set `alert_thresholds` as spend ratios such as
`[0.5, 0.8, 0.95]`; when tracked spend crosses a threshold, the gateway records one durable alert event per budget
scope and budget period. If `alert_webhook_url` is configured, the gateway delivers the alert after the usage
transaction commits and stores delivery status for retry/inspection. Failed or pending deliveries can be retried
manually, or by enabling the background retry worker in gateway configuration. The worker respects retry backoff,
records `next_delivery_attempt_at`, and moves alerts to `delivery_status: "dead_letter"` once the configured max
attempt count is reached.

When `enable_metrics` is true, `/metrics` exposes budget alert observability via
`gateway_budget_alerts_created_total`, `gateway_budget_alert_webhook_deliveries_total`,
`gateway_budget_alert_webhook_delivery_duration_seconds`, `gateway_budget_alert_webhook_retry_runs_total`,
`gateway_budget_alert_webhook_retry_selected_total`, and `gateway_budget_alert_webhook_dead_letters_total`.

Tag-scoped budget example:

```json
{
  "max_budget": 1000,
  "budget_duration_sec": 2592000,
  "scope_type": "tag",
  "match_tags": {
    "team": "platform"
  },
  "alert_thresholds": [0.5, 0.8, 0.95],
  "alert_webhook_url": "https://hooks.example.com/budget-alerts"
}
```

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/budgets` | Create a user/project budget or tag-scoped budget group. | Master key |
| `GET` | `/v1/budgets` | List budgets. | Master key |
| `GET` | `/v1/budgets/alerts` | List budget alert events, optionally filtered by `budget_id`, `scope_type`, or `scope_id`. | Master key |
| `POST` | `/v1/budgets/alerts/{alert_id}/deliver` | Retry webhook delivery for a budget alert event. | Master key |
| `GET` | `/v1/budgets/{budget_id}` | Get a specific budget. | Master key |
| `GET` | `/v1/budgets/{budget_id}/alerts` | List alert events for a specific budget. | Master key |
| `PATCH` | `/v1/budgets/{budget_id}` | Update a budget. | Master key |
| `DELETE` | `/v1/budgets/{budget_id}` | Delete a budget. | Master key |

### Pricing

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/pricing` | Set or update model pricing. | Master key |
| `GET` | `/v1/pricing` | List all model pricing. | API key or master key |
| `GET` | `/v1/pricing/{model_key}` | Get effective pricing for a model. Optional `as_of` query param. | API key or master key |
| `GET` | `/v1/pricing/{model_key}/history` | Get full pricing history for a model. | API key or master key |
| `DELETE` | `/v1/pricing/{model_key}` | Delete a pricing entry. | Master key |

### Routing

The standalone gateway includes a self-hosted operator dashboard at `/admin`. The page is not part of the OpenAPI
schema; it uses the configured master key in the browser to call the same management APIs for routing policies,
projects, route traces, usage summaries, budget alerts, and policy revision rollback.

Omit `model`, set it to `null`, or set it to `default_routing` on `POST /v1/chat/completions`,
`POST /v1/responses`, or `POST /v1/routing/resolve` to resolve a request through the default routing policy or through
the policy attached to `project_id`. The `default_routing` sentinel is case-insensitive and trimmed before matching.
Standalone routing supports
non-streaming chat completions and non-streaming Responses requests; routed Responses requests are translated into chat
messages and returned in a Responses-shaped payload with the served `vendor` and canonical `provider/model` ID. Route
traces include the original gateway endpoint so `/v1/responses` traffic can be filtered and summarized separately from
`/v1/chat/completions`.

Routing policies accept the native `strategy` + `config` shape or a Merge-style `default_strategy` shape. A
`default_strategy` of `{"type": "fallback", "providers": [...]}` maps to `single` when one provider is configured and
to `priority` when multiple providers are configured; providers are tried in ascending `priority` order. A
`default_strategy` of `{"type": "intelligent", "axis": "cost" | "performance" | "intelligence", "providers": [...]}`
maps to the native `intelligent` strategy and stores axis-specific complexity thresholds. A
`default_strategy` of `{"type": "weighted_score", "providers": [...], "scoring": {...}}` maps to the native
`weighted_score` strategy. Native strategies are
`single`, `priority`, `lowest_cost`, `least_latency`, `cost_tier`, `intelligent`, and `weighted_score`. Candidates can be configured
either as a flat `candidates` list or as tiered `tiers` (`simple`, `medium`, `complex`, `reasoning`). Set
`config.fallback_enabled` to `false` when only the first selected candidate should be attempted.
The `least_latency` strategy uses recent successful route traces; `config.latency_sample_limit` controls how many
traces are inspected.
The `weighted_score` strategy ranks candidates by a normalized weighted score across configured quality, estimated
cost, and recent latency. Candidate `quality_score`, `benchmark_score`, `score`, or `intelligence_score` values may be
provided as `0..1` or `0..100` values. Configure `config.scoring.weights.quality`, `.cost`, and `.latency`, or use
flat `quality_weight`, `cost_weight`, and `latency_weight` keys. Dry-run responses and route traces include
`routing_score` and `score_components` for every scored candidate.
Use `POST /v1/routing-policies/{policy_id}/eval-scores` to ingest uploaded eval or benchmark rows into a
`weighted_score` policy. Duplicate rows for the same model are averaged by `sample_count`; matched candidates are
updated with `quality_score` and eval metadata, and the policy change is recorded as an immutable revision.
For scheduled or generated eval runs, use `scripts/apply_eval_scores.py` to normalize JSON, JSONL/NDJSON, or CSV eval
artifacts into the same endpoint payload. The script supports dry-run payload generation and can post directly to a
running gateway with `GATEWAY_MASTER_KEY`.
Non-default policies can also declare `config.match.tags` and will be selected before the default policy when a
request's `tags` contain all configured key/value pairs. For Merge-style tag routing, policies can also declare
`config.match.conditions` with `logic: "and" | "or"`, or nested `all`/`any` groups. Supported operators are `eq`, `ne`,
`gt`, `gte`, `lt`, `lte`, `in`, `not_in`, `contains`, `starts_with`, `ends_with`, and `exists`.
Matched policies can also set `config.match.rollout_percentage` (or `percentage`) for deterministic canary rollout.
The bucket is computed from request tags; set `config.match.bucket_by` to use one tag such as `tenant` or `user_id` as
the stable bucket key. Canary-selected policies appear as `policy_source: "canary_match"` in route traces and dry-run
responses.
Set `config.guardrails.enabled` to inspect routed request text before provider dispatch. Guardrails support exact
`blocked_terms`, named `blocked_patterns`, built-in PII checks (`email`, `ssn`, `credit_card`), and prompt-injection
phrase checks. The default action is `block`; set `action: "observe"` to surface violations in dry-run and route traces
without rejecting the request. Use `config.guardrails.presets` for managed OSS presets such as `pii`,
`prompt_injection`, `credential_leak`, `dlp`, `baseline`, and `strict`; presets expand to local classifiers before
provider dispatch and dry-run responses/route traces report which preset names were applied or ignored.
`config.guardrails.external_classifiers` can call HTTP classifier services before
provider dispatch. Each classifier receives `{ "text": "..." }` and may return `blocked: true`, `flagged: true`,
`violations`, or a numeric `score` compared with the classifier's configured `threshold`. Set `fail_closed: true` to
treat classifier transport or response errors as guardrail violations.
Use `config.guardrails.redactions` to mask sensitive routed request content before provider dispatch. Redactions can
use built-in PII patterns and named regex `patterns`; dry-run responses and route traces report replacement counts and
rule names without storing the original sensitive values.
Set `config.context.enabled` with `strategy: "trim_messages"` or `strategy: "summarize_messages"` and
`max_prompt_tokens` to compress long routed chat requests before provider dispatch. Trimming preserves
system/developer messages by default, preserves the newest `preserve_last_messages` turns, and drops older turns until
the estimated prompt fits. Summarization keeps the same preserved messages but replaces older turns with a deterministic
extractive summary message before dispatch. Dry-run responses and route traces expose context metadata without storing
the original dropped conversation text.
Use `config.constraints` to enforce governance before a provider is called: `allowed_providers`, `blocked_providers`,
`allowed_models`, `blocked_models`, `allowed_regions`, `blocked_regions`, `max_estimated_cost`, and
`allow_unknown_cost`. Candidates can declare `region` or `regions` metadata. Set `require_region_match: true` and
`region_tag` (default `region`) to require the request tag's region to be supported by the candidate before dispatch.
Set `config.health.enabled` to derive passive provider health from recent route traces. `config.health.mode` can be
`observe`, `downrank`, or `skip_unhealthy`; the default mode is `downrank`. Health tuning knobs include
`sample_limit`, `min_samples`, `degraded_failure_rate`, and `unhealthy_failure_rate`.
Use `POST /v1/routing/resolve` to dry-run a routing decision without calling a provider. The response includes the
selected model, ordered fallback candidates, rejected candidates, policy source, tier, estimated tokens, cost, latency
stats, weighted score components when used, and provider health details when enabled.
Routing policies maintain immutable revision history. `POST`, `PATCH`, and `DELETE` operations record snapshots with
an action, revision number, timestamp, and optional `change_note`; default-policy demotions are recorded as
`unset_default` revisions. Operators can apply a previous revision to roll a policy back; the rollback itself is
recorded as a new `apply_revision` entry instead of mutating history.
Routing policies also have a rollout `status`: `draft`, `active`, or `archived`. Only active policies can be selected
by live routing, attached to projects, or marked default. Draft and archived policies can still be inspected with
`POST /v1/routing/resolve` by passing `policy_id`, which lets operators preview staged policy changes without sending
traffic to them.

Native policy shape:

```json
{
  "name": "Default cost router",
  "strategy": "lowest_cost",
  "is_default": true,
  "status": "active",
  "config": {
    "fallback_enabled": true,
    "constraints": {
      "blocked_providers": ["openai"],
      "max_estimated_cost": 0.01,
      "require_region_match": true,
      "region_tag": "region"
    },
    "health": {
      "enabled": true,
      "mode": "downrank",
      "min_samples": 3,
      "unhealthy_failure_rate": 0.5
    },
    "match": {
      "any": [
        { "tag": "tenant", "operator": "eq", "value": "vip" },
        { "tag": "account", "operator": "starts_with", "value": "enterprise_" }
      ],
      "priority": 10,
      "rollout_percentage": 25,
      "bucket_by": "tenant"
    },
    "guardrails": {
      "enabled": true,
      "presets": ["dlp", "prompt_injection"],
      "pii": { "enabled": true, "types": ["email", "ssn"] },
      "prompt_injection": { "enabled": true },
      "blocked_terms": ["internal-only"],
      "external_classifiers": [
        {
          "name": "prompt-shield",
          "url": "https://classifiers.example.com/prompt-shield",
          "threshold": 0.75,
          "timeout_seconds": 2,
          "fail_closed": true
        }
      ],
      "redactions": {
        "enabled": true,
        "replacement": "[REDACTED]",
        "pii": { "enabled": true, "types": ["email", "ssn"] },
        "patterns": [
          { "name": "account_id", "pattern": "acct_[0-9]+" }
        ]
      }
    },
    "context": {
      "enabled": true,
      "strategy": "summarize_messages",
      "max_prompt_tokens": 6000,
      "preserve_system_messages": true,
      "preserve_last_messages": 4,
      "summary_max_tokens": 512
    },
    "candidates": [
      {
        "model": "openai:gpt-4o-mini",
        "input_price_per_million": 0.15,
        "output_price_per_million": 0.60,
        "regions": ["us", "eu"]
      },
      "anthropic:claude-3-5-haiku-latest"
    ]
  }
}
```

Weighted custom scoring:

```json
{
  "name": "Benchmark-weighted router",
  "strategy": "weighted_score",
  "is_default": true,
  "config": {
    "scoring": {
      "weights": { "quality": 0.7, "cost": 0.2, "latency": 0.1 },
      "default_quality_score": 0.5,
      "unknown_cost_score": 0,
      "unknown_latency_score": 0.5
    },
    "candidates": [
      {
        "model": "openai:gpt-4o",
        "quality_score": 0.95,
        "input_price_per_million": 5.0,
        "output_price_per_million": 15.0
      },
      {
        "model": "openai:gpt-4o-mini",
        "quality_score": 0.72,
        "input_price_per_million": 0.15,
        "output_price_per_million": 0.60
      }
    ]
  }
}
```

Eval score import:

```json
{
  "scores": [
    {
      "model": "openai:gpt-4o",
      "score": 92,
      "metric": "mt_bench",
      "sample_count": 50
    },
    {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "quality_score": 0.72,
      "metric": "internal_eval",
      "sample_count": 100
    }
  ],
  "change_note": "Import nightly eval results"
}
```

Generated eval pipeline:

```bash
/tmp/uv-bin/uv run python scripts/apply_eval_scores.py \
  --input evals/nightly.jsonl \
  --policy-id rp_123 \
  --metric nightly_eval \
  --change-note "Import nightly eval results"
```

Merge-style policy shape:

```json
{
  "name": "Quality-first router",
  "is_default": true,
  "default_strategy": {
    "type": "intelligent",
    "axis": "intelligence",
    "providers": [
      { "provider": "openai", "model": "gpt-4o-mini" },
      { "provider": "anthropic", "model": "claude-3-5-haiku-latest" },
      { "provider": "openai", "model": "gpt-4o" }
    ]
  }
}
```

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/routing/resolve` | Preview a `default_routing` decision without calling a provider. | Master key |
| `POST` | `/v1/routing-policies` | Create a routing policy. | Master key |
| `GET` | `/v1/routing-policies` | List routing policies. Optional `status` filter: `draft`, `active`, `archived`. | Master key |
| `POST` | `/v1/routing-policies/{policy_id}/clone` | Clone a routing policy into a draft for staged edits. | Master key |
| `GET` | `/v1/routing-policies/{policy_id}/revisions` | List immutable routing policy revisions. | Master key |
| `GET` | `/v1/routing-policies/{policy_id}/revisions/{revision}` | Get one routing policy revision. | Master key |
| `POST` | `/v1/routing-policies/{policy_id}/revisions/{revision}/apply` | Apply an old revision as a new audited policy revision. | Master key |
| `POST` | `/v1/routing-policies/{policy_id}/eval-scores` | Apply uploaded eval or benchmark scores to a weighted routing policy. | Master key |
| `GET` | `/v1/routing-policies/{policy_id}` | Get a routing policy. | Master key |
| `PATCH` | `/v1/routing-policies/{policy_id}` | Update a routing policy. | Master key |
| `DELETE` | `/v1/routing-policies/{policy_id}` | Delete a routing policy. | Master key |
| `POST` | `/v1/projects` | Create a project and optionally attach a routing policy and budget. | Master key |
| `GET` | `/v1/projects` | List projects. | Master key |
| `GET` | `/v1/projects/{project_id}` | Get a project. | Master key |
| `PATCH` | `/v1/projects/{project_id}` | Update a project's policy, budget, active/blocked state, name, or metadata. | Master key |
| `DELETE` | `/v1/projects/{project_id}` | Delete a project. | Master key |
| `GET` | `/v1/route-traces` | List routing traces. Filters: `project_id`, `user_id`, `policy_id`, `endpoint`, `status`. | Master key |
| `GET` | `/v1/route-traces/summary` | Summarize counts, cost, and latency by model, policy, policy source, endpoint, provider, and strategy. Filters: `project_id`, `user_id`, `policy_id`, `endpoint`, `status`. | Master key |
| `GET` | `/v1/route-traces/{trace_id}` | Get one routing trace by id. | Master key |

### Usage

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `GET` | `/v1/usage` | List usage logs. Filters: `start_date`, `end_date`, `user_id`, `project_id`, `tag_key`, `tag_value`. | Master key |
| `GET` | `/v1/usage/summary` | Summarize usage count, tokens, and cost by project, user, model, provider, endpoint, status, and tag. Filters: `start_date`, `end_date`, `user_id`, `project_id`, `tag_key`, `tag_value`. | Master key |
