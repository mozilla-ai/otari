# API Reference

All endpoints are under `http://localhost:8000` by default.

For full request/response schemas, see the [OpenAPI spec](public/openapi.json) or the interactive docs at `/docs` when Otari is running.

## Endpoint availability

| Endpoint group | Standalone | Connected to otari.ai |
|---|---|---|
| Health (`/health*`) | Yes | Yes |
| Chat completions (`/v1/chat/completions`) | Yes | Yes |
| Messages (`/v1/messages`, `/v1/messages/count_tokens`) | Yes | Yes |
| Responses (`/v1/responses`) | Yes | Yes |
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
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions. Supports streaming and tool use (`otari_code_execution`, `otari_web_search`, MCP). | Standalone: API key or master key. Connected: `Authorization` bearer token from otari.ai. |

See [Use with opencode](use-with-opencode.md) for a full client setup.

### Messages

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/messages` | Anthropic Messages API-compatible endpoint. Supports streaming, tool use, and extended thinking. Routes to any provider in the catalog (non-Anthropic models are translated to/from the Messages format automatically). | Standalone: API key or master key. Connected: `Authorization` bearer token from otari.ai. |
| `POST` | `/v1/messages/count_tokens` | Anthropic-compatible input-token count for a Messages request. Returns `{"input_tokens": N}`. Counts locally (no provider call, no budget debit); the count is an approximation. Used by clients such as Claude Code for context-window management. | Standalone: API key or master key. Connected: `Authorization` bearer token from otari.ai. |

See [Use with Claude Code](use-with-claude-code.md) for a full client setup.

### Responses

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/responses` | OpenAI Responses API-compatible endpoint. Supports streaming. | Standalone: API key or master key. Connected: `Authorization` bearer token from otari.ai. |

## Standalone-only endpoints

### Embeddings

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/embeddings` | Generate embeddings for text input. | API key or master key |

### Models

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `GET` | `/v1/models` | List available models from pricing entries. | API key or master key |
| `GET` | `/v1/models/{model_id}` | Get a specific model. | API key or master key |

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

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/v1/budgets` | Create a budget. | Master key |
| `GET` | `/v1/budgets` | List budgets. | Master key |
| `GET` | `/v1/budgets/{budget_id}` | Get a specific budget. | Master key |
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

### Usage

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `GET` | `/v1/usage` | List usage logs. Filters: `start_date`, `end_date`, `user_id`. | Master key |
