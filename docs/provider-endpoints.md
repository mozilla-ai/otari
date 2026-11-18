# Provider endpoints

A provider endpoint lets an app built on Otari register a model server its
user brings (a base URL, an API key, and sometimes extra request fields). The
user's traffic then goes through the gateway with usage recorded, without
touching any budget.

## Turn it on

Provider endpoints are off until an operator turns them on in `config.yml`:

```yaml
provider_endpoints_enabled: true
```

or `OTARI_PROVIDER_ENDPOINTS_ENABLED=true`. Stored keys need `OTARI_SECRET_KEY`,
as organization provider keys do.

## Owners and names

An endpoint belongs to a workspace, and optionally to one user in it:

- **Workspace-wide**: every caller in the workspace reaches it.
- **A user's own**: only that user reaches it, and it shadows a workspace-wide
  endpoint of the same name for them.

Callers reach an endpoint as `<name>:<model>`, for example `my-vllm:qwen3`, on
`/api/v1/chat/completions`, `/api/v1/responses` and `/api/v1/messages`. The API
key decides the workspace and user, so no request can reach someone else's
endpoint.

A name is letters, digits, `.`, `_` and `-`, and may not be a provider's name
or a configured instance's. An instance added later with an endpoint's name
takes over the selector, and the endpoint stops resolving until renamed.

Endpoints are not routing targets: a routing policy that names one does not
resolve it, and neither do embeddings, batches or the other pass-through routes.

## Managing endpoints

The API is `/api/v1/provider-endpoints`, for deployment operators (the master
key). An app typically creates or updates a user's endpoint when the user saves
their model settings:

```bash
curl -X POST http://localhost:8000/api/v1/provider-endpoints \
  -H "Authorization: Bearer $OTARI_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-vllm",
    "provider": "openai",
    "api_base": "https://vllm.example.com/v1",
    "api_key": "sk-...",
    "user_id": "alice",
    "default_params": {"chat_template_kwargs": {"enable_thinking": false}}
  }'
```

- `workspace_id` defaults to the deployment's default workspace; omit
  `user_id` for a workspace-wide endpoint.
- `GET`, `PATCH` and `DELETE` take the endpoint's `id`.
- The key is never returned, only its last four characters.
- A change applies at once on the worker that served it and within 30 seconds
  on the others.
- `provider` is `openai` for an OpenAI-compatible server or `anthropic` for an
  Anthropic-compatible one. Callers can use any of the three routes either
  way, except that `/api/v1/responses` needs an `openai` endpoint.

## Default request fields

`default_params` are added to every request body sent to the endpoint; a field
the caller sets wins. Use them for fields Otari does not model, such as vLLM's
`chat_template_kwargs`. Credential and transport fields (`api_key`, `api_base`,
`client_args` and the like) and request-shaping fields (`model`, `messages`,
`input`, `stream`) are refused.

## Billing

The endpoint's owner pays the upstream, so a request never counts toward a
budget and is never refused for missing pricing. It is recorded under the
endpoint's name (`my-vllm:qwen3`) with `counts_toward_budget` false, so the app
can charge from the token counts.

## Network safety

An endpoint's URL comes from a tenant, so it must reach public addresses only,
whatever `provider_allow_private_hosts` says:

- A host resolving to a private, loopback, link-local, shared or reserved
  address is refused when the endpoint is saved, and checked again on every
  request (Otari connects to the address it checked).
- Redirects are not followed, and the environment's HTTP proxy is not used.
- An endpoint with an API key must use `https`.

A model server on a private address, such as llama.cpp on your laptop or a
compose network, is a hosted provider instead, where
`provider_allow_private_hosts` applies. See
[Connect a llama.cpp server](providers/llamacpp.md) and
[Hosted providers and BYO credentials](hosted-providers.md).

Each gateway worker holds at most 100 concurrent requests open to one endpoint
address. A request that finds none free waits up to 10 seconds, then fails.
