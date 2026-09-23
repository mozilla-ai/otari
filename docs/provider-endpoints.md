# Provider endpoints

Some apps built on Otari let their users bring their own model server: a base
URL, an API key, and sometimes request fields that server expects. A provider
endpoint is how such an app registers one with Otari, so the user's traffic goes
through the gateway like everything else, with usage recorded and without
touching anyone's budget.

Provider endpoints are a standalone-mode feature: a hybrid gateway resolves
models through its control plane, and a hosted one serves no inference. They are
off until an operator turns them on:

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
`/v1/chat/completions`, `/v1/responses` and `/v1/messages`. The API key decides
the workspace and the user, so no request field can reach somebody else's
endpoint. A name is letters, digits, `.`, `_` and `-`, and it may not be a
provider's name or a configured instance's, because that selector already means
something.

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

`workspace_id` defaults to the deployment's default workspace; omit `user_id`
for a workspace-wide endpoint. `GET`, `PATCH` and `DELETE` take the endpoint's
`id`. The key is never returned, only its last four characters. A change takes
effect at once on the worker that served it and within 30 seconds on the others.

`provider` picks the implementation that speaks to the server: `openai` for an
OpenAI-compatible one, `anthropic` for an Anthropic-compatible one. Callers can
use whichever route suits them either way, except that the responses route
needs an `openai` endpoint.

## Default request fields

`default_params` are added to every request body sent to the endpoint, beneath
the caller's own: a field the caller sets wins. They are for fields Otari does
not model, such as vLLM's `chat_template_kwargs`, and are sent as they are.
Credential and transport fields (`api_key`, `api_base`, `client_args` and the
like) and the fields that shape the request itself (`model`, `messages`,
`input`, `stream`) are refused.

## Billing

The endpoint's owner pays the upstream, so a request to one never counts toward
a budget and is never refused for missing pricing. It is still recorded, under
the endpoint's name (`my-vllm:qwen3`), with `counts_toward_budget` false, so the
app can charge for it however it likes from the token counts.

## Network safety

An endpoint's URL comes from a tenant, not the operator, so it must reach public
addresses only, whatever `provider_allow_private_hosts` says. Otari refuses a
URL whose host resolves to a private, loopback, link-local, shared or reserved
address when the endpoint is saved. On every request it resolves the host
again, refuses the request if any answer is not public, and connects to an
address it checked rather than resolving the name a second time. Redirects are
not followed. An endpoint with an API key must use `https`, since the key
travels in every request. Requests to an endpoint do not use the environment's
HTTP proxy.
