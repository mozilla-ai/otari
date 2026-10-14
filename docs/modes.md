# Runtime modes

Otari has three deployment modes. Standalone serves both the control plane and
data plane. Hosted serves only a multi-tenant control plane. Hybrid serves a data
plane connected to an external control plane such as otari.ai.

## Comparison

| | Standalone | Hosted | Hybrid |
| --- | --- | --- | --- |
| Local database | Yes | Yes | No |
| Local management API | Yes | Yes | No |
| Local inference | Yes | No | Yes |
| Provider credentials | Local config or database | Organization-scoped for tenants; deployment config is operator-only | Resolved per request from the platform |
| Usage | Stored locally | Received and stored by the control plane | Reported to the platform |
| Dashboard | Full local dashboard | Organization-scoped control plane | Health and control-plane link |

`GET /api/v1/bootstrap` publishes the effective mode, sign-in methods, available
management surfaces, and the management or data-plane URL the dashboard needs.

## Mode and who runs it

The mode says what a process serves. It does not say who runs the process.

- A **self-hosted** gateway is one you run on your own infrastructure. It runs in
  standalone mode, or in hybrid mode when it connects to otari.ai.
- **otari.ai's own gateway** is the one mozilla.ai runs to serve otari.ai's
  inference. It runs in hybrid mode.

"Hosted" names the control-plane mode. It is not the opposite of
"self-hosted": a hosted deployment serves no inference, and anyone can run one.

## Standalone

Standalone is the default when neither `OTARI_MODE` nor `OTARI_AI_TOKEN` is
set. It serves the full API, stores keys, budgets, and usage locally, and resolves
providers from configuration or stored credentials.

SQLite is useful for evaluation. Use PostgreSQL for a durable deployment.

Standalone supports local aliases and routing policies, including failover,
weighted and priority routing, conditional selection, and learned routing. Multi-provider
fallback is therefore available without otari.ai when you configure a policy.

## Hosted

Set `OTARI_MODE=hosted` when one process is the control plane for multiple
organizations.

Hosted mode serves the management API and `GET /api/v1/models`, but does not serve
inference, files, batches, or other data-plane operations. Those paths return a
descriptive `404`. Set `data_plane_url` so the error and dashboard snippets
point clients to the correct gateway.

The dashboard hides deployment-wide provider management, leaving the
organization-scoped provider keys that a standalone deployment also has. It also hides the Playground, whose
whole purpose is to dispatch a completion. Deployment-wide APIs still require
operator authority; organization-scoped APIs apply membership and role checks.

A hosted control plane and its hybrid gateways form one system: the gateway
resolves credentials from the control plane and reports usage back so the
control plane can debit the correct tenant.

## Hybrid, connected to otari.ai

Hybrid mode is selected when `OTARI_AI_TOKEN` is present. Set
`OTARI_MODE=hybrid` as well if you want startup to require that mode. Conflicting
configurations fail at startup:

- `hybrid` without a platform token
- `standalone` or `hosted` with a platform token

The token is the gateway credential created in otari.ai, commonly prefixed
`gw_`. It is not the user token sent by clients.

```bash
export OTARI_AI_TOKEN=gw_your_gateway_token
otari serve
```

Hybrid serves health, bootstrap, Chat Completions, Messages, Responses,
embeddings, rerank, image generation, audio transcription and speech,
moderations, direct MCP tool execution (`POST /api/v1/mcp/execute`), and the
guardrail hook check (`POST /api/v1/hooks/check`). Files, batches, direct search
and decision calls (`/decisions`, `/systemone`, `/routing/recommend`) are
standalone only. It
does not initialize the local management database or use local provider
configuration. Clients authenticate with an otari.ai user
token, accepted in the same header forms as standalone mode
(`Authorization: Bearer`, `Otari-Key`, or `x-api-key`).

The gateway asks the platform to resolve an ordered set of provider attempts,
tries retryable fallbacks before a response begins, and reports each outcome.
Workspace MCP and web-search configuration are resolved through the same control
plane.

## Managed models and BYO credentials

Hybrid mode can receive two kinds of provider credential:

- A workspace's own provider key. The upstream provider bills that workspace.
  Any hybrid gateway can use it, including a self-hosted one.
- A mozilla.ai-managed credential. Usage is billed through otari.ai. Only
  otari.ai's own gateway receives it.

Managed model identifiers use the catalog values published by otari.ai. otari.ai
refuses a managed credential to a self-hosted gateway, so platform-owned secrets
never leave mozilla.ai's infrastructure.

## Deployment-paid credentials in standalone

A build that binds `ModelProviderPort` through `OTARI_BOOTSTRAP` can serve a
model on a credential the deployment owns, once an organization's own key, a
stored provider and `config.yml` have all missed. The deployment pays that
upstream bill, so the request also goes through `BillingPort`. Before the
provider call, the organization is granted any credit it is owed and refused
with `402` (`insufficient_funds`) when it has no funds paid in. When the
request settles, its cost, including the cost an abandoned or failed request
still owes, is charged to the organization. No funds are held while the
request runs, so concurrent requests can take an organization below zero. A
request on any other credential never reaches either port, and the core's
billing adapter does nothing, so a deployment with no overlay is unaffected.

Today this covers Chat Completions, Messages, Responses and the Playground for a
plain model or a single-target alias or policy. A routing policy with fallback
candidates and the pass-through endpoints never reach `ModelProviderPort`.

## Internal protocol

The gateway and control plane exchange provider resolution, MCP configuration,
web-search policy, and usage reports. Integrators building a compatible control
plane should use the normative [hybrid-mode protocol](hybrid-mode-protocol.md)
rather than this overview.
