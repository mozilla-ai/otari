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

`GET /v1/bootstrap` publishes the effective mode, sign-in methods, available
management surfaces, and the management or data-plane URL the dashboard needs.

## Standalone

Standalone is the default when neither `OTARI_MODE` nor `OTARI_AI_TOKEN` is
set. It serves the full API, stores keys, budgets, and usage locally, and resolves
providers from configuration or stored credentials.

SQLite is useful for evaluation. Use PostgreSQL for a durable deployment.

Standalone supports local aliases and routing policies, including failover,
weighted routing, conditional selection, and learned routing. Multi-provider
fallback is therefore available without otari.ai when you configure a policy.

## Hosted

Set `OTARI_MODE=hosted` when one process is the control plane for multiple
organizations.

Hosted mode serves the management API and `GET /v1/models`, but does not serve
inference, files, batches, or other data-plane operations. Those paths return a
descriptive `404`. Set `data_plane_url` so the error and dashboard snippets
point clients to the correct gateway.

The dashboard hides deployment-wide provider management and exposes
organization-scoped provider keys instead. Deployment-wide APIs still require
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

Hybrid serves health, bootstrap, Chat Completions, Messages, and Responses. It
does not initialize the local management database or use local provider
configuration. Clients authenticate with an otari.ai user token in
`Authorization: Bearer <token>`.

The gateway asks the platform to resolve an ordered set of provider attempts,
tries retryable fallbacks before a response begins, and reports each outcome.
Workspace MCP and web-search configuration are resolved through the same control
plane.

## Managed models and BYO credentials

Hybrid mode can receive two kinds of provider credential:

- A workspace's own provider key. The upstream provider bills that workspace,
  and the key may be used through a self-hosted gateway.
- A mozilla.ai-managed credential. Usage is billed through otari.ai and the
  credential is returned only to the gateway operated by mozilla.ai.

Managed model identifiers use the catalog values published by otari.ai. A
self-hosted gateway that requests a managed credential is refused; this prevents
platform-owned secrets from leaving managed infrastructure.

## Internal protocol

The gateway and control plane exchange provider resolution, MCP configuration,
web-search policy, and usage reports. Integrators building a compatible control
plane should use the normative [hybrid-mode protocol](hybrid-mode-protocol.md)
rather than this overview.
