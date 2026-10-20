# otari.ai

[otari.ai](https://otari.ai) is Otari run by Mozilla.ai as a service. It is the
same gateway and the same dashboard these docs describe, so everything under
Guides and Integrations applies to it. What otari.ai adds is that you do not run
anything:

- **No deployment to operate.** The gateway, its database, the dashboard and the
  [tool services](../self-hosting/tool-services.md) are already running, and the
  [Self-hosting](../self-hosting/deploy.md) section does not apply.
- **Otari-managed models.** Catalog models served on Mozilla.ai's own provider
  keys, so a new organization can send requests before it adds a key of its
  own; see [Where a request's credential comes from](../guides/provider-keys.md#where-a-requests-credential-comes-from).
- **A balance.** Requests on Otari-managed models and otari.ai's hosted tools are
  paid from the organization's balance; see [Billing](billing.md). Requests on
  your own provider keys are billed by that provider.
- **Two regions.** An account lives in the United States or the European Union,
  and its base URL says which: see [Connect to Otari](../get-started/connect.md).

To start, follow the otari.ai tab of the [Quickstart](../get-started/quickstart.md).
