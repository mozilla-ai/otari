---
icon: key
---

# Provider keys

Provider keys are the upstream credentials your organization uses to call LLM providers like OpenAI, Anthropic, or Mistral. They live at the organization level. Workspaces inherit provider access by default and can pin or disable keys per provider.

## Two ways to get provider access

### Otari-managed keys

Use the Mozilla.ai inference catalog without bringing your own key material. Otari handles the upstream credential, and usage is billed against your [balance](billing.md).

### Your own keys

Bring your existing provider API keys and let Otari use them on your behalf. The upstream provider bills you directly, and Otari adds no per-token routing charge.

Most organizations use a mix. Managed keys are convenient for catalog models. Your own keys are for specific accounts, contracts, or providers not in the catalog.

## How workspaces resolve provider keys

When a workspace makes a request, Otari resolves which provider key to use in this order:

1. If the workspace has a key pinned for the provider, use that.
2. Otherwise, use the organization default key for that provider.
3. If there is no default, or the workspace has turned it off, use another of the organization's keys for that provider that the workspace has not turned off.

This means a workspace inherits everything by default but can opt out of specific keys or override defaults per provider. When the organization has no key for the provider, otari.ai's own keys can serve the request; see [The credential ladder](hosted-providers.md#the-credential-ladder).

## Add a key for your own endpoint

A key can point at a compatible endpoint other than the provider's own, such as Synthetic or a vLLM server: enter its **API base URL** when you add the key. Requests then name the key by its **Request prefix** rather than by the provider, for example `synthetic:hf:Qwen/Qwen3.8-27B`.

The prefix is made from the key's name unless you enter one, and never changes once set, because your applications send it. To cap the key's spend, narrow a spend ceiling to its prefix, not to the provider. A key added without a base URL cannot gain one later: add a new key instead.

## Control workspace access

Each workspace's **Providers** tab, under **Organization** → **Workspaces**, lists the keys and hosted providers it can use. There you can turn a key off, limit which models the workspace may call, or pin a key with **Always use this key**. A row you changed is marked **Customized**, and **Reset** returns it to the organization's settings. For the steps, see [Manage provider keys](manage-provider-keys.md#control-workspace-access).

**Require zero data retention**, at the top of the tab, limits the workspace to providers declared to keep no prompts or completions; see [Zero data retention](hosted-providers.md#zero-data-retention). Other providers are then refused, and the workspace's model lists show only declared ones.

## Provider keys vs API keys

These are two different things, and the naming is unfortunately close:

* **Provider keys** are your upstream credentials, like your OpenAI or Anthropic key, that Otari uses to call providers. They are managed at the organization level.
* **API keys** are the workspace-scoped tokens your application sends to otari.ai. They are managed per workspace.

Your application sends an API key to Otari. Otari uses a provider key to call the upstream provider. The application never sees the provider key.

## How security works

* **Organization scope**. Provider keys are owned by the organization, not copied into each workspace.
* **Workspace controls**. Each workspace decides which organization keys are effective for its workloads.
* **Role gating**. Only admins and owners can add or remove provider key material, or change which keys a workspace can access.
* **Runtime resolution**. Otari resolves the right provider key at request time. The workspace's API key never carries the upstream secret.

Removing a provider key from Otari does not revoke it with the upstream provider. After deleting a key from Otari, also revoke it in the provider's dashboard.

Ready to create a key? See [Manage provider keys](manage-provider-keys.md).
