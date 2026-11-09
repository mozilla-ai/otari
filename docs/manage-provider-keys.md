---
description: Add provider keys at the organization level, then control which workspaces can use them.
icon: key
---

# Manage provider keys

For background, see [Provider keys](provider-keys.md).

## Add a provider key

There are two ways to get provider access:

### Otari-managed keys

These are Otari's pre-provisioned keys you can use without bringing your own key material. Otari handles the upstream credential, and requests on them are paid from your organization's [balance](billing.md). There is nothing to add: create an API key, as in the [Quickstart](quickstart.md), and your requests can use them.

### Your own keys

Bring your existing provider API keys and let Otari use them on your behalf. The upstream provider bills you directly, and Otari adds no per-token routing charge. When your organization has a key of its own for a provider, Otari uses it first. To add your own key:

1. From the main dashboard, click **Organization**.
2. Select **Providers**, then **Inference**.
3. Click **Add provider key** in the top-right corner.
4. Select the **Provider** from the dropdown.
5. Enter a **Name** to easily distinguish multiple keys for the same provider, such as `production`, `staging`, or `backup`.
6. Paste your **API key**.
7. For a compatible endpoint of your own, such as Synthetic or a vLLM server, enter its **API base URL**. Requests then reach this key's models by its **Request prefix**; see [Add a key for your own endpoint](provider-keys.md#add-a-key-for-your-own-endpoint).
8. Click **Add provider key** to save it.

![The New provider key dialog, with OpenAI selected and a key named production.](assets/screenshots/manage-provider-keys-add.png)

## Set a default key per provider

If you have multiple keys for the same provider, mark one as the default so workspaces inherit it automatically.

1. On **Inference**, find the keys for the provider.
2. Click **Make default provider** (the star icon) on the key you want workspaces to inherit.

![Two OpenAI keys on the Inference page, production marked as the default.](assets/screenshots/manage-provider-keys-list.png)

Workspaces inherit the organization default per provider unless you define a different key.

## Control workspace access

Once a provider key has been added at the organization level, open **Organization** → **Workspaces**, open the relevant workspace, and select its **Providers** tab to manage its provider access. Click a provider to show its keys. For each key, you can:

* Leave it as it is to follow the organization.
* Turn its switch off to prevent the workspace from using that provider key.
* Select **Always use this key** from the key's **More** menu to ensure the workspace uses a specific provider key. Choose Stop always using this key in the same menu to follow the organization again.
* Use its Models control to restrict access to specific models from that provider.

A row you changed is marked Customized, and Reset in its menu returns it to the organization's settings.

![A workspace's Providers tab, with the production and backup OpenAI keys each switched on.](assets/screenshots/manage-provider-keys-workspace.png)

To replace a key without breaking anything, see [Rotate a provider key](rotate-a-provider-key.md).
