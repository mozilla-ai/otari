---
description: Replace a provider credential without breaking active workloads.
icon: rotate
---

# Rotate a provider key

Rotate a provider key by adding its replacement first, and keep the old one
active until requests succeed with the new one. Provider keys are looked up on
every request, so your applications' API keys need no change.

Rotate when a key has been exposed, or you suspect it has, when your
organization rotates keys on a schedule, or when you move traffic to a new
provider account.

## Add the new key

1. Create a new API key in your provider's dashboard. Do not revoke the old key
   yet.
2. In Otari, open **Organization**, then **Providers**.
3. Click **Add provider key**, select the same provider, enter a distinct **Key
   Name**, and paste the new key.
4. Make the new key the provider's default with its **star icon**.
5. For each workspace that pins the old key, click **Edit** and select **Always
   use this key** for the new key. A workspace keeps using a pinned key after
   the organization's default changes.
6. Send a test request from each affected workspace, and confirm it in the
   workspace's **Overview**, on its **Requests** tab.

## Retire the old key

1. Back in **Organization**, then **Providers**, **Archive** the old key. An
   archived key can then be deleted.
2. Revoke the old key in the provider's dashboard. Removing a key from Otari
   does not revoke it with the provider.

If you suspect the key was exposed, assign the replacement immediately, then
revoke the exposed key with the provider.
