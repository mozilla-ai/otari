---
description: Replace a provider credential without breaking active workloads.
icon: rotate
---

# Rotate a provider key

Rotate a provider key by adding its replacement first. Keep the existing key active until requests succeed with the new credential.

Provider keys are resolved at request time. Your application's API keys do not need changes during rotation.

## When to rotate a provider key

* A provider key has been compromised, or you suspect it has.
* Your organization has a periodic key rotation policy.
* You're migrating traffic to a new provider account.

## Rotate the key

1. Create a new API key in your provider's dashboard. Do not revoke the existing key yet.
2. In otari.ai, select **Organization**, then **Providers**, then **Inference**.
3. Click **Add provider key**. Select the same provider, enter a distinct **Name**, and paste the new API key.
4. Set the new key as the provider's default by clicking **Make default provider** (its star icon).
5. For each workspace that pins the old key, open it on **Workspaces**, select its **Providers** tab, and choose **Always use this key** from the new key's **More** menu.
6. Send a test request from each affected workspace. Confirm it completes successfully, and not on a fallback, in the workspace's **Overview**, on its **Requests** tab.

## Remove the old key

Once requests are flowing through the new key, revoke the old key in the upstream provider dashboard.

1. Return to **Organization** → **Providers** → **Inference** in Otari.
2. **Archive** the old provider key. An archived key can then be deleted.

Update every workspace that pins the old key before revoking it. A workspace can continue using a pinned key even after you change the organization default.

If you suspect compromise, create and assign the replacement immediately. Then revoke the exposed key with the provider.

## Rotate a key for your own endpoint

A key with its own [request prefix](provider-keys.md#add-a-key-for-your-own-endpoint) is rotated in place, since no other key can take its prefix: edit the key on **Inference**, paste the new API key, and save.
