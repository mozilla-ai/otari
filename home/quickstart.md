---
description: Go from sign-up to your first request in a few minutes.
---

# Quickstart

In this quickstart you send your first request through [otari.ai](https://otari.ai):
you create an API key in your default workspace, pick a model, send one request,
and see it in the workspace's usage. It takes about five minutes, and all you
need is an otari.ai account.

To run Otari on your own machine instead, follow the
[Self-hosted quickstart](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/self-hosting).

## 1. Open your workspace

Log in to [otari.ai](https://otari.ai) and open your default workspace from the
top-left corner. This quickstart uses its **API Keys**, **Models** and **Usage**
tabs.

## 2. Create an API key

1. Open the **API Keys** tab and click **Create Key**.
2. Give the key a name. It can use every model your workspace can, unless you
   limit it under **Advanced**.
3. Copy the key, which starts with `otk_`. It is shown only once.

Keep the key in your shell for the next steps:

```bash
export OTARI_API_KEY=otk_...
```

Treat it like a password: keep it in your shell or a secret store, never in
source control.

## 3. Pick a model

Open the **Models** tab, find a model with the filters (modality, provider,
vendor, capabilities), and click **Use this model** to copy its name. This
quickstart uses `deepseek/deepseek-v4-pro-0813`.

## 4. Send a request

otari.ai serves an OpenAI-compatible API at `https://api.otari.ai/api/v1`, or at
`https://eu.api.otari.ai/api/v1` for an account in the European Union:

```bash
curl https://api.otari.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OTARI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek/deepseek-v4-pro-0813",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}]
  }'
```

The answer is an OpenAI-style chat completion, with the model's reply in
`choices[0].message.content`.

## 5. See it in Usage

Open the **Usage** tab: your request is listed with its model, its tokens and
its cost.

You now have a key your applications can call otari.ai with, and every request
they send is recorded against your workspace.

## Next steps

- [Use with Claude Code](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/guides/agents-and-frameworks/use-with-claude-code): point Claude Code at otari.ai.
- [Access control](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/guides): workspaces, members, keys and budgets.
- [Routing policies](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/reference/gateway/routing): fall back to another model when one fails.
- [Inference Guardrails](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/guides/guardrails/guardrails): check requests and responses.
