# DeepSeek Harness

[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`) is
DeepSeek's open-source agent. It can use Otari for its models and for its
`web_search` tool, set up in one file.

DeepSeek Harness is in developer preview, so its settings may change between
releases.

## Quick start

1. Create a key on the dashboard's **API keys** page, and turn on
   [web search](web-search.md) for its workspace.
2. Create `~/.dsh` with `mkdir -p ~/.dsh`, and save this in it as
   `cordis.patch.yml`:

   ```yaml
   - id: llm-pi-ai
     config:
       providers:
         otari:
           apiKeyEnv: OTARI_API_KEY
           api: openai-completions
           baseURL: https://api.otari.ai/api/v1
           models:
             - id: deepseek/deepseek-v4.1-flash
   - id: agent-default-model
     config:
       provider: otari
       model: deepseek/deepseek-v4.1-flash
   - id: web
     config:
       searchProvider: deepseek-search-api
       fetchProvider: http
   - id: web-deepseek-api
     config:
       searchBaseURL: https://api.otari.ai/api/v1
       searchApiKeyEnv: OTARI_API_KEY
   ```

   Replace `deepseek/deepseek-v4.1-flash`, in both places, with a model your
   workspace serves; see [Choosing a model](#choosing-a-model).

3. Export the key and start dsh:

   ```bash
   export OTARI_API_KEY=<your-otari-key>
   npx @deepseek-ai/dsh web
   ```

The file applies to every dsh profile, so `dsh tui` and `dsh headless` use
Otari too. `DSH_HOME` moves `~/.dsh`.

If you already have this file, add the entries to it instead of replacing it.
Each entry replaces that entry's whole config in every profile, so a provider
you added on dsh's **Models** page is dropped unless you list it under
`providers` beside `otari`.

Both URLs end in `/api/v1`. EU accounts use `https://eu.api.otari.ai/api/v1`,
and a gateway you run uses its URL plus `/api/v1`. See
[Base URL](https://docs.otari.ai/reference#base-url).

## Choosing a model

dsh only uses the models listed under `models`, and each `id` goes to Otari
unchanged. Add an item for each model you want in the model picker:

```yaml
          models:
            - id: deepseek/deepseek-v4.1-flash
            - id: anthropic:claude-sonnet-4-6
```

- A catalog id such as `deepseek/deepseek-v4.1-flash` lets Otari choose the
  provider, and `provider:model` pins one.
- A [routing policy](https://docs.otari.ai/reference/gateway/routing) name works as an `id` too.

`agent-default-model` is the model new sessions start on, and must be one of
the listed models. See [Model format](https://docs.otari.ai/reference/gateway/models#model-format) and
[Listing available models](https://docs.otari.ai/reference/gateway/models#listing-available-models).

The **Models** page in dsh cannot change what this file sets, so add models
here, then restart dsh.

## Web search

The `web` and `web-deepseek-api` entries send each `web_search` call to
[`POST /api/v1/search`](https://docs.otari.ai/reference/gateway/tools) with your Otari key. Without them, dsh
offers no `web_search` on an Otari model unless you also have a DeepSeek API
key.

- The key needs the `web:search` [scope](api-keys.md#scopes), which new keys
  have.
- The workspace's **Max results** applies, but its **Allowed domains** and
  **Blocked domains** do not: they cover search inside a model request, not a
  direct search. See
  [Per-workspace web-access policy](https://docs.otari.ai/reference/gateway/tools#per-workspace-web-access-policy).
- `GET /api/v1/tools` shows whether the key can search: web search's
  `direct_available` is `true`.

`web_fetch` runs on your machine, through dsh's own HTTP fetcher, which can
reach local and private-network addresses.

## See also

- [Hermes Agent](hermes.md)
- [Mistral Vibe](vibe.md)
