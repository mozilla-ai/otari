# Mistral Vibe

[Mistral Vibe](https://docs.mistral.ai/vibe/code/cli/configuration) is
Mistral's coding agent for the terminal. It can use any OpenAI-compatible API
as a provider, including Otari.

## Quick start

1. Create a key on the dashboard's **API keys** page.
2. Add Otari as a provider, and a model on it, in `~/.vibe/config.toml`.
   `active_model` belongs above the file's first table header, `[...]` or
   `[[...]]`, and replaces any `active_model` already there:

   ```toml
   active_model = "glm-5.3"

   [[providers]]
   name = "otari"
   api_base = "https://api.otari.ai/api/v1"
   api_key_env_var = "OTARI_API_KEY"
   api_style = "openai"
   backend = "generic"

   [[models]]
   name = "z-ai/glm-5.3"
   provider = "otari"
   alias = "glm-5.3"
   ```

3. Export the key and start Vibe:

   ```bash
   export OTARI_API_KEY=<your-otari-key>
   vibe
   ```

`api_base` ends in `/api/v1`; Vibe adds `/chat/completions` itself. EU accounts
use `https://eu.api.otari.ai/api/v1`, and a gateway you run uses its URL plus
`/api/v1`. See [Base URL](api-reference.md#base-url).

A `.vibe/config.toml` in a project you trust overrides the settings it sets,
and the rest still come from `~/.vibe/config.toml`; `VIBE_HOME` moves the home
one. Name the provider something other
than `mistral`: a provider with a built-in one's name replaces it.

## Choosing a model

Each `[[models]]` entry is one model Vibe can use. `name` goes to Otari
unchanged, so write it as Otari spells it, and `alias` is what you pick it by:
in `active_model`, or in Vibe's model picker. Add an entry for each model you
want to switch between:

```toml
[[models]]
name = "moonshotai/kimi-k3.1-pro"
provider = "otari"
alias = "kimi-k3.1-pro"
```

- On otari.ai, a catalog id such as `z-ai/glm-5.3` lets Otari choose the
  provider, and `anthropic:claude-sonnet-4-6` pins one.
- On your own gateway, use `provider:model` for a provider it has, such as
  `openai:gpt-4o`.

See [Model format](models.md#model-format) and
[Listing available models](models.md#listing-available-models).

## See also

- [opencode](opencode.md)
- [Codex](codex.md)
