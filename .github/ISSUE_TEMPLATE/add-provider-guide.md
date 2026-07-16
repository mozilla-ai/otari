---
name: "Docs: add a provider guide"
about: "Write a setup guide for one LLM provider, following the OpenAI exemplar"
title: "docs: add provider guide for <Provider>"
labels: ["good first issue", "documentation"]
---

## What

Add a short setup guide for **&lt;Provider&gt;** at `docs/providers/<provider-key>.md`, matching the existing exemplar at `docs/providers/openai.md`.

The provider is already in the reference table at `docs/models.md`. This issue adds the deeper "how to set it up, and the gotchas" page for it.

## Why

Each provider has its own auth, base URL, and model-naming quirks. A per-provider page saves the next user from rediscovering them, and it's a small, well-scoped way for a new contributor to help.

## Steps

1. Copy `docs/providers/openai.md` to `docs/providers/<provider-key>.md`. Use the config key from `docs/models.md` (for example `anthropic`, `mistral`, `groq`).
2. Fill in each section for this provider:
   - [ ] **Prerequisites**: where to get an API key (link the provider's console)
   - [ ] **Configure**: the `providers:` block; note if this provider needs `api_base` or extra fields (check `config.example.yml` and `docs/configuration.md`)
   - [ ] **Choose a model**: 2 to 3 real `provider:model` examples
   - [ ] **Verify**: point readers to the Quickstart for starting Otari and creating a client key, then keep a provider-specific request example and swap the model name
   - [ ] **Troubleshooting**: at least one provider-specific gotcha
   - [ ] **Pricing**: optional, only if you have real numbers
3. Test it end to end against a running Otari.
4. Link the new page from the docs navigation (`docs/index.md`) and any provider index (`docs/providers/README.md` or `docs/models.md`) if one exists.

## Acceptance criteria

- [ ] New file at `docs/providers/<provider-key>.md`
- [ ] Same section order as `docs/providers/openai.md`
- [ ] Config snippet matches the schema in `config.example.yml`
- [ ] At least one provider-specific troubleshooting row
- [ ] Linked from `docs/index.md` and any provider index that exists

## Good picks to start

Anthropic, Mistral, Groq, or OpenRouter are good starter picks. Leave Vertex AI and Bedrock for later; they need service accounts or AWS credentials. Ollama also works well, but it is a local-server setup rather than a hosted `api_key` flow.

## Resources

- Exemplar: `docs/providers/openai.md`
- Config schema: `config.example.yml`
- Provider list and model format: `docs/models.md`
