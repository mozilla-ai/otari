# Configure providers

How a self-hosted deployment reaches providers on credentials it owns: providers
in `config.yml`, named instances, local model servers and HuggingFace routing.
Organizations can also bring their own keys in the dashboard
([Provider keys](../guides/provider-keys.md)), and model selectors are covered in
[Models](../guides/models.md). Step-by-step guides exist for
[OpenAI](providers/openai.md) and [Gemini](providers/gemini.md).

## Configuring a provider

Declare a provider under `providers` in `config.yml`, or add it on the
Hosted providers page. Either way it serves requests and its models are
discovered. Supply the key from the environment with a `${VAR}` reference:

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
```

A key reaches a provider only from where a declaration names it: an entry's own
`api_key` (a `${VAR}` reference included), a hosted provider's key, or
an organization's own provider key, which declares the provider for that
organization's requests. A provider's native variable (such as `OPENAI_API_KEY`)
is never read for it otherwise, so an entry with no `api_key`, or one carrying
only `client_args`, has no credential. A request it cannot serve is refused with
a `424` and the `provider_not_configured` code. Otherwise a variable present on
the server would serve every caller that has no credential of its own.

Keyless local backends (`ollama`, `llamacpp`, `llamafile`, `lmstudio`, `vllm`)
and providers authenticating from ambient cloud credentials (`bedrock`,
`sagemaker`, `vertexai`) need a declaration too, but no key: a bare entry
declares one. Undeclared, they are refused like any other provider, because the
local server and the cloud role belong to the deployment.

```yaml
providers:
  ollama:
  bedrock:
```

The one exception is a build whose extension serves the request on a credential
the deployment owns (`ModelProviderPort`, asked only after every other source
missed); a plain deployment has none. The gateway logs a warning at startup
naming each `<PROVIDER>_API_KEY` variable that is set and that no entry's
`api_key` reads, and ignores an entry for a keyed provider that names no key.

Provider support is endpoint-specific. A provider that supports chat may not
support Responses, images, audio, rerank, or batches. Unsupported combinations
are rejected before dispatch when Otari can identify them.

## Named provider instances

Use a named instance when several backends share one provider implementation or
when a self-hosted server speaks a compatible protocol:

```yaml
providers:
  home_lab:
    provider_type: openai
    api_base: "https://models.example.com/v1"
    api_key: ${HOME_LAB_TOKEN}
    models:
      - qwen3-32b
```

Call this model as `home_lab:qwen3-32b`. Pricing and usage also use the instance
name. `provider_type: openai-compatible` and `openai_compatible` are accepted
aliases for the OpenAI implementation.

The optional `models` list supplies discovery for a backend that has no
model-listing endpoint. It does not restrict direct dispatch.

Named instances are local configuration and do not apply in hybrid mode, where
the control plane resolves each attempt.

A workspace or a single user can also bring its own server, reached the same
way (`my-vllm:qwen3`) but visible only to its owner. See
[Provider endpoints](provider-endpoints.md).

## Local providers

Ollama, llama.cpp, and llamafile can run without an API key. Add a provider entry
when you want Otari to discover their models:

```yaml
providers:
  ollama:
    api_base: "http://gpu-box:11434"
```

Without the entry, a direct selector can still work if the backend is reachable
at the provider's default address. Otari does not probe local servers merely
because their provider exists in any-llm.

## HuggingFace provider routing

A HuggingFace model may be served by several inference providers. Pin the backend
when deterministic routing and pricing matter:

```text
huggingface:zai-org/GLM-4.6:together
huggingface:zai-org/GLM-4.6:novita
```

Otari splits only the first colon, so the backend suffix reaches HuggingFace
unchanged. Auto, cheapest, or fastest routing cannot be assigned one reliable
backend price; configure pinned selectors when enforcing budgets.
