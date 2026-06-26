# Models

Otari routes requests to LLM providers through [any-llm-sdk](https://pypi.org/project/any-llm-sdk/). This page covers the model format, supported providers, and capabilities.

## Model format

Models are specified as `provider:model_name`:

```
openai:gpt-4o
anthropic:claude-sonnet-4-6
mistral:mistral-large-latest
vertexai:gemini-2.0-flash
```

The `provider` prefix tells Otari which backend to route to. The `model_name` is passed directly to that provider's API.

### Pinning a HuggingFace inference backend

HuggingFace Inference Providers is a router: the same model id (for example `zai-org/GLM-4.6`) can be served by several backends (Together, Novita, and others), and in the default "auto" mode the backend, and therefore the price, is chosen at request time. To route (and price) deterministically, pin a backend with a `:<backend>` suffix on the model id, which the HuggingFace router honors server side:

```text
huggingface:zai-org/GLM-4.6:together
huggingface:zai-org/GLM-4.6:novita
```

The pinned-selector grammar is `huggingface:<model>:<backend>`. Otari splits the provider off the first `:`, so everything after it (`<model>:<backend>`) is forwarded as the model id and the `:<backend>` suffix reaches the router unchanged. The router also accepts policy suffixes such as `:cheapest`, `:fastest`, `:preferred`, and `:auto`.

This grammar is the contract consumers build against. The otari.ai platform's pricing UI, for instance, offers each priceable backend as a pinned `huggingface:<model>:<backend>` selector, because a pinned selector resolves to a single backend, which is what makes a HuggingFace model priceable (auto mode cannot be priced from the model id alone).

## Supported providers

Otari depends on `any-llm-sdk[all]`. Provider support can change as the SDK evolves.

Use this list as a quick reference for common providers supported by the current Otari build.

| Provider | Config key | Example model | Notes |
|----------|-----------|---------------|-------|
| Anthropic | `anthropic` | `anthropic:claude-sonnet-4-6` | |
| AWS Bedrock | `bedrock` | `bedrock:anthropic.claude-v2` | AWS credentials required |
| Azure OpenAI | `azureopenai` | `azureopenai:gpt-4o` | Requires `api_base` |
| Azure Anthropic | `azureanthropic` | `azureanthropic:claude-sonnet-4-6` | Requires `api_base` |
| Cerebras | `cerebras` | `cerebras:llama3.1-8b` | |
| Cohere | `cohere` | `cohere:command-r-plus` | Also supports rerank |
| DashScope | `dashscope` | `dashscope:qwen-turbo` | Alibaba Cloud |
| Databricks | `databricks` | `databricks:dbrx-instruct` | Requires `api_base` |
| DeepInfra | `deepinfra` | `deepinfra:meta-llama/Llama-3-70b` | |
| DeepSeek | `deepseek` | `deepseek:deepseek-chat` | |
| Fireworks | `fireworks` | `fireworks:llama-v3-70b` | |
| Gemini | `gemini` | `gemini:gemini-2.0-flash` | |
| Groq | `groq` | `groq:llama3-70b-8192` | |
| HuggingFace | `huggingface` | `huggingface:meta-llama/Llama-3-70b` | Pin a backend with `:<backend>` (see [Pinning a HuggingFace inference backend](#pinning-a-huggingface-inference-backend)) |
| Inception | `inception` | `inception:mercury-coder-small` | |
| Llama.cpp | `llamacpp` | `llamacpp:default` | Local server |
| Llamafile | `llamafile` | `llamafile:default` | Local server |
| LM Studio | `lmstudio` | `lmstudio:local-model` | Local server |
| MiniMax | `minimax` | `minimax:abab5.5-chat` | |
| Mistral | `mistral` | `mistral:mistral-large-latest` | |
| Moonshot | `moonshot` | `moonshot:moonshot-v1-8k` | |
| Nebius | `nebius` | `nebius:llama-3-70b` | |
| Ollama | `ollama` | `ollama:llama3` | Local server |
| OpenAI | `openai` | `openai:gpt-4o` | |
| OpenRouter | `openrouter` | `openrouter:openai/gpt-4o` | |
| Perplexity | `perplexity` | `perplexity:llama-3-sonar-large` | |
| SageMaker | `sagemaker` | `sagemaker:my-endpoint` | AWS credentials required |
| SambaNova | `sambanova` | `sambanova:llama3-70b` | |
| Together | `together` | `together:meta-llama/Llama-3-70b` | |
| Vertex AI | `vertexai` | `vertexai:gemini-2.0-flash` | Requires service account |
| Vertex AI Anthropic | `vertexaianthropic` | `vertexaianthropic:claude-sonnet-4-6` | Requires service account |
| vLLM | `vllm` | `vllm:my-model` | Self-hosted |
| Voyage | `voyage` | `voyage:voyage-large-2` | Embeddings only |
| WatsonX | `watsonx` | `watsonx:ibm/granite-13b` | |
| xAI | `xai` | `xai:grok-2` | |

## Capabilities

Not all providers support all endpoints. Here's what each endpoint type requires:

| Endpoint | Capability | Example providers |
|----------|-----------|-------------------|
| `/v1/chat/completions` | Chat completion | Most providers |
| `/v1/messages` | Anthropic Messages API | Anthropic, Vertex AI Anthropic |
| `/v1/responses` | OpenAI Responses API | OpenAI |
| `/v1/embeddings` | Text embeddings | OpenAI, Cohere, Voyage, Vertex AI |
| `/v1/moderations` | Content moderation | OpenAI |
| `/v1/rerank` | Document reranking | Cohere |
| `/v1/images/generations` | Image generation | OpenAI, Vertex AI |
| `/v1/audio/transcriptions` | Audio transcription | OpenAI |
| `/v1/audio/speech` | Text-to-speech | OpenAI |
| `/v1/batches` | Batch processing | OpenAI, Anthropic |

In deployments connected to otari.ai, the final model/provider choices are resolved by otari.ai routing policy, not by local `providers` configuration.

## Configuring a provider

In `config.yml`:

```yaml
providers:
  openai:
    api_key: "sk-..."
```

Or via environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

Both approaches work. Config file values take precedence over environment variables.

For the full configuration reference, see [Configuration](configuration.md).

## Multiple named instances of one implementation

The `providers` map is keyed by an **instance name**. Normally that name is the
any-llm implementation itself (`openai`, `anthropic`, ...), so a key like
`openai` both names the instance and selects the implementation. To run two
backends that share an implementation, for example real OpenAI alongside a
self-hosted OpenAI-compatible server (vLLM, llama.cpp, LM Studio), give each a
distinct instance name and set `provider_type` to the underlying
implementation:

```yaml
providers:
  openai:                       # key is a real provider, no provider_type needed
    api_key: ${OPENAI_API_KEY}

  home_lab:                     # custom instance name
    provider_type: openai       # underlying any-llm implementation
    api_base: "https://nathans-mac-studio.example.ts.net/v1"
    api_key: ${HOME_LAB_TOKEN}
```

Route to an instance with `instance_name:model`. A request for
`home_lab:deepseek-v4-flash` resolves instance `home_lab` to
`provider_type: openai` and dispatches to any-llm with `provider=openai`,
`model=deepseek-v4-flash`, and the instance's `api_base` / `api_key`. any-llm
never sees the name `home_lab`; it is an Otari-level routing key.
`openai:gpt-4o` continues to hit real OpenAI. Pricing and usage are keyed on the
instance name (`home_lab:deepseek-v4-flash`), so configure pricing under that
key (or run with `require_pricing: false` for an unpriced self-hosted backend).

`provider_type: openai-compatible` and `provider_type: openai_compatible` are
both accepted as aliases for `openai`.

Existing configs are unaffected: a key with no `provider_type` is its own
implementation, exactly as before.

Named instances are a standalone-mode feature. In hybrid mode the local
`providers` map is empty (per-request credentials come from otari.ai), so there
are no instances to resolve and the platform's routing policy decides the
provider.

### Declaring models for backends without `/v1/models`

`/v1/models` lists an instance's models by calling the backend's model-listing
endpoint. When a backend does not expose `/v1/models`, declare the served model
ids so they still appear in the listing:

```yaml
providers:
  edge_box:
    provider_type: openai
    api_base: "https://edge.example.ts.net/v1"
    api_key: ${EDGE_TOKEN}
    models:
      - llama-3.3-70b
      - qwen3-32b
```

The declared `models` are listed as `edge_box:<model>`. Direct requests work
with or without this list; it only affects discovery.

## Listing available models

Query Otari to see which models are available:

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer <your-api-key>"
```
