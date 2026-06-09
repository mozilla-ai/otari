# Supported Models

The gateway routes requests to LLM providers through [any-llm-sdk](https://pypi.org/project/any-llm-sdk/). This page covers the model format, supported providers, and capabilities.

## Model format

Models are specified as `provider:model_name`:

```
openai:gpt-4o
anthropic:claude-sonnet-4-6
mistral:mistral-large-latest
vertexai:gemini-2.0-flash
```

The `provider` prefix tells the gateway which backend to route to. The `model_name` is passed directly to that provider's API.

## Supported providers

The gateway depends on `any-llm-sdk[all]`. Provider support can change as the SDK evolves.

Use this list as a quick reference for common providers supported by the current gateway build.

> **Supported doesn't mean callable by default.** To call a model from this list in [standalone mode](modes.md), it needs a pricing entry. By default (`require_pricing: true`), the gateway rejects any billable request for a model it can't price, returning HTTP 402. This is what keeps unpriced models from bypassing your budget cap.
>
> Two ways to make a model callable:
> - Add a pricing entry for it (see [Configuration](configuration.md#pricing))
> - Or set `require_pricing: false` to allow unpriced models

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
| HuggingFace | `huggingface` | `huggingface:meta-llama/Llama-3-70b` | |
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

The example providers are illustrative, not exhaustive. Support often varies by the specific model within a provider, not just the provider itself, so treat this as a starting point and check the model you intend to use.
 
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

## Listing available models

Query the gateway to see which models are available:

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer <your-api-key>"
```
