# Connect a llama.cpp server

Serve a model you run yourself with llama.cpp's `llama-server` through Otari.
See [Models](https://mozilla-ai.gitbook.io/otari-preview/wBxOPVaXYufbmAHVFj2k/reference/gateway/models) for Otari's model-name format.

## Before you start

- A running Otari with `OTARI_SECRET_KEY` set, and its master key. The
  [Self-hosted quickstart](../self-hosted-quickstart.md) sets up both.
- `llama-server` and a GGUF model file.

## Start the server with an alias

Start `llama-server` with `--alias`, so the model id it serves is a stable name
rather than the path of the GGUF file:

```bash
llama-server -m ./models/qwen3-8b-q4_k_m.gguf --alias qwen3-8b --host 0.0.0.0 --port 8080
```

## Add the provider

On **Hosted providers** in the dashboard's deployment view, add a provider of
type **llama.cpp** (`llamacpp`):

1. Leave the API key blank. A llama.cpp server needs none, so use this provider
   rather than `openai` with a placeholder key.
2. Set the API base URL to where Otari reaches the server:
   `http://host.docker.internal:8080/v1` from the Compose container on macOS or
   Windows, or the host's address on Linux.
3. Save. Otari lists the model the server serves, under its alias.

Or add it through the API with the master key:

```bash
curl -X POST http://localhost:8000/api/v1/hosted-providers \
  -H "Authorization: Bearer $OTARI_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"provider": "llamacpp", "api_base": "http://host.docker.internal:8080/v1"}'
```

## Price the model and switch it on

Bundled pricing does not cover a local model, so it arrives switched off. On
the provider's models, set a price on its row (0 is accepted) and switch it on.
The id you price is the alias, and it is also the model you request.

## Verify

Send a request with your Otari API key (`tk-...`), naming the model as
`llamacpp:<alias>`:

```bash
curl http://localhost:8000/api/v1/chat/completions \
  -H "Authorization: Bearer tk-..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llamacpp:qwen3-8b",
    "messages": [{"role": "user", "content": "Say hello."}]
  }'
```

The answer is an OpenAI-style chat completion with the model's reply in
`choices[0].message.content`.

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| HTTP 424 (code `provider_not_configured`) | The model is not switched on under the provider, or the name after `llamacpp:` is not the alias. |
| A price set under the alias never matches | The server was started without `--alias`, so it reports the GGUF file path as the model id. Restart it with `--alias`. |
| HTTP 502 `The gateway could not connect to the provider` | Otari cannot reach the API base URL. From a container, `localhost` is the container itself; use `host.docker.internal` or the host's address. |
