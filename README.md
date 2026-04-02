# gateway

Gateway is a standalone service that proxies LLM requests with API key management,
budget enforcement, usage tracking, and OpenAI-compatible endpoints.

## Install

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements-dev.txt -e .
```

## Run locally

```bash
cp config.example.yml config.yml
uv run gateway serve --config config.yml
```

Or use the dev target:

```bash
make dev
```

## Run tests

```bash
make test
```

## Docker

```bash
docker compose up --build
```
