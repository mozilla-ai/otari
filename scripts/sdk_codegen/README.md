# SDK control-plane codegen

Generates typed **control-plane** clients (API keys, users, budgets, pricing,
usage) for each language SDK from the gateway's OpenAPI spec, using
[OpenAPI Generator](https://openapi-generator.tech/).

## What is and isn't generated

`generate.py` filters the spec to operations tagged `keys`, `users`, `budgets`,
`pricing`, `usage` (plus the transitive closure of the schemas they reference)
and generates only those. These endpoints have fully typed responses in the spec.

Deliberately **not** generated:

- **Inference / proxy** (chat, responses, embeddings, messages, moderations,
  rerank, audio, images, models, health): the published SDKs wrap the official
  OpenAI SDK for those endpoints; the spec leaves them loosely typed; and OpenAPI
  Generator does not emit streaming (SSE) consumption code.
- **Batches**: the batch routes use `response_model=None`, so their responses are
  untyped (`{}`) in the spec, and generation would produce untyped `object`
  results. Every SDK already implements batches with hand-written typed models, so
  generating them would both regress typing and collide with existing code.

So that surface stays hand-written.

## Local usage

Requires a JRE and the OpenAPI Generator CLI:

```bash
npm install -g @openapitools/openapi-generator-cli
uv run python scripts/sdk_codegen/generate.py --language all --out-dir dist
# one language, plus the filtered spec for inspection:
uv run python scripts/sdk_codegen/generate.py --language python \
  --write-filtered-spec dist/control-plane.openapi.json
```

Set `OPENAPI_GENERATOR_CLI` if the CLI is not named `openapi-generator-cli` on PATH.

## Automation

`.github/workflows/gateway-sdk-codegen.yml` runs this on release, on spec change
to `main`, or manually, and opens a PR against each SDK repo with the regenerated
client dropped into:

| Language   | SDK repo                      | Target path           |
|------------|-------------------------------|-----------------------|
| python     | `mozilla-ai/otari-sdk-python` | `src/otari/_generated`|
| typescript | `mozilla-ai/otari-sdk-ts`     | `src/generated`       |
| go         | `mozilla-ai/otari-sdk-go`     | `otari/generated`     |
| rust       | `mozilla-ai/otari-sdk-rust`   | `src/generated`       |

The matrix in the workflow mirrors `TARGETS` in `generate.py`; keep them in sync.

**Required secret:** `SDK_CODEGEN_TOKEN`, a fine-grained PAT or GitHub App token
with `Contents:write` and `Pull-requests:write` on the four SDK repos. The default
`GITHUB_TOKEN` cannot push to other repositories.

## Follow-up (per SDK repo)

This pipeline lands the generated client in the target path. Wiring it into each
SDK's public surface (re-exporting the control-plane methods from the hand-written
client) is a per-repo change, tracked separately.
