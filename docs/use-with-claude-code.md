# Use with Claude Code

[Claude Code](https://code.claude.com) speaks the Anthropic Messages API and lets
you redirect it at any compatible endpoint with a couple of environment
variables. Otari exposes that surface (`POST /v1/messages` and
`POST /v1/messages/count_tokens`) in both standalone and hybrid modes, so you
can route Claude Code through Otari to get budgets, usage tracking, and traces
without changing how you use the CLI.

## Quick start

```bash
export ANTHROPIC_BASE_URL="https://api.otari.ai"   # your Otari base URL, no /v1
export ANTHROPIC_AUTH_TOKEN="tk_your_otari_token"  # sent as Authorization: Bearer
export ANTHROPIC_MODEL="anthropic:claude-sonnet-4-6"
claude
```

Claude Code appends `/v1/messages` and `/v1/messages/count_tokens` to
`ANTHROPIC_BASE_URL` itself, so the base URL must be the Otari root (for local
development, `http://localhost:8000`).

### Standalone mode: allow Claude Code's user_id

Claude Code attaches its own `metadata.user_id` to every request. In standalone
mode Otari binds spend to the API key's own user and, by default, rejects a
request that names a different user (`403 permission_error`). Set
`reject_user_mismatch: false` in your config so Claude Code's `user_id` is
ignored and spend is bound to the key. (Hybrid mode authenticates via
the user token and does not compare `metadata.user_id`, so this setting does not
apply there.)

Use `ANTHROPIC_AUTH_TOKEN` (not `ANTHROPIC_API_KEY`): it is sent as
`Authorization: Bearer <token>`, which is the scheme Otari accepts for
both standalone API keys and connected user tokens. `ANTHROPIC_API_KEY` sends an
`x-api-key` header instead, which Otari does not read.

### settings.json

The same configuration works in `~/.claude/settings.json` (or a project-level
`.claude/settings.json`):

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.otari.ai",
    "ANTHROPIC_AUTH_TOKEN": "tk_your_otari_token",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "anthropic:claude-opus-4-8",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "anthropic:claude-sonnet-4-6",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "anthropic:claude-haiku-4-5"
  }
}
```

## You are not limited to Claude models

Claude Code speaks the Anthropic *wire format*, but the `model` field is just a
string Otari forwards to [any-llm](https://github.com/mozilla-ai/any-llm).
any-llm translates the Messages format to and from each provider's native API, so
**any model in the catalog works** — OpenAI, Mistral, Mozilla.ai inference
models, Moonshot, and so on — not just Anthropic's Opus/Sonnet/Haiku.

Set the model strings to whatever your deployment expects:

- **Connected to otari.ai:** use `mzai:<catalog-id>` for a
  managed open-weight model (e.g. `mzai:moonshotai/Kimi-K2.6`), or
  `provider/model` for one of your own provider keys (e.g. `openai/gpt-4o`,
  `anthropic/claude-sonnet-4-6`). An `mzai:` prefix selects the managed catalog,
  so adding it to a proprietary model routes it there instead of your key and it
  will not resolve.
- **Standalone:** use `provider:model`, e.g. `openai:gpt-4o`,
  `mistral:mistral-large-latest`, or `anthropic:claude-sonnet-4-6`.

Claude Code uses two tiers you can point independently: a primary model for the
agent loop (`ANTHROPIC_MODEL`, or the `ANTHROPIC_DEFAULT_*_MODEL` tier the
`opus`/`sonnet`/`haiku` aliases resolve to) and a small/fast model for background
work like title generation (`ANTHROPIC_DEFAULT_HAIKU_MODEL`). Set both so neither
tier falls back to a model your Otari instance does not serve. For example, to run the
CLI entirely on managed open-weight models (a capable main model and a smaller
background model):

```bash
export ANTHROPIC_BASE_URL="https://api.otari.ai"
export ANTHROPIC_AUTH_TOKEN="tk_your_otari_token"
export ANTHROPIC_DEFAULT_OPUS_MODEL="mzai:moonshotai/Kimi-K2.6"
export ANTHROPIC_DEFAULT_SONNET_MODEL="mzai:moonshotai/Kimi-K2.6"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="mzai:Qwen/Qwen3-32B"
claude
```

### Caveats with non-Claude models

Routing works at the protocol level, but quality does not transfer
automatically. Claude Code's prompts and tool-use loop are tuned for Claude
models; other models run through the same Otari path but may produce weaker
tool calling and agentic behavior. Anthropic-only features (extended thinking,
prompt caching via `cache_control`) have no equivalent on most providers and are
dropped in translation. Treat non-Claude models as usable, not equivalent.

## What Otari provides

- **`POST /v1/messages`** — the Anthropic Messages endpoint Claude Code drives,
  with streaming and tool use.
- **`POST /v1/messages/count_tokens`** — Claude Code calls this every turn to keep
  the prompt within the context window. Otari counts locally (no provider
  call, no budget debit) and returns `{"input_tokens": N}`. The count is an
  approximation; it is used only to gauge headroom, not for billing.

Token usage and cost from `/v1/messages` are recorded and reconciled exactly as
for `/v1/chat/completions`, so Claude Code sessions show up in usage and budgets
like any other client.
