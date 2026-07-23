# Use with Claude Code

[Claude Code](https://code.claude.com) speaks the Anthropic Messages API and lets
you redirect it at any compatible endpoint with a couple of environment
variables. Otari exposes that surface (`POST /v1/messages` and
`POST /v1/messages/count_tokens`) in both standalone and hybrid modes, so you
can route Claude Code through Otari to get budgets, usage tracking, and traces
without changing how you use the CLI.

## Quick start

Claude Code appends `/v1/messages` and `/v1/messages/count_tokens` to
`ANTHROPIC_BASE_URL` itself, so the base URL must be the Otari root, not
`/v1`. For local development, use `http://localhost:8000`.

### Connected to otari.ai

```bash
export ANTHROPIC_BASE_URL="https://api.otari.ai"   # your Otari base URL, no /v1
export ANTHROPIC_AUTH_TOKEN="tk_your_otari_token"  # sent as Authorization: Bearer
export ANTHROPIC_MODEL="anthropic:claude-sonnet-4-6"
claude
```

### Standalone

Claude Code attaches its own `metadata.user_id` to every request. In standalone
mode Otari binds spend to the API key's own user and, by default, rejects a
request that names a different user (`403 permission_error`). Set
`reject_user_mismatch: false` in your config so Claude Code's `user_id` is
ignored and spend is still bound to the key's user.

```yaml
reject_user_mismatch: false
```

Then run Claude Code against your local Otari:

```bash
export ANTHROPIC_BASE_URL="http://localhost:8000"
export ANTHROPIC_AUTH_TOKEN="<your-otari-api-key>"
export ANTHROPIC_MODEL="anthropic:claude-sonnet-4-6"
claude
```

Use `ANTHROPIC_AUTH_TOKEN` (not `ANTHROPIC_API_KEY`): it is sent as
`Authorization: Bearer <token>`, which is the scheme Otari accepts for
both standalone API keys and connected user tokens. `ANTHROPIC_API_KEY` is sent
as an `x-api-key` header instead. Otari does read that header, so it also
authenticates, but `ANTHROPIC_AUTH_TOKEN` is the variable Claude Code intends for
a third-party endpoint and works uniformly across both modes.

### settings.json

The same configuration works in `~/.claude/settings.json` (or a project-level
`.claude/settings.json`). Replace the values with your deployment's URL, token,
and model defaults:

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

## Choosing a model

Claude Code speaks the Anthropic wire format, but the `model` field is just a
string Otari forwards to [any-llm](https://github.com/mozilla-ai/any-llm).
Examples in this page use `provider:model`; Otari also accepts
`provider/model` in both standalone and connected mode.

- **Claude models:** `anthropic:claude-sonnet-4-6`,
  `anthropic:claude-opus-4-8`, `anthropic:claude-haiku-4-5`
- **Standalone, any configured provider:** `openai:gpt-4o`,
  `mistral:mistral-large-latest`, `anthropic:claude-sonnet-4-6`
- **Connected to otari.ai, managed models:** `mzai:<catalog-id>`, for example
  `mzai:moonshotai/Kimi-K2.6`. These run only through otari.ai's hosted gateway.
- **Connected to otari.ai, your own provider keys:** `openai:gpt-4o`,
  `mistral:mistral-large-latest`, `anthropic:claude-sonnet-4-6`

If you use Claude Code's `opus`, `sonnet`, or `haiku` aliases, set the matching
`ANTHROPIC_DEFAULT_*_MODEL` variables so Claude Code does not fall back to a
model your Otari deployment does not serve.

### Non-Claude models

Otari can route Claude Code to non-Claude models, but Claude Code is tuned for
Claude. Expect weaker tool use on some models, and expect Anthropic-specific
features such as extended thinking or prompt caching to be dropped when the
target provider does not support them.

## See also

- [Modes](modes.md) for standalone vs connected behavior
- [API reference](api-reference.md) for the Messages endpoints and auth rules
