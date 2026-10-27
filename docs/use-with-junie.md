# Use with Junie

[Junie](https://junie.jetbrains.com) is JetBrains' coding agent. Two different JetBrains products can route through Otari, and each needs its own setup:

- **Junie CLI**, the terminal agent, through a custom model profile.
- **JetBrains AI Assistant**, inside PyCharm, IntelliJ IDEA and the other IDEs, through its OpenAI-compatible provider.

Otari exposes an OpenAI-compatible endpoint (`POST /api/v1/chat/completions`), so either one gets budgets, usage tracking and traces without changing how you code.

## Junie CLI

Junie CLI reaches an OpenAI-compatible backend through a custom model profile. Write one to `~/.junie/models/otari.json`:

```json
{
  "baseUrl": "https://api.otari.ai/api/v1/chat/completions",
  "id": "openai:gpt-4o-mini",
  "apiType": "OpenAICompletion",
  "apiKey": "${OTARI_JUNIE_KEY}",
  "displayName": "Otari",
  "providerName": "otari",
  "primaryModel": { "id": "openai:gpt-4o-mini" },
  "fasterModel": { "id": "openai:gpt-4o-mini" }
}
```

Two details differ from most clients:

- `baseUrl` is the full chat completions URL, not the API root. Most clients take `/api/v1` and append the path themselves. Junie does not, so give it your Otari base URL plus `/api/v1/chat/completions`: `http://localhost:8000/api/v1/chat/completions` for local development, or your own deployment's URL.
- `apiKey` accepts a `${VAR}` reference, so the key stays out of the file.

Export the key, then select the profile by its `providerName`:

```bash
export OTARI_JUNIE_KEY=<your-otari-key>
junie --model custom:otari "Add a docstring to the add function in calc.py"
```

Any model in your catalog works in `id`, `primaryModel` and `fasterModel`, using the `provider:model` selector that `GET /api/v1/models` returns for your key.

### Junie sends a `user` field

Junie CLI puts `"user": "primary"` in every chat completion body, for every internal role it runs, including the main agent, the step summarizer and the chat title creator. Otari reads `user` as naming the Otari user to bill, so by default a key other than the master key refuses a request that names anyone but its own user, with HTTP 403.

The CLI does not surface that refusal. The task stops with a message about an artifact it could not build, and the real reason appears only under `~/.junie/logs/`:

```
LLMBadResponseException: 'user' field does not match the authenticated API key's user
```

There are two ways to accept it, and they differ in what you get back.

**Make the key a [service key](access-control.md#service-keys-and-end-users).** This is the better option, and it needs no setting relaxed: leave `reject_user_mismatch` on the deployment default. On a service key a name other than the key's own user names one of its end users, so Otari creates an end user for the name Junie sends and records that end user's spend separately. You get per-client spend instead of one undifferentiated total, and you can give the end user a budget of its own.

**Or set `reject_user_mismatch=false` on the key.** Simpler, and it is a per-key override so the deployment-wide default stays as it is. Spend binds to the key's own user, so the whole of Junie's usage lands in one place. Nothing is weakened either way, because spend binds to the key's own user whenever the key is not a service key.

See [Mismatched `user` fields](access-control.md#mismatched-user-fields) for the full picture.

### Automatic discovery does not find Otari

`junie --provider litellm --litellm-url <url>` asks for `GET /v1/model/info`, an administrative route of the LiteLLM proxy. It is not part of the OpenAI API and Otari does not serve it, so discovery finds no models. Use a custom model profile instead.

Two more things about the CLI model picker:

- It has a tab per provider. A custom profile appears under its own `providerName`, not under the JetBrains AI tab.
- A model whose price Otari does not know is shown as $0.00, rather than as unknown.

## JetBrains AI Assistant

In PyCharm, IntelliJ IDEA and the other IDEs, AI Assistant can add an OpenAI-compatible provider. Give it the Otari API root (`/api/v1`) and one of your keys.

AI Assistant lists models with `GET /api/v1/models` and sends no `user` field, so it needs no change to the key's settings. It also ignores the context window Otari reports for a model and shows its own default instead.

## See also

- [Use with Claude Code](use-with-claude-code.md): point the Claude Code CLI at Otari.
- [Use with Codex](use-with-codex.md): route the Codex CLI through Otari over the Responses API.
- [Use with opencode](use-with-opencode.md): point the opencode CLI at Otari.
- [Access control](access-control.md): how budgets bind to a key's user.
