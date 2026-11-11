# Junie

[Junie](https://junie.jetbrains.com) is JetBrains' coding agent. Two JetBrains
products can route through Otari's OpenAI-compatible API, each with its own
setup:

- **Junie CLI**, the terminal agent, through a custom model profile.
- **JetBrains AI Assistant**, inside PyCharm, IntelliJ IDEA and the other IDEs.

## Junie CLI

1. On the dashboard's **API keys** page, create a key that accepts Junie's
   `user` field (see [Junie sends a `user` field](#junie-sends-a-user-field)).
2. Write a profile to `~/.junie/models/otari.json`:

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

3. Export the key and select the profile by its file name, `custom:otari`:

   ```bash
   export OTARI_JUNIE_KEY=<your-otari-key>
   junie --model custom:otari "Add a docstring to the add function in calc.py"
   ```

Things to get right in the profile:

- `baseUrl` is the full chat completions URL, not the API root, because Junie
  does not add the path itself. EU accounts use
  `https://eu.api.otari.ai/api/v1/chat/completions`, and a gateway you run uses
  its URL plus `/api/v1/chat/completions`. See
  [Base URL](https://docs.otari.ai/reference#base-url).
- The profile is selected by file name. `providerName` is only a label.
- `id`, `primaryModel` and `fasterModel` take any model your key can use, as
  Otari spells it; see [Model format](https://docs.otari.ai/reference/gateway/models#model-format) and
  [Listing available models](https://docs.otari.ai/reference/gateway/models#listing-available-models).

### Junie sends a `user` field

Junie CLI puts a `user` field in every request, with a value you do not control.
An ordinary key refuses it with `403`. The CLI does not show that refusal: the
task stops with a message about an artifact it could not build, and the real
reason is only in `~/.junie/logs/`:

```
LLMBadResponseException: 'user' field does not match the authenticated API key's user
```

Pick one of two fixes:

- **Make the key a [service key](end-users.md)** (recommended). Otari records
  Junie's value as an end user with its own spend, which you can put on a
  budget.
- **Accept the field on the key**: set **Mismatched `user` field** to
  **Always accept** under **Advanced** in the key form. All of Junie's usage
  then lands on the key's user.

Either way, spend stays within the key owner's account. See
[Mismatched `user` fields](api-keys.md#mismatched-user-fields).

### Troubleshooting the CLI

- `junie --provider litellm --litellm-url <url>` finds no models, because it
  asks for a LiteLLM-only route Otari does not serve. Use a custom profile.
- The model picker has tabs (`tab` and `shift+tab`), and a custom profile is
  not on the JetBrains AI tab that opens first.
- A model whose price Otari does not know shows as $0.00.

## JetBrains AI Assistant

In AI Assistant's settings, add an OpenAI-compatible provider with the Otari API
root (such as `https://api.otari.ai/api/v1`) and one of your keys.

AI Assistant sends no `user` field, so any key works. It shows its own default
context window rather than the one Otari reports.

## See also

- [Claude Code](claude-code.md)
- [Codex](codex.md)
- [opencode](opencode.md)
- [How a request is attributed](access-control.md#how-a-request-is-attributed)
