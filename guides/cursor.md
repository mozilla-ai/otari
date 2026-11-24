# Cursor

[Cursor](https://cursor.com) can send its Chat and Agent requests to any
OpenAI-compatible API, including Otari, through its OpenAI base URL override.

## Quick start

1. Create a key on the dashboard's **API keys** page.
2. In Cursor, open **Cursor Settings** > **Models**.
3. Under **API Keys**, enter your Otari key as the **OpenAI API Key** and turn
   on **Use OpenAI API Key**. Turn on **Override OpenAI Base URL**, set it to
   `https://api.otari.ai/api/v1`, and choose **Save**.
4. Under **Models**, choose **Add model**, enter `z-ai/glm-5.3`, and turn it
   on.
5. Pick `z-ai/glm-5.3` in Chat or Agent.

EU accounts use `https://eu.api.otari.ai/api/v1`. See
[Base URL](https://docs.otari.ai/reference#base-url).

## Choosing a model

The model name you add goes to Otari unchanged, so write it as Otari spells it:
a catalog id such as `moonshotai/kimi-k3.1-pro` lets Otari choose the provider,
and `anthropic:claude-sonnet-4-6` pins one. Add one entry for each model you
want to pick from. See [Model format](https://docs.otari.ai/reference/gateway/models#model-format) and
[Listing available models](https://docs.otari.ai/reference/gateway/models#listing-available-models).

## What goes through Otari

- **Chat and Agent** use the model you added. **Tab** completions always use
  Cursor's own models.
- Cursor sends these requests from its own servers, not from your machine, so
  the base URL has to be reachable from the internet. A gateway on `localhost`
  or a private network does not work.
- While the override is on, Cursor sends its built-in OpenAI models to the
  same base URL. Turn off the ones you do not want to reach Otari.

## See also

- [opencode](opencode.md)
- [Mistral Vibe](vibe.md)
