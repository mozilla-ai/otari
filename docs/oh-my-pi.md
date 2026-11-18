# oh-my-pi

[oh-my-pi](https://github.com/can1357/oh-my-pi) (`omp`) is a coding agent built on
pi. The same [pi-otari](https://github.com/mozilla-ai/pi-otari) package that
serves [pi](pi.md) is also an omp plugin.

## Quick start

1. Install the plugin:

   ```bash
   omp plugin install @mozilla-ai/pi-otari
   ```

2. Log in with your Otari API key, from a shell or inside omp:

   ```bash
   omp login otari
   ```

3. Pick an `otari` model with `/model` and prompt as usual. Every model your key
   can use is listed; there is no `/scoped-models` step.

omp caches the model list for a day. Run `omp models refresh` after you change
models in Otari.

On a gateway you run, set `OTARI_BASE_URL` to its URL plus `/api/v1` first, as on
[pi](pi.md#on-a-gateway-you-run). For CI and containers, set
`OTARI_API_KEY` instead of logging in.

## Differences from pi

Sessions, web search and fetch, and guardrails work as on pi. Subagent routing does not: it relies on pi-subagents, which runs
on pi only. The details are under "Install and use on omp" in the
[pi-otari README](https://github.com/mozilla-ai/pi-otari#install-and-use-on-omp).

## See also

- [pi](pi.md)
- [Claude Code](claude-code.md)
