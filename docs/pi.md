# pi

[pi](https://pi.dev) is a minimal terminal coding agent. The
[pi-otari](https://github.com/mozilla-ai/pi-otari) extension adds Otari as a
provider, so pi can use every model your workspace serves, with its budgets,
routing and usage tracking.

## Quick start

You need pi 0.81.0 or newer and an Otari API key.

1. Install the extension:

   ```bash
   pi install npm:@mozilla-ai/pi-otari
   ```

2. Start `pi` and run `/login otari`. Paste your API key at the prompt, which is
   kept out of the chat.
3. Run `/scoped-models`, search for `otari`, enable the models you want, and press
   Ctrl+S.
4. Run `/model` and pick an `otari` model. The status line shows
   `Otari → <model-id>`, and you prompt as usual.

The model list comes from your workspace and is cached; opening `/model` refreshes
it after you change models in Otari.

## On a gateway you run

Point the extension at your gateway's URL plus `/api/v1` before starting pi, then
log in as above:

```bash
export OTARI_BASE_URL=http://localhost:8000/api/v1
pi
```

EU accounts on otari.ai use `https://eu.api.otari.ai/api/v1`. For CI and
containers, set `OTARI_API_KEY` instead of logging in.

## What else it does

- **Sessions.** Each request carries pi's session id, so a run shows as one
  [session](sessions.md).
- **Web search and fetch.** Otari's [built-in tools](tools.md) are offered to the
  model where the deployment runs them. `OTARI_WEB_SEARCH=off` and
  `OTARI_WEB_FETCH=off` turn them off.
- **Subagent routing.** With [pi-subagents](https://github.com/nicobailon/pi-subagents)
  installed, Otari can pick the model each new subagent runs on; see
  [Subagent routers](routing.md#subagent-routers). It needs pi 0.99.0 or newer
  and a gateway with a subagent router configured, and is off until you turn it
  on in pi's settings.
- **Guardrails.** With the `otari` CLI on `PATH`, pi checks the rules in your
  repository's `.otari/guardrails.yml` before each `bash`, `read`, `edit` and
  `write` call, and when the agent is about to stop. If `otari` cannot run or
  times out, the call is allowed with a warning; see
  [Guardrails](https://github.com/mozilla-ai/pi-otari#guardrails) in the
  extension's README.

Every setting and the troubleshooting list are in the
[pi-otari README](https://github.com/mozilla-ai/pi-otari#readme).

## See also

- [oh-my-pi](oh-my-pi.md)
- [Claude Code](claude-code.md)
- [opencode](opencode.md)
