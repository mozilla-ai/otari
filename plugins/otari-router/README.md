# otari-router

Claude Code plugin for [Otari](https://github.com/mozilla-ai/otari), the
open-source LLM gateway. When Claude Code spawns a subagent, the plugin asks
the gateway which model that subagent should run on, and starts it there.

Switching the model inside a running conversation cold-starts its prompt
cache. A new subagent has no cache yet, so the moment it is spawned is where a
cheaper model is free. Claude Code can take Otari's pick at that moment even
when its own requests go to Anthropic directly.

## Installation

In a Claude Code session:

```
/plugin marketplace add mozilla-ai/otari
/plugin install otari-router@otari
```

From a terminal, the same two steps are `claude plugin marketplace add
mozilla-ai/otari` and `claude plugin install otari-router@otari`.

The install asks for two values, and both are required:

| Option | What it is |
| --- | --- |
| `otari_url` | Base URL of the gateway, such as `https://otari.example.com`. Plain `http` only for a gateway on this machine, since the key travels in a header. |
| `otari_api_key` | The key each recommendation is billed to. Stored in secure storage, never in a settings file. |

Change them later with `/plugin configure otari-router`. The plugin is active
in the session that installed it and in every session after.

To remove it: `claude plugin uninstall otari-router@otari`.

## What the plugin provides

One hooks module with two hooks:

1. **`agent.spawn`**: sends the spawn (subagent type, task, parent model, the
   model the caller asked for) to `POST /api/v1/routing/recommend` and starts
   the subagent on the model the gateway recommends. A fork is not asked
   about: Claude Code keeps a fork on its parent's model. When the gateway
   does not answer within three seconds, or answers with an error, the
   subagent starts on the model it would have had anyway, and a line in the
   transcript says so.
2. **`turn.complete`**: when a subagent finishes, logs the model it ran on and
   its token counts, so the pick can be checked against what it cost.

Each spawn leaves a dim transcript line such as
`Explore agent-… starts on claude-haiku-4-5, Otari's choice (jev-1.13.0 chose haiku with 81%)`.

## Requirements

- An Otari gateway. A standalone one needs a
  [decision provider](../../docs/configuration.md#decision-providers)
  configured, and the recommendation is one decision-model call billed to the
  key above; on otari.ai a managed recommender answers, with nothing to
  configure. A hybrid gateway does not serve the route, so the plugin would
  fall back on every spawn. See [Use with Claude Code](../../docs/use-with-claude-code.md#let-otari-choose-a-subagents-model)
  for what the gateway does with the request.
- A Claude Code build with the mods API, which hooks modules run on.

## Development

```bash
claude plugin validate plugins/otari-router
claude plugin test plugins/otari-router
```

To run the checkout's copy in your own sessions, add the repository folder as
a marketplace, `claude plugin marketplace add /path/to/otari`, and install
from it. A plugin installed from a folder is read in place, so an edit reaches
an open session with `/reload-plugins`. Claude Code lays editor typings into
`.claude-plugin/types/` under the plugin when it loads; that folder is
gitignored.

A plugin installed from GitHub is a copy pinned by `version` in
`plugin.json`. A change to the plugin bumps that version, and users pick it up
with `claude plugin update otari-router@otari`.

## License

[Apache-2.0](../../LICENSE)
