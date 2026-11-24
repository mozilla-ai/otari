# Set up coding tools with one command

`otari setup` connects the coding tools on a machine to Otari. You approve a key
with a spending limit in the browser, and the command does the rest: it stores
the key, writes a settings file for each tool it finds, and sends one test request.

It works on a gateway whose operator has
[registered the otari CLI](oauth-apps.md#register-the-otari-cli).

## Run it

```bash
uvx --from otari-agent otari setup --url https://your-gateway.example
```

The command prints an approval link and opens it. The page names the app,
"Otari CLI", and starts from a limit of 10 USD and a key that expires in 30 days;
change either before you approve. `--limit` and `--expiration` change what the
page starts from.

When it finishes it has:

- stored the key in the macOS Keychain, or in an owner-only file under
  `~/.config/otari/keys/` on other systems;
- written a settings file for each of Claude Code, Codex and OpenCode that is
  installed (`~/.claude/otari.json`, `~/.codex/otari.config.toml`,
  `~/.config/otari/opencode.json`);
- sent one request to the model the tools start on, and printed the reply, its
  cost and what is left on the key.

Your own settings for those tools are not changed, so a plain `claude` or `codex`
behaves as before. No settings file holds the key: Claude Code and Codex run a
command that reads it from where it is stored, and OpenCode is handed it when it starts.

## Start a tool

```bash
otari claude
otari codex
otari opencode
```

Each starts the tool with the settings above. Anything after the command name
goes to the tool unchanged, so `otari codex exec "fix the tests"` works. Without
`otari` installed, run them as `uvx --from otari-agent otari claude`, or install
it with `uv tool install otari-agent`.

## Let a coding agent run it

`otari setup` never asks a question, so an agent can run it for you: the one
step left to you is the approval in the browser. `--json` prints a single JSON
object when it is done. Neither form prints the key. The setup guide's Agent tab
gives a prompt to paste that does this.

An agent that can run commands as you can read anything you can, a stored key
included. The limit and expiry you approve bound what that key can spend, and the
API keys page revokes it.

## Options

| Option | What it does |
| --- | --- |
| `--url` | The gateway's API address. Also read from `OTARI_URL`. |
| `--tool` | Configure only this tool: `claude-code`, `codex` or `opencode`. Repeatable. |
| `--model` | The model the tools start on. Default: the first coding model the workspace serves. |
| `--limit`, `--expiration` | The spending limit and lifetime the approval page starts from. |
| `--new-key` | Ask for a new key even if a working one is stored. |
| `--no-browser` | Print the approval link without opening it. |
| `--no-test` | Skip the test request. |
| `--json` | Print one JSON object when done. |

## Undo it

```bash
otari logout
```

revokes the key, removes it from this machine, and deletes the settings files the command wrote.
