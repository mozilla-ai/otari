# Agent Guardrails

Agent Guardrails are rules a repository keeps about what a coding agent may do to
its working tree. They are checked while the agent works: a forbidden tool call is
refused before it runs, and a turn that ends in a forbidden state is blocked
before the agent hands it back.

The rules live in `.otari/`, committed with the code. The `otari hook` command
runs them from the agent's own hook system. It needs no gateway, no account and no
credential, unless you choose to [report its checks to Otari](#reporting-to-otari).

These are not [Inference Guardrails](guardrails.md), which check what a model is
asked before the provider is called. This page walks through setting Agent Guardrails up;
every field, gate type, option and limit is in the
[Agent Guardrails reference](https://docs.otari.ai/reference/agent-guardrails/agent-guardrails-reference).

## Install

`otari hook` is part of the `otari` command, which installs without the gateway
server:

```bash
brew install mozilla-ai/tap/otari
```

Without Homebrew, install the sdist from a Release:

```bash
uv tool install https://github.com/mozilla-ai/otari/releases/download/vX.Y.Z/otari_agent-X.Y.Z.tar.gz
```

| Command | What it does |
| --- | --- |
| `otari hook setup` | Registers the hook in a supported agent's settings file. |
| `otari hook` | The hook itself. The agent runs it; you do not. |
| `otari guardrails validate` | Checks a guardrail offline, and dry-runs it against a command or path. |
| `otari guardrails check` | Runs the gates against a change, such as a pull request. |
| `otari guardrails generate` | Proposes gates from the repository's AGENTS.md or CLAUDE.md. |

Options for each are under [Command-line reference](https://docs.otari.ai/reference/agent-guardrails/agent-guardrails-reference#command-line-reference).

## Guardrails and gates

A guardrail is the intent ("do not hand-edit the changelog"). A gate is one check
that enforces it. A guardrail file names the intent under `policy.id` and lists its
gates under `gates`.

Each gate has:

- an `enforcement`: `required` blocks the tool call or the end of the turn, and
  `advisory` warns while work continues;
- a `runs` list: the moments it fires, before a tool call or at the end of the
  turn.

| Type | Fails when | Example rule |
| --- | --- | --- |
| `path` | a path in scope matches a forbidden glob | nothing hand-edits a generated file |
| `command` | a command in scope matches a forbidden phrase | no `git push --force` |
| `command_if_changed` | a path changed but a required command never ran | a changed spec must be followed by its generator |
| `judge` | a model reading the diff reports the rubric was not met | the change follows the error-handling conventions |
| `verifier` | a script in the repository exits non-zero | anything the others cannot express |

The first three are plain pattern matches. `verifier` runs your script. `judge` is
the only type that calls a model, and it can only be `advisory`.

## Registering the hook

Run `otari hook setup` inside the Git repository you want to guard.

### Claude Code

```bash
otari hook setup
```

This adds a `PreToolUse` and a `Stop` entry to `.claude/settings.local.json`.
Restart Claude Code if a gate does not fire in the current session.

If the repository has no guardrail yet, `setup` offers to write a starter
`.otari/guardrails.yml` with four `advisory` gates: no `git push --force`, no
`.env` in the tree, and a pair that refuses reading `.env` through the read tool
and through the shell.

Re-run `setup` after adding your first `command` gate or your first gate on
`pre_tool_use.read_target`, so the hook also listens for the tool that gate needs.
Re-running leaves your other hooks and permissions untouched. What `setup` writes
is under [Registering it by hand](https://docs.otari.ai/reference/agent-guardrails/agent-guardrails-reference#registering-it-by-hand).

### Codex

```bash
otari hook setup --harness codex
```

This writes the same pair of entries to `.codex/hooks.json`. Codex runs a
project's hooks only after you accept them: start Codex in the repository and run
`/hooks`. Each developer accepts them separately in each clone.

The Codex integration is newer than the Claude Code one; see
[Registering it for Codex](https://docs.otari.ai/reference/agent-guardrails/agent-guardrails-reference#registering-it-for-codex)
for what it cannot see.

## Writing a gate

A gate is one entry under `gates:` in a guardrail file:

```yaml
  - id: no-hand-edited-changelog
    type: path
    runs: [pre_tool_use.edit_target, stop.working_tree]
    enforcement: required
    forbidden: ["CHANGELOG.md"]
    message: >-
      CHANGELOG.md is generated at release time. Do not hand-edit it.
```

An `Edit` call naming `CHANGELOG.md` is now refused before it runs, and the agent
receives `message`.

For a write rule, name both `runs` entries. `pre_tool_use.edit_target` refuses
the edit tools, and `stop.working_tree` catches a file a shell command wrote, at
the end of the turn. See [Why a write rule names both moments](#why-a-write-rule-names-both-moments).

### Refusing the command instead

A `command` gate refuses a shell command before it runs:

```yaml
  - id: no-in-place-shell-edits
    type: command
    runs: [pre_tool_use.command]
    enforcement: required
    forbidden: ["sed -i", "perl -pi"]
    message: >-
      Edit files with the Edit tool rather than in place from the shell, so
      the repository's path rules can see what is about to be written.
```

Put the tool in the phrase, not the file name. A phrase matches consecutive
words, so `"sed -i CHANGELOG.md"` never matches `sed -i '' 's/x/y/' CHANGELOG.md`.
The matching rules are under [`command`](https://docs.otari.ai/reference/agent-guardrails/agent-guardrails-reference#command).

### Keeping secrets out of the transcript

A secrets rule takes two gates: a `path` gate on `pre_tool_use.read_target`,
which refuses the `Read` tool, and a `command` gate for shell reads such as
`cat .env`. A read has no end-of-turn check to fall back on; see
[Reads have no second moment](#reads-have-no-second-moment). The `path` half is
shown under [Refusing a read](https://docs.otari.ai/reference/agent-guardrails/agent-guardrails-reference#refusing-a-read).

## Where the files live

Start with the single file `otari hook setup` creates:

```
.otari/
  guardrails.yml
  verifiers/
    no-conflict-markers.sh
```

When one file gets crowded, split it into a directory, grouped by concern rather
than by gate type:

```
.otari/
  guardrails/
    secrets.yml
    generated-artifacts.yml
    architecture/
      layer-rules.yml
  verifiers/
    check-architecture.sh
```

`otari hook` reads `.otari/guardrails.yml` and every `.yml` or `.yaml` under
`.otari/guardrails/`, nested ones included, and combines them. Both shapes can
coexist. Keep verifier scripts in `.otari/verifiers/`. The rules for combining
files are under [Composing several files](https://docs.otari.ai/reference/agent-guardrails/agent-guardrails-reference#composing-several-files).

## Checking a guardrail before it runs

A gate that never matches looks the same as a clean result, so check it first.

1. Check the guardrail for errors and warnings:

   ```bash
   otari guardrails validate
   ```

   An error is a gate that cannot work; a warning is a gate that may not mean
   what you intended. Each one is listed under
   [Validation errors and warnings](https://docs.otari.ai/reference/agent-guardrails/agent-guardrails-reference#validation-errors-and-warnings).

2. Dry-run it against the paths and commands you expect a session to touch:

   ```
   $ otari guardrails validate --path docs/public/openapi.json --command "make postman"
   .otari/guardrails: composed from 8 files, 24 gate(s), schema 1.0.
   0 error(s), 0 warning(s).

   PreToolUse, Bash: make postman
     quiet      use-pnpm-not-npm (required)
     quiet      no-force-push (advisory)
     20 gate(s) do not apply here.

   PreToolUse, Edit/Write: docs/public/openapi.json
     quiet      no-hand-edited-changelog (required)
     ...

   Stop, the finished turn: 1 changed path(s), 1 command(s)
     quiet      openapi-changed-needs-postman (required)
     fires      openapi-changed-needs-generator (required)
     would run  no-leftover-conflict-markers (verifier, required)
     skipped    no-narrative-comments (judge, when_changed does not match)
   ```

   Check that the gates you expect say `fires`. No model is called, so a `judge`
   gate says `would run`. The labels are under
   [Dry-run output](https://docs.otari.ai/reference/agent-guardrails/agent-guardrails-reference#dry-run-output).

3. If a warning describes what you intend, list its code in the gate's
   [`accept_warnings`](https://docs.otari.ai/reference/agent-guardrails/agent-guardrails-reference#accept_warnings-on-any-gate).

4. In CI, make warnings fail too:

   ```bash
   otari guardrails validate --strict
   ```

   If the hook runs with a raised judge cap, pass `validate` the same
   `--max-judges` value.

## Checking a pull request in CI

`otari hook` runs only where it is registered, so a pull request from someone
without the hook skips every gate. `otari guardrails check` runs the gates against
a change instead:

```bash
otari guardrails check --base origin/main
```

This checks everything since the branch left `origin/main`, untracked files
included. Without `--base` it checks uncommitted changes. `command` and
`command_if_changed` gates cannot run here, since a diff has no tool calls; see
[`otari guardrails check`](https://docs.otari.ai/reference/agent-guardrails/agent-guardrails-reference#otari-guardrails-check).

To run it on every pull request:

1. Take the guardrail files and verifiers from the base commit, so a pull request
   cannot change the gates that check it: check out the base beside the pull
   request and pass `--guardrails-from <base checkout>`.
2. Run `--type path --type verifier` in the pull request's tree. Stage the whole
   pull request against its merge base first (for example
   `git reset --soft <merge base>`), since a verifier that reads `git status` sees
   only uncommitted changes.
3. Run `--type judge` as a separate job, since it needs a model CLI and a
   credential. Run it from the base branch's workflow (`pull_request_target` on
   GitHub), fetch the pull request as Git objects only, and pass
   `--base <base> --head <head>`, so no pull request code runs next to the
   credential. Codex still sees the diff with its tools in a read-only sandbox,
   which can read files, so on untrusted diffs prefer `claude -p` or run Codex
   where no credential is readable.
4. Make the `path` and `verifier` job a required check, and review changes to it
   and its verifier scripts, since that job runs the pull request's code.

Otari's own repository does this in
[`.github/workflows/otari-guardrails-check.yml`](https://github.com/mozilla-ai/otari/blob/54c67f5986b090db1198ebf43f9c879b3f5779a2/.github/workflows/otari-guardrails-check.yml)
and [`.github/workflows/otari-guardrails-judges.yml`](https://github.com/mozilla-ai/otari/blob/54c67f5986b090db1198ebf43f9c879b3f5779a2/.github/workflows/otari-guardrails-judges.yml).

## Reporting to Otari

`otari hook` can report every gate it checks to an Otari server, so a team sees
which gates ran, which blocked, and which never fire. Set the server and an API
key where the agent runs:

```bash
export OTARI_URL="http://localhost:8000"     # your gateway's URL
export OTARI_API_KEY="tk-your-import-key"    # must be budget-exempt
```

- **The key must be budget-exempt.** Only a deployment's operator can create one
  (see [Importing external usage](external-usage.md#enforcement-boundary)), so
  reporting works on a gateway you run, not on otari.ai today.
- **Claude Code** does not pass these variables to its hooks; set them in the
  `env` block of `~/.claude/settings.json`, as
  [Send Claude Code sessions to Otari](use-with-claude-code.md#send-claude-code-sessions-to-otari)
  shows. Keep the key out of project settings files you commit.
- **The server sees your gates**, including their messages and your own gates in
  `~/.otari/`. Point `OTARI_URL` only at a server you trust with them. It keeps
  up to 1,000 versions of the guardrail per workspace; past that, gates show by
  id alone.

Reporting never slows a check or changes its result. Sessions in the dashboard
then show each gate's message where it blocked, list declared gates that never
ran, and, when the agent also exports its spans, show a gate under the tool call
or turn it checked.

- `OTARI_HOOK_REPORT_CONTENT=1` also sends the command or paths a gate checked,
  kept only where the workspace keeps content (see
  [Keeping what a span was about](traces.md#keeping-what-a-span-was-about)).
- `OTARI_HOOK_REPORT=off` turns reporting off.

What a report carries, and the server's limits, are under
[What `otari hook` reports](https://docs.otari.ai/reference/agent-guardrails/agent-guardrails-reference#what-otari-hook-reports).

## Generating gates from AGENTS.md or CLAUDE.md

`otari guardrails generate` proposes gates from the repository's AGENTS.md, or
CLAUDE.md when that is the only one:

```bash
otari guardrails generate
```

1. Confirm the model CLI it names (`claude -p` or `codex exec`) before it sends
   the file.
2. Answer each proposal with `y`, `n`, `e` (edit) or `q`. Only accepted gates are
   written.
3. Run `otari guardrails validate` on the result.

Rerunning it proposes from scratch, including gates you declined before. Options
are under [`otari guardrails generate`](https://docs.otari.ai/reference/agent-guardrails/agent-guardrails-reference#otari-guardrails-generate).

## Your own guardrail in `~/.otari/`

Rules that belong to you rather than to a repository (no `git reset --hard`, run
the linter before stopping) go in your home directory, in the same shapes:

```
~/.otari/
  guardrails.yml
  guardrails/
    git-safety.yml
  verifiers/
    lint-before-stop.sh
```

Your gates apply in every repository where the hook runs, only for you, and a
repository's files cannot turn them off. Their ids get a `user:` prefix.

- A `verifier` names its script relative to your home directory, such as
  `.otari/verifiers/lint-before-stop.sh`, and the script must be inside
  `~/.otari/verifiers/`; see [`verifier`](https://docs.otari.ai/reference/agent-guardrails/agent-guardrails-reference#verifier).
- `otari guardrails validate` checks your files with the repository's. Add
  `--repo-only` to check the repository's alone.
- `otari guardrails check` always leaves your files out, so CI reads the same on
  every machine.

How the two sets combine is under [Composing several files](https://docs.otari.ai/reference/agent-guardrails/agent-guardrails-reference#composing-several-files).

### Turning your own gates off in one repository

```bash
otari hook setup --repo-only
```

The hook then enforces that repository's guardrail alone. Pass the flag to
`setup` rather than editing the settings file, because a later `setup` run
rewrites the entry. If your Claude Code user settings also register the hook for
every repository, that entry still enforces your gates; `setup --repo-only` warns
about it.

### Running your guardrail in every repository

`otari hook setup` registers the hook in one repository only. To run your gates
everywhere:

1. Add the entries `setup` writes (see [Registering it by hand](https://docs.otari.ai/reference/agent-guardrails/agent-guardrails-reference#registering-it-by-hand))
   to Claude Code's user settings file, `~/.claude/settings.json`, with the
   matcher `Edit|Write|NotebookEdit|Read|Bash`.
2. Remove the per-repository entries on that machine, or the hook can run twice
   for each event.

## Recommendations

- **Name both moments in a write rule**: `pre_tool_use.edit_target` alone misses
  shell writes.
- **Start a new rule `advisory`**, and make it `required` once it has run without
  false positives.
- **Write `message` for the agent.** It is all the agent sees. Say what to do
  instead, and name the document the rule came from.
- **Name a real command in `forbidden`**: `["npm install"]`, not `["npm"]`, which
  also refuses `grep -rn npm web/`.
- **Give each required command its own `command_if_changed` gate**, because any
  one entry in `require` satisfies a gate.
- **Rules over generated files**: the generator writes the file too, and
  `stop.working_tree` cannot tell the two apart. Use `command_if_changed` ("if
  this changed, the generator must have run").
- **Keep verifiers read-only**, since they run at the same time against one
  working tree.
- **Scope `judge` gates** with `when_changed`, since each is a model call per
  `Stop` event, and set `priority` on one that must run. See
  [`judge`](https://docs.otari.ai/reference/agent-guardrails/agent-guardrails-reference#judge).
- **Run `validate --strict` and `check` in CI.**

## How Agent Guardrails work, and their limits

A rule in AGENTS.md is context that competes with everything else the agent
reads. A gate is evaluated: it matches the evidence or it does not. The evidence
is the path or command a tool call declares, and `git status --porcelain` at the
end of the turn, not the agent's own account of what it did. A `PreToolUse` gate
runs before the action, so a refusal leaves nothing to undo.

A `judge` gate is the exception, and is advisory for that reason: a model's
verdict is not reproducible, and the diff it reads can be influenced by an
attacker.

### Why a write rule names both moments

`pre_tool_use.edit_target` sees the path an `Edit`, `Write` or `NotebookEdit` call
declares, and refuses it before it writes. A shell command declares no path:

```bash
echo "## 1.2.0" >> CHANGELOG.md
sed -i '' 's/foo/bar/' CHANGELOG.md
python scripts/bump.py
```

Nothing before the command runs says what it will write. `stop.working_tree`
covers that: at the end of the turn it checks the same globs against
`git status --porcelain`, which sees the file whatever wrote it, except an
ignored file or a change already committed in the turn. It cannot prevent the
write, so the turn is blocked and the change has to be reverted.

A `command` gate can refuse the spellings it names, but not what a script does
inside. It catches mistakes by a cooperative agent; it is not a security boundary.

### Reads have no second moment

A read changes nothing, so `git status` has nothing to report at the end of the
turn. `cat .env` is refused by a `command` gate or not at all, which is why a
secrets rule is two gates.

### Known gaps

- There are no reusable packs of gates to share across a team's repositories.
- `otari hook setup` registers per repository only; running your gates everywhere
  needs a hand-written user-level entry.
- A pull request cannot be checked against `command` or `command_if_changed`
  gates.
- Nothing reports a secret that was already read through the shell.
- Claude Code overrides a `Stop` gate that blocks eight times in a row, and
  Codex's Code Mode bypasses `PreToolUse`; see
  [How the harness integrations collect evidence](https://docs.otari.ai/reference/agent-guardrails/agent-guardrails-reference#how-the-harness-integrations-collect-evidence)
  and [Registering it for Codex](https://docs.otari.ai/reference/agent-guardrails/agent-guardrails-reference#registering-it-for-codex).
- A verifier from a branch you check out to review runs on the next `Stop` event
  without you choosing to run it; see [`verifier`](https://docs.otari.ai/reference/agent-guardrails/agent-guardrails-reference#verifier).
