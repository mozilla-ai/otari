# Agent Guardrails

Agent Guardrails are rules a repository keeps about what a coding agent may do
to its working tree, checked while the agent works. A forbidden tool call is
refused before it runs; a turn that ends in a forbidden state is blocked before
the agent hands it back.

The rules live in `.otari/`, committed with the code. `otari hook` runs them: a
coding agent's hook system invokes it, it collects the evidence that event
carries, and it evaluates the rules in process. It needs no gateway, no
account, and no credential, unless you choose to
[report its checks to Otari](#reporting-to-otari).

This is not [Inference Guardrails](guardrails.md), which check request input and
output at inference time. A gate checks agent actions and repository diffs.

This page walks through setting them up. Every field, gate type, command-line
option and limit is specified in the
[Agent Guardrails reference](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/reference/agent-guardrails/agent-guardrails-reference).

Contents:

- [Install](#install)
- [Guardrails and gates](#guardrails-and-gates)
- [Registering the hook](#registering-the-hook)
- [Writing a gate](#writing-a-gate)
- [Where the files live](#where-the-files-live)
- [Checking a guardrail before it runs](#checking-a-guardrail-before-it-runs)
- [Checking a pull request in CI](#checking-a-pull-request-in-ci)
- [Reporting to Otari](#reporting-to-otari)
- [Generating gates from AGENTS.md or CLAUDE.md](#generating-gates-from-agentsmd-or-claudemd)
- [Your own guardrail in `~/.otari/`](#your-own-guardrail-in-otari)
- [Recommendations](#recommendations)
- [How Agent Guardrails work, and their limits](#how-agent-guardrails-work-and-their-limits)

## Install

`otari hook` is part of the `otari` command, which installs on its own and
needs none of the gateway server:

```bash
brew install mozilla-ai/tap/otari
```

Without Homebrew, install the sdist from a Release:

```bash
uv tool install https://github.com/mozilla-ai/otari/releases/download/vX.Y.Z/otari_agent-X.Y.Z.tar.gz
```

A source checkout's virtualenv carries the same command. See
[cli/README.md](https://github.com/mozilla-ai/otari/blob/main/cli/README.md) for the rest of that distribution.

Five commands apply here:

| Command | What it does |
| --- | --- |
| `otari hook setup` | Registers the callback in a supported agent's own settings file. |
| `otari hook` | The callback itself. The harness invokes it; it is not run by hand. |
| `otari guardrails validate` | Checks the composed guardrail offline, and dry-runs it against a command or path. |
| `otari guardrails check` | Runs the gates against a change, such as a pull request, without a session. |
| `otari guardrails generate` | Proposes gates from the repository's own AGENTS.md or CLAUDE.md. |

Their options are listed under
[Command-line reference](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/reference/agent-guardrails/agent-guardrails-reference#command-line-reference).

## Guardrails and gates

A guardrail is the intent ("do not hand-edit the changelog"). A gate is one
check that enforces it. A guardrail file names the intent under `policy.id` and
lists its gates under `gates`.

Each gate has an `enforcement`: a `required` gate blocks (the tool call is
refused, or the turn does not end), and an `advisory` gate warns while the work
continues. Each gate also has a `runs` list, which says at which moment it
fires and what it can see there: before a tool call, or at the end of the turn.

Five gate types are available:

| Type | Fails when | Example rule |
| --- | --- | --- |
| `path` | a path in scope matches a forbidden glob | nothing hand-edits a generated file |
| `command` | a command in scope matches a forbidden phrase | no `git push --force` |
| `command_if_changed` | a path changed but a required command never ran | a changed spec must have been followed by its generator |
| `judge` | a model reading the diff reports the rubric was not met | the change follows the repository's error-handling conventions |
| `verifier` | a script in the repository exits non-zero | anything the four above cannot express |

The first three are mechanical: a glob or phrase match, with no model and no
subprocess. `verifier` runs a script the repository supplies. `judge` is the
only type that calls a model, and it can only be `advisory`.

## Registering the hook

Run `otari hook setup` from inside the repository being guarded. It requires a
Git repository: it refuses outside one, and `git status` is where the `Stop`
event's evidence comes from.

### Claude Code

```bash
otari hook setup
```

This writes a `PreToolUse` entry and a `Stop` entry into
`.claude/settings.local.json`, both naming this install's own
`otari hook --harness claude-code`. Restart Claude Code if a gate does not fire
in the session `setup` ran in.

When the repository has no guardrail yet, `setup` offers to write a starter
`.otari/guardrails.yml` first, so there is something to check rather than a hook
that always passes. The starter carries one `command` gate refusing
`git push --force` and one `path` gate keeping `.env` out of the tree, both
`advisory`.

Re-running `setup` updates both entries in place and leaves every other hook and
permission in the file untouched. Re-run it when the guardrail gains its first
`command` gate or its first gate naming `pre_tool_use.read_target`, so the
`PreToolUse` matcher picks up the tool that gate needs. An ordinary `path` gate
needs no re-run. What `setup` writes is shown under
[Registering it by hand](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/reference/agent-guardrails/agent-guardrails-reference#registering-it-by-hand).

### Codex

```bash
otari hook setup --harness codex
```

The same pair of entries, written to `.codex/hooks.json` and naming
`otari hook --harness codex`.

Then accept them: Codex does not run a project's hooks until they are accepted.
Start Codex in the repository and run `/hooks`, which lists what the project
declares and asks for approval. Until then the file exists and nothing fires.
This is per developer and per repository: a teammate who clones the repository
and runs `otari hook setup --harness codex` accepts them separately.

The Codex integration is newer and less exercised than the Claude Code one; see
[Registering it for Codex](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/reference/agent-guardrails/agent-guardrails-reference#registering-it-for-codex)
for what it cannot see.

## Writing a gate

A gate is one mapping under `gates:` in a guardrail file:

```yaml
  - id: no-hand-edited-changelog
    type: path
    runs: [pre_tool_use.edit_target, stop.working_tree]
    enforcement: required
    forbidden: ["CHANGELOG.md"]
    message: >-
      CHANGELOG.md is generated at release time. Do not hand-edit it.
```

With that in place, an `Edit` call naming `CHANGELOG.md` is refused before it
runs, the agent receives `message`, and `git status` afterwards reports no
change, because the edit never happened.

Name both `runs` entries for a write rule. `pre_tool_use.edit_target` refuses
the edit tools before they write, and `stop.working_tree` catches anything a
shell command wrote, once the turn is over; see
[Why a write rule names both moments](#why-a-write-rule-names-both-moments).
`otari hook setup` already registers the `Stop` hook, so nothing further is
installed for the second entry.

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

Name the tool in the phrase, not the file. A phrase matches a contiguous run of
tokens, so `sed -i '' 's/x/y/' CHANGELOG.md` never fires a
`"sed -i CHANGELOG.md"` phrase, and `head -5 .env` never fires a `head .env`
one. The matching rules and a measured table of what matches are under
[`command`](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/reference/agent-guardrails/agent-guardrails-reference#command).

### Keeping secrets out of the transcript

A rule about secrets is two gates: a `path` gate that runs at
`pre_tool_use.read_target`, which refuses the `Read` tool, and a `command` gate
for the shell reads (`cat .env`) the first one cannot see. A read has no
end-of-turn moment to fall back on; see
[Reads have no second moment](#reads-have-no-second-moment). The `path` half
is shown under [Refusing a read](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/reference/agent-guardrails/agent-guardrails-reference#refusing-a-read).

## Where the files live

One dot-directory, organized inside, the way `.github/` and `.vscode/` are:

```
.otari/
  guardrails.yml
  verifiers/
    no-conflict-markers.sh
```

A repository with more concerns than one file holds keeps a directory instead:

```
.otari/
  guardrails/
    secrets.yml
    generated-artifacts.yml
    tests.yml
    architecture/
      layer-rules.yml
  verifiers/
    check-architecture.sh
    no-conflict-markers.sh
    no-stranded-docblocks.py
```

`otari hook` reads `.otari/guardrails.yml` and every `.yml` and `.yaml` under
`.otari/guardrails/`, nested ones included, and composes them into one
guardrail. Both shapes may coexist, which is what a repository looks like
partway through splitting one file into several.

Start with the single file `otari hook setup` scaffolds. Splitting it later
moves gates between files rather than migrating them. When you split, group by
concern rather than by gate type: a file is the unit someone shares or lifts
out of another repository. The rules a set of files must meet are under
[Composing several files](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/reference/agent-guardrails/agent-guardrails-reference#composing-several-files).

Keep verifier scripts in `.otari/verifiers/`, a sibling of both shapes, so no
non-guardrail file sits under the scanned directory. A `verifier` gate names
its script by repo-relative path, so this is a convention, not a requirement.

## Checking a guardrail before it runs

A guardrail is otherwise only checked when it runs, where a gate that silently
does not match looks the same as a clean result.

1. Check the composed guardrail for errors and warnings:

   ```bash
   otari guardrails validate
   ```

   An error is a gate that cannot do its job whatever the session does; a
   warning is a gate that may not mean what you intended. Each finding is
   listed under
   [Validation errors and warnings](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/reference/agent-guardrails/agent-guardrails-reference#validation-errors-and-warnings).

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

   Check that the gates you meant to fire say `fires`. Nothing is contacted and
   no model is called, so a `judge` gate says `would run` rather than running.
   The labels are defined under
   [Dry-run output](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/reference/agent-guardrails/agent-guardrails-reference#dry-run-output).

3. Where a warning describes a design you intend, name its code in the gate's
   [`accept_warnings`](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/reference/agent-guardrails/agent-guardrails-reference#accept_warnings-on-any-gate).

4. Make warnings fail too, which is what a CI invocation wants:

   ```bash
   otari guardrails validate --strict
   ```

   If the hook runs with a raised judge gate cap, give `validate` the same
   `--max-judges` value, so it warns about the same gates the hook skips.

## Checking a pull request in CI

`otari hook` runs only on a machine that registered it, so a pull request from
a contributor or an agent without the hook skips every gate.
`otari guardrails check` runs the gates against a change instead:

```bash
otari guardrails check --base origin/main
```

That checks the change from where the branch left `origin/main` to the working
tree, untracked files included. Without `--base` it checks the uncommitted
change, the way a `Stop` event does. Which gate types can run against a change,
and how the exit status is decided, is under
[`otari guardrails check`](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/reference/agent-guardrails/agent-guardrails-reference#otari-guardrails-check).

To run it on every pull request:

1. Take the guardrail files and verifier scripts from the base commit rather
   than the pull request, so a change cannot alter the gates that check it:
   check out the base commit beside the pull request and pass
   `--guardrails-from <base checkout>`.
2. Run the `path` and `verifier` gates in the pull request's tree, with
   `--type path --type verifier`. A verifier that reads `git status` sees only
   uncommitted changes, so stage the whole pull request against its merge base
   first (for example `git reset --soft <merge base>`).
3. Run the `judge` gates as a separate job, with `--type judge`, since they need
   a signed-in model CLI and a credential. Keep the credential away from the
   pull request's code: run this job from the base branch's copy of the
   workflow (`pull_request_target` on GitHub), fetch the pull request as Git
   objects only, and pass `--base <base> --head <head>` so only its diff is
   read. Nothing from the pull request then runs next to the credential.
4. Make the `path` and `verifier` job a required status check, and review
   changes to that workflow and to the scripts its verifiers call: that job runs
   the pull request's own code.

Otari's own repository does this in
[`.github/workflows/otari-guardrails-check.yml`](https://github.com/mozilla-ai/otari/blob/main/.github/workflows/otari-guardrails-check.yml)
and
[`.github/workflows/otari-guardrails-judges.yml`](https://github.com/mozilla-ai/otari/blob/main/.github/workflows/otari-guardrails-judges.yml).

## Reporting to Otari

`otari hook` can report every gate it checks to an Otari server, so a team sees
which gates ran, which blocked, and which never fire. Setting the server and an
API key where the agent runs turns it on, for every repository the agent works
in:

```bash
export OTARI_URL="http://localhost:8000"     # or your Otari deployment
export OTARI_API_KEY="tk-your-import-key"    # budget-exempt, as for an OTLP export
```

The server receives each gate's definition, its `message` included, and that
covers your own gates in `~/.otari/`, so set `OTARI_URL` only for a server you
would show your guardrails to. A session then shows each gate's message where it
blocked, and lists the gates the guardrail declares that never ran. When the
agent also exports its own spans, as
[Claude Code can](use-with-claude-code.md#send-claude-code-sessions-to-otari), a
gate shows under the tool call it checked, and a `Stop` gate under the turn it
held. Claude Code does not pass `OTEL_*` variables to its hooks, so it needs
`OTARI_URL` and `OTARI_API_KEY` set in the `env` block of its settings, as that
page shows.

Reporting never slows a check or changes its verdict. To also send the command
or paths a check read, set `OTARI_HOOK_REPORT_CONTENT=1`; a server keeps them
only where it keeps content (see
[Keeping what a span was about](traces.md#keeping-what-a-span-was-about)).
`OTARI_HOOK_REPORT=off` turns reporting off. What a report carries, how it is
queued and sent, and what a judge gate's usage costs are in
[What `otari hook` reports](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/reference/agent-guardrails/agent-guardrails-reference#what-otari-hook-reports).

## Generating gates from AGENTS.md or CLAUDE.md

`otari guardrails generate` proposes gates from a repository's own AGENTS.md, or
CLAUDE.md where that is the only doc:

```bash
otari guardrails generate
```

1. Confirm the model CLI it names (`claude -p` or `codex exec`) before it sends
   the doc anywhere.
2. Walk the proposals one at a time, answering `y`, `n`, `e` (edit) or `q`.
   Only accepted gates are written, each parsed first, so an invalid field is
   reported in the terminal.
3. Run `otari guardrails validate` on the result.

Rerunning it after AGENTS.md changes proposes from scratch and asks about every
candidate again, including ones a prior run declined. Where it writes, and its
options, are under
[`otari guardrails generate`](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/reference/agent-guardrails/agent-guardrails-reference#otari-guardrails-generate).

## Your own guardrail in `~/.otari/`

Some rules belong to a person, not to a repository: no `git reset --hard`, no
reading `.env` files, run the linter before stopping. Put them in your home
directory, in the same two shapes a repository uses:

```
~/.otari/
  guardrails.yml
  guardrails/
    git-safety.yml
  verifiers/
    lint-before-stop.sh
```

`otari hook` composes these with the repository's own files. Your gates apply in
every repository where the hook runs, nobody else gets them, and a repository
cannot turn them off. Their IDs gain a `user:` prefix; how the two sets compose,
and what happens when one side fails to load, is under
[Composing several files](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/reference/agent-guardrails/agent-guardrails-reference#composing-several-files).

1. Write the gates as you would in a repository. A `verifier` gate names its
   script relative to your home directory, for example
   `verifier: .otari/verifiers/lint-before-stop.sh`, and the script must be
   inside `~/.otari/verifiers/`; see
   [`verifier`](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/reference/agent-guardrails/agent-guardrails-reference#verifier) for the ownership rule
   it must meet.
2. Run `otari guardrails validate` in a repository. It checks your files with
   the repository's and lists which gates come from where. Use `--repo-only`
   to check a repository's rules without yours, so a warning in one of your
   files cannot fail `--strict`.

`otari guardrails check` leaves your files out, so a CI check reads the same on
every machine.

### Running your guardrail in every repository

`otari hook setup` registers the hook in one repository only, and does not
write a user-level entry. To make your gates apply in every repository:

1. Add the entries `setup` writes (shown under
   [Registering it by hand](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/reference/agent-guardrails/agent-guardrails-reference#registering-it-by-hand))
   to `~/.claude/settings.json`, with the widest `PreToolUse` matcher,
   `Edit|Write|NotebookEdit|Read|Bash`, because one entry now serves the gates
   of every repository.
2. Register the hook in one place only. Remove the per-repository entry from
   `.claude/settings.local.json`, or skip the user-level one on that machine;
   with both, the hook can run twice for each event.

## Recommendations

- **Name both moments in a write rule.** A `path` gate naming only
  `pre_tool_use.edit_target` enforces less than it appears to.
- **Start a new rule `advisory`**, and promote it to `required` once it has run
  without a false positive. A false positive on a `required` gate blocks real
  work.
- **Write `message` for the agent.** It is the only part of the gate the agent
  sees, and it determines whether the next attempt is a fix or a repeat. State
  what to do instead, and name the document the rule came from.
- **Name a real invocation in a `forbidden` phrase**: `["npm install"]` rather
  than `["npm"]`, which also refuses `grep -rn npm web/`.
- **Give each required command its own `command_if_changed` gate**, because any
  one entry in `require` satisfies the gate.
- **Scope a rule over a generated file with care.** The command that regenerates
  the file writes it too, and a `stop.working_tree` gate cannot tell the two
  apart. Express the rule as `command_if_changed` ("if this changed, the
  generator must have run"), or accept that the regeneration turn reports the
  gate.
- **Use `verifier` for what the mechanical types cannot express**, and keep the
  script read-only, since verifiers run concurrently against one working tree.
- **Keep `judge` gates scoped.** Each is one model call per applicable `Stop`
  event. Scope each with `when_changed` so a session that touched nothing
  relevant does not pay for it, and give one that must run a `priority`. A cap
  limits how many run per event, and raising it does not raise the time they
  share, so a high cap suits fast judges; see
  [`judge`](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/reference/agent-guardrails/agent-guardrails-reference#judge). Set
  `OTARI_HOOK_JUDGE_DRY_RUN` to see which calls would be made without making
  them.
- **Run `otari guardrails validate --strict` and `otari guardrails check` in
  CI**, as described above. A gate that has stopped matching reports the same as
  a gate that passed.
- **Keep one file until the concerns can be named**, then split by concern.

## How Agent Guardrails work, and their limits

### Why gates rather than written rules

A rule in AGENTS.md is read as context and competes with everything else in the
window. A gate is evaluated: it either matches the evidence or it does not, and
a `required` one that matches refuses the call or blocks the turn.

A gate reads what happened rather than what was reported. Its evidence is the
path a tool call declares, the command a tool call declares, and
`git status --porcelain` once the turn is over. Where `otari hook` reads the
session transcript, it reads the tool calls recorded in it, not the agent's
account of the session.

A `PreToolUse` gate runs before the action, so a refusal leaves nothing to
undo. CI reports a force-push after the force-push.

Guardrails are committed, so they clone, review, and diff like the rest of the
repository, and both supported harnesses read the same files.

A `judge` gate is the exception to all of this, and is advisory for that
reason: a model's verdict is not reproducible the way a match is, and the diff
it reads is content an attacker can influence.

### Why a write rule names both moments

`pre_tool_use.edit_target` sees the path an `Edit`, `Write` or `NotebookEdit`
call declares in its own arguments, and refuses the call, so the write never
lands. It sees only tools that declare a path. A shell command declares none:

```bash
echo "## 1.2.0" >> CHANGELOG.md
sed -i '' 's/foo/bar/' CHANGELOG.md
python scripts/bump.py
```

None of those is visible to a `PreToolUse` path gate, and no evidence exists
that would make them visible. Before the command runs there is only its text,
and enumerating what an arbitrary shell line will write is not decidable. An
agent reaches the shell often, including immediately after an `Edit` call is
refused.

`stop.working_tree` covers that. At the end of the turn the same globs are
checked against `git status --porcelain`, which reports the file whatever wrote
it. It cannot prevent the write, so the turn is blocked over a file that has
already changed, and the change has to be reverted.

A `command` gate can refuse the shell command itself, but only the spellings it
names, and nothing a script does internally. It catches mistakes by a
cooperative agent; it is not a boundary against one working around it.

### Reads have no second moment

A write the shell slips past still lands in the tree, so `stop.working_tree`
reports it. A read lands nowhere: nothing changes, `git status` has nothing to
report, and no end-of-turn evidence exists or could exist. `cat .env` is
refused by a `command` gate or not at all, which is why a secrets rule is two
gates.

### Known gaps

- There are no reusable packs. A guardrail is written in the repository it
  guards, or in `~/.otari/` for one person's own rules; sharing gates or
  verifiers across a team's repositories is not built.
- `otari hook setup` registers the hook per repository only, so running your
  own gates in every repository needs a hand-written user-level entry.
- A pull request cannot be checked against a `command` or `command_if_changed`
  gate. Both read a session's tool calls, and a change carries none.
- Nothing reports a secret that was already read. `pre_tool_use.read_target`
  refuses the `Read` tool and does nothing else; a session that reached a secret
  through the shell leaves no signal a guardrail can raise afterwards.
- A blocking `Stop` gate has a finite budget in Claude Code, and Codex's Code
  Mode bypasses `PreToolUse`; see
  [How the harness integrations collect evidence](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/reference/agent-guardrails/agent-guardrails-reference#how-the-harness-integrations-collect-evidence)
  and [Registering it for Codex](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/reference/agent-guardrails/agent-guardrails-reference#registering-it-for-codex).
- A verifier from a branch you check out to review runs on the next `Stop`
  event without you deciding to run it; see
  [`verifier`](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/reference/agent-guardrails/agent-guardrails-reference#verifier).
