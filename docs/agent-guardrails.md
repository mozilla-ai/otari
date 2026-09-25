# Agent Guardrails

Agent Guardrails are rules a repository keeps about what a coding agent may do
to its working tree, checked while the agent works. A forbidden tool call is
refused before it runs; a turn that ends in a forbidden state is blocked before
the agent hands it back.

The rules live in `.otari/`, committed with the code. `otari hook` runs them: a
coding agent's hook system invokes it, it collects the evidence that event
carries, and it evaluates the rules in process. It needs no gateway, no
account, and no credential.

This is not [Inference Guardrails](guardrails.md), which check request input and
output at inference time. A gate checks agent actions and repository diffs.

Contents:

- [Install](#install)
- [Registering the hook](#registering-the-hook)
- [Writing a gate](#writing-a-gate)
- [Guardrails and gates](#guardrails-and-gates)
- [Where the files live](#where-the-files-live)
- [Checking a guardrail before it runs](#checking-a-guardrail-before-it-runs)
- [Generating gates from AGENTS.md or CLAUDE.md](#generating-gates-from-agentsmd-or-claudemd)
- [Recommendations](#recommendations)
- [Why gates rather than written rules](#why-gates-rather-than-written-rules)
- [Status and known gaps](#status-and-known-gaps)

Every field and gate type is specified in the
[Agent Guardrails reference](agent-guardrails-reference.md).

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
[cli/README.md](../cli/README.md) for the rest of that distribution.

Four commands apply here:

| Command | What it does |
| --- | --- |
| `otari hook setup` | Registers the callback in a supported agent's own settings file. |
| `otari hook` | The callback itself. The harness invokes it; it is not run by hand. |
| `otari guardrails validate` | Checks the composed guardrail offline, and dry-runs it against a command or path. |
| `otari guardrails generate` | Proposes gates from the repository's own AGENTS.md or CLAUDE.md. |

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
`otari hook --harness claude-code`. Claude Code passes its own
`hook_event_name` in the payload, so one callback serves both events. Restart
Claude Code if a gate does not fire in the session `setup` ran in.

When the repository has no guardrail yet, `setup` offers to write a starter
`.otari/guardrails.yml` first, so there is something to check rather than a hook
that always passes. The starter carries one `command` gate refusing
`git push --force` and one `path` gate keeping `.env` out of the tree, both
`advisory`.

Re-running `setup` updates both entries in place rather than adding duplicates,
and leaves every other hook and permission in the file untouched. Re-run it when
the guardrail gains its first `command` gate or its first gate naming
`pre_tool_use.read_target`: the `PreToolUse` matcher is built from what the
guardrail asks for, and those are the two tools it has to add. The edit tools
are always in the matcher, so an ordinary `path` gate needs no re-run.

### Codex

```bash
otari hook setup --harness codex
```

The same pair of entries, written to `.codex/hooks.json` and naming
`otari hook --harness codex`.

Codex does not run a project's hooks until they are accepted. Start Codex in the
repository and run `/hooks`, which lists what the project declares and asks for
approval. Until then the file exists and nothing fires. This is per developer
and per repository: a teammate who clones the repository and runs
`otari hook setup --harness codex` accepts them separately.

The Codex integration is newer and less exercised than the Claude Code one. A
session running through Codex's own Code Mode receives no `PreToolUse` dispatch
for a shell or `apply_patch` call wrapped in JS (openai/codex#23411, open
upstream), so only the `Stop` event reaches it.

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

### Why a write rule names both moments

The two `runs` entries above cover two different moments, and a write rule
generally needs both.

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
already changed and the change has to be reverted.

`otari hook setup` registers the `Stop` hook unconditionally, so nothing further
is installed for this; the gate itself has to name `stop.working_tree`.
`otari guardrails validate` warns about a `path` gate that names only
`pre_tool_use.edit_target`.

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

That gate names the tool, not the file, because a phrase matches a contiguous
run of tokens. `sed -i '' 's/x/y/' CHANGELOG.md` puts three tokens between `-i`
and the filename, so a `"sed -i CHANGELOG.md"` phrase never fires. For the same
reason `head -5 .env` is not matched by a `head .env` phrase.

A `command` gate catches the spellings it names and nothing a script does
internally. It is a footgun-catcher for a cooperative agent, not a boundary
against one working around it. The matching rules and a measured table of what
matches are under
[`command`](agent-guardrails-reference.md#command).

### Reads have no second moment

A write the shell slips past still lands in the tree, so `stop.working_tree`
reports it. A read lands nowhere: nothing changes, `git status` has nothing to
report, and no `Stop` evidence exists or could exist. `cat .env` is refused by a
`command` gate or not at all.

A rule about secrets is therefore two gates: a `path` gate running at
`pre_tool_use.read_target`, which refuses the `Read` tool, and a `command` gate
for the shell.

## Guardrails and gates

A guardrail is the intent ("do not hand-edit the changelog"). A gate is one
check that enforces it. A guardrail file names the intent under `policy.id` and
lists its gates under `gates`.

Every gate declares an `id` unique across the whole guardrail, a `type`, an
`enforcement`, a `runs` list, and a `message` shown when it fails.

`enforcement` is `required` or `advisory`. A `required` gate blocks: the tool
call is refused, or the turn does not end. An `advisory` gate warns and the work
continues. A `judge` gate's warning also reaches the agent, not only you, since
`advisory` is the only enforcement it is allowed; see
[Who hears an advisory finding](agent-guardrails-reference.md#who-hears-an-advisory-finding).

`runs` says when the gate fires and what it can see there. The two moments a
write rule uses are covered in
[Why a write rule names both moments](#why-a-write-rule-names-both-moments);
every legal value is specified in
[When a gate runs](agent-guardrails-reference.md#when-a-gate-runs-runs).

Five gate types are available:

| Type | Fails when | Example rule |
| --- | --- | --- |
| `path` | a path in scope matches a forbidden glob | nothing hand-edits a generated file |
| `command` | a command in scope matches a forbidden phrase | no `git push --force` |
| `command_if_changed` | a path changed but a required command never ran | a changed spec must have been followed by its generator |
| `judge` | a model reading the diff reports the rubric was not met | the change follows the repository's error-handling conventions |
| `verifier` | a script in the repository exits non-zero | anything the four above cannot express |

The first three are mechanical: a glob or phrase match, no model and no
subprocess. `verifier` runs a script the repository supplies, and its exit code
is as reproducible as a glob, so it may be `required`. `judge` is the only type
that calls a model, and its `enforcement` must be `advisory`: a model verdict is
not reproducible, and the diff it reads is attacker-influenceable content.

Each type's fields are specified under
[Gate types](agent-guardrails-reference.md#gate-types).

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
    no-conflict-markers.sh
    no-stranded-docblocks.py
```

`otari hook` reads `.otari/guardrails.yml` and every `.yml` and `.yaml` under
`.otari/guardrails/`, nested ones included, and composes them into one
guardrail. Both shapes may coexist, which is what a repository looks like
partway through splitting one file into several.

`otari hook setup` scaffolds the single file, as above. Splitting it later moves gates
between files rather than migrating them, so there is no cost to starting there.
Group a directory by concern rather than by gate type: a file is the unit
someone shares or lifts out of another repository. Otari's own repository keeps
eight.

Every file parses on its own, a gate id is unique across the whole set, and one
unparseable file yields no guardrail rather than a partial one. The rules that
apply once there is more than one file are under
[Composing several files](agent-guardrails-reference.md#composing-several-files).

`.otari/verifiers/` is a sibling of both shapes rather than a child of either,
so no non-guardrail file sits under the scanned directory. A `verifier` gate
names its script by repo-relative path, so any path works; this is the
convention, not a requirement.

A repository still carrying `.otari-guardrails.yml` at its root is enforcing no
gates: nothing reads that path. Move the file to `.otari/guardrails.yml`
unchanged. `otari hook` reports this on every event until it moves.

## Checking a guardrail before it runs

A guardrail is otherwise only checked when it runs, where a gate that silently
does not match looks the same as a clean result.

```bash
otari guardrails validate
otari guardrails validate --path CHANGELOG.md --command "make postman"
otari guardrails validate --strict
```

With no arguments it composes everything the hook composes and reports two kinds
of finding. An error is a gate that cannot do its job whatever the session does:
a missing verifier script, a glob that matches nothing, a phrase carrying a
shell separator no segment can contain. A warning is a gate that runs and may
not mean what its author intended: a `**` glob that skips the shallow case, a
one-token `forbidden` phrase, a `path` gate naming only
`pre_tool_use.edit_target`, more `judge` or `verifier` gates than one `Stop`
event evaluates. `--strict` exits non-zero on a warning as well, which is what a
CI invocation wants.

`--path` and `--command` dry-run the guardrail against evidence supplied on the
command line, at every moment a real session would offer it:

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

Nothing is contacted and no model is called, so a `judge` gate reports
`would run` rather than running. The full output is documented under
[Checking a guardrail before it runs](agent-guardrails-reference.md#checking-a-guardrail-before-it-runs).

## Generating gates from AGENTS.md or CLAUDE.md

`otari guardrails generate` proposes gates from a repository's own AGENTS.md, or
CLAUDE.md where that is the only doc:

```bash
otari guardrails generate
```

It resolves a locally installed model CLI (`claude -p` or `codex exec`), names
it and asks for confirmation before sending the doc anywhere, asks it to propose
gates for the rules that look mechanically checkable, then walks the proposals
one at a time: accept, reject, edit, or quit. Nothing is written for a skipped or
declined proposal, and every accepted gate is parsed before it is appended, so an
invalid field is reported in the terminal rather than the first time the hook
runs.

It is a one-shot proposal tool rather than a sync: rerunning it after AGENTS.md
changes proposes from scratch and asks about every candidate again, including
ones a prior run declined. Flags are documented under
[Generating gates from AGENTS.md/CLAUDE.md](agent-guardrails-reference.md#generating-gates-from-agentsmdclaudemd).

## Recommendations

A write rule names both moments. A gate that names only
`pre_tool_use.edit_target` enforces less than it appears to; see
[above](#why-a-write-rule-names-both-moments).

Start a new rule `advisory` and promote it to `required` once it has run without
a false positive. A false positive on a `required` gate blocks real work.

Write `message` for the agent. It is the only part of the gate the agent sees,
and it determines whether the next attempt is a fix or a repeat. State what to
do instead, and name the document the rule came from.

Name a real invocation in a `forbidden` phrase. `["npm install"]` rather than
`["npm"]`: a one-token phrase matches that token in any position, so a bare
`npm` also refuses `grep -rn npm web/`.

Give each required command its own `command_if_changed` gate. Any one entry in
`require` satisfies the gate, so `require: ["make a", "make b"]` passes once
`make a` has run.

Scope a rule over a generated file with care. The command that regenerates the
file writes it too, and a `stop.working_tree` gate cannot distinguish the two.
Either express the rule as `command_if_changed` ("if this changed, the generator
must have run") or accept that the regeneration turn reports the gate.

Use `verifier` for what the mechanical types cannot express. The script runs with
the repository root as its working directory, exits 0, 1 or anything else for
pass, fail and error, and may be `required`. It carries the same trust as a
Makefile target or a pre-commit hook. Keep it read-only: applicable verifiers run
concurrently against one working tree.

Keep `judge` gates few and scoped. Each costs one model call per applicable
`Stop` event, at most five run per event, and all of them are advisory. Give a
gate that must run a `priority`, and scope it with `when_changed` so a session
that touched nothing relevant does not pay for it. Setting
`OTARI_HOOK_JUDGE_DRY_RUN` in the harness's environment reports which calls would
have been made, and their approximate size, without making them.

Run `otari guardrails validate --strict` in CI. A gate that has stopped matching
reports the same as a gate that passed.

Keep one file until the concerns can be named, then split by concern.

## Why gates rather than written rules

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

## Status and known gaps

Five gate types are available, both harness integrations are installed commands,
and there are no reusable packs: a guardrail is written in the repository it
guards, and a `verifier` script is always a path inside that repository. Sharing
gates or verifiers across repositories is deferred.

A blocking `Stop` gate has a finite budget. Claude Code overrides a `Stop` hook
after eight consecutive blocks without progress and ends the turn with a warning
(`CLAUDE_CODE_STOP_HOOK_BLOCK_CAP` raises the limit). `otari hook` continues to
block rather than standing down, and reports that the budget is running out, so
the remaining attempts go to fixing the gate or explaining why it cannot be
fixed.

Nothing reports a secret that was already read. `pre_tool_use.read_target`
refuses the `Read` tool and does nothing else; a session that reached a secret
through the shell leaves no signal a guardrail can raise afterwards. That is
incident response rather than enforcement, and is not built.

Codex carries two further limitations; see [Codex](#codex) above.
