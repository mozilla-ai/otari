# Opening an automation PR

How the scheduled review workflows open their pull request. The task prompt
that sent you here says what to change; this says how to propose it.

## Before you start

Run `gh pr list --label automation --state all --limit 50 --json number,title,state,headRefName,closedAt`.
These workflows run more often than the window they look at, so most of what
you find an earlier run has already seen:

- Do not redo a fix that an open automation PR already makes, and avoid a
  change that would conflict with one still waiting for review.
- Do not propose again what a closed, unmerged automation PR proposed: a
  maintainer declined it. Read its comments with `gh pr view <number>
  --comments` when you need to know why.

## The change

- Work on a new branch off `main` named `automation/<task>-<short-slug>`,
  where `<task>` is the name the workflow gave you.
- Before opening the PR, run the validation the task prompt lists for the code
  you touched, and fix or drop whatever fails.
- If a change affects a generated artifact, regenerate it as
  [AGENTS.md](../../AGENTS.md#generated-artifacts) describes and commit it.
- Commit messages follow Conventional Commits. Do not add a `Co-Authored-By`
  trailer or a "Generated with Claude Code" line.

If nothing is worth proposing, say so and exit without creating a PR.

## The pull request

Open it with `gh pr create --label automation`. If the label does not exist
yet, create it first with `gh label create automation --color ededed
--description "Opened by a scheduled automation workflow"`.

- **Title**: Conventional Commits, which `otari-pr-title.yml` enforces:
  `type(scope): summary`, all lower case, no trailing period, with the type
  taken from that workflow's list by what the change is.
- **Body**: written from `.github/pull_request_template.md` with **every section
  kept**; `pr-template-check.yml` closes a PR missing `## PR Type`,
  `## Checklist` or `## AI Usage`.
  - **Description**: what was wrong and why the change fixes it, in plain
    English for a reviewer with no context.
  - **How to test it locally**: the commands you ran and what they reported.
  - **Checklist**: tick only what is true of this run, and leave "I understand
    the code I am submitting" for the human who takes the PR over.
  - **AI Usage**: tick "This is fully AI-generated" and "I am an AI Agent
    filling out this form", and name `Claude Code (Opus), run by the <task>
    automation workflow` under **AI Model/Tool used**.
  - End the body with the run link the workflow gave you.

Write in the house style AGENTS.md sets: US English, and no em dashes or `--`
used as separators.
