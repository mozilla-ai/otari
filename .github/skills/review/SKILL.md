---
name: review
description: Review a pull request or diff for this repository against otari's project standards. Use when asked to review a PR, inspect a diff, or prepare review feedback.
---

# PR Review Skill

## Review procedure

1. Read the project instructions first: [AGENTS.md](../../../AGENTS.md) (the repo-root
   `CLAUDE.md` is a one-line `@AGENTS.md` import, so read `AGENTS.md` itself), plus
   `CONTRIBUTING.md` and anything else under `.github/` that bears on the change.
2. Read every changed file in the PR fully, not just the diff hunks. Use the PR head
   revision for line numbers; the local tree may sit on a different branch.
3. Load the scoped guidance that matches the changed paths. This is the step most often
   skipped, and it is where this repo keeps its real review rules:
   - Any `src/gateway/**` change → [src/gateway/AGENTS.md](../../../src/gateway/AGENTS.md)
     and [backend-standards](../backend-standards/SKILL.md).
   - Any `web/**` change → [web/AGENTS.md](../../../web/AGENTS.md) and
     [frontend-standards](../frontend-standards/SKILL.md).
   - Glob-match every `applyTo` pattern in [.github/instructions/](../../instructions/)
     against the changed files and apply each file that matches:
     [security-review](../../instructions/security-review.instructions.md),
     [performance-review](../../instructions/performance-review.instructions.md),
     [frontend-standards](../../instructions/frontend-standards.instructions.md).
     CodeRabbit loads all three as review guidance, through the directory glob in
     `.coderabbit.yaml` rather than through their `applyTo` frontmatter (its
     `path_instructions` is empty). No bot reviewing here reads that frontmatter now
     that Copilot is gone, so it is a note to a human reader and to you: glob it
     yourself and read the files that match.
4. Check the repo-specific gates below.
5. Draft the review, then re-read the draft and drop anything that is not actionable.
6. Ask whether to post. Never post without a go-ahead for that specific PR.
7. When posting, attribute the review to the agent that wrote it.

## Repo-specific gates

Check these on every review; each has broken a PR here before.

- **Generated artifacts.** Touching a route, schema, or even a route docstring makes
  `docs/public/openapi.json` and the Postman collection stale, and both must be
  regenerated and committed (`uv run python scripts/generate_openapi.py`, then
  `make postman`). The one `openapi-spec` CI job runs `make openapi-check` and
  `make postman-check`, so a PR that regenerates only the spec still fails.
- **Dashboard generated files.** The bundle (`src/gateway/static/dashboard/`) is *not*
  committed, so a `web/src` change leaves nothing to commit there. Two things under `web/`
  are committed and drift-checked, and a PR that changes them and not the artifact fails CI:
  the API client (`web/src/client/schema.ts`) and the route tree
  (`web/src/routeTree.gen.ts`). Screenshot baselines are neither committed nor checked yet,
  so a PR that moves a page owes no PNGs; what it does owe is a screenshot entry for a page
  it adds, so the page is covered when the suite becomes a gate.
- **Dashboard toolchain.** `web/` is pnpm. A PR that adds a dependency carries the
  `pnpm-lock.yaml` change; one that adds a dependency with an install script also needs an
  `allowBuilds` entry in `web/pnpm-workspace.yaml`; and a dependency HeroUI also depends on
  has to be pinned to HeroUI's exact version, or pnpm gives the two separate copies.
- **Layering.** `make lint` runs `scripts/check_architecture.py` before Ruff, so Ruff
  passing is not enough. Services must not import the API layer, repositories must not
  import services or the API layer, API routes must not import `sqlalchemy.orm`, and
  repository modules end in `_repository.py`.
- **Mode coverage.** New management routes are standalone-only; confirm `register_routers()`
  placement matches, and that hybrid mode still resolves credentials per request.
- **Error boundaries.** Public error responses must not leak provider or internal detail,
  and nothing may log secrets, tokens, or raw API keys.
- **Test hygiene.** A global `reruns` policy is out; a genuinely flaky test carries
  `@pytest.mark.flaky(reruns=...)` with a stated reason. Integration tests need PostgreSQL.
- **Prose style.** No em dashes, and no double hyphens used as separators, in README, docs,
  doc comments, commit messages, or PR descriptions. CLI flags and numeric ranges are fine.
- **PR title.** Squash-merge means the PR title is what git-cliff parses, so it must be a
  conventional commit. `CHANGELOG.md` is generated at release time and must not be
  hand-edited.

## Review expression

- Terse and actionable only. One inline comment per issue, anchored to the exact line,
  stating the problem and the concrete fix. No praise sandwich, no "looks solid overall",
  no closing verdict paragraph.
- Post inline through the GitHub reviews API with a `comments` array plus a brief summary
  body, not as one prose comment.
- Call out something positive only when it is genuinely non-obvious, never as filler.
- No blockers found → `APPROVE` with the inline findings attached, not `COMMENT`.
- Lead with correctness, security, and performance. Skip restating what the code does.

## Relationship to `/code-review`

The workflow-backed `/code-review` command auto-discovers `CLAUDE.md` files only. Because
this repo's rules sit behind `AGENTS.md` links, that workflow does not read
`.github/instructions/` or the standards skills on its own. When using it, name the
applicable paths in the command arguments; when reviewing by hand, follow step 3 above.
