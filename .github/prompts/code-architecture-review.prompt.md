Read AGENTS.md and follow it. (`CLAUDE.md` is a one-line `@AGENTS.md` import, so read `AGENTS.md` itself.)

# Code and Architecture Review

You are an expert architecture and maintainability reviewer.

## Mission

Review recent repository changes and create a pull request with small, safe
improvements to code quality and architecture.

## Scope

Only files changed since `<window start>` in these areas, and the code they
call into directly:

- Gateway: `src/gateway/**`
- Laptop CLI: `cli/src/otari_agent/**`
- Dashboard: `web/src/**`

List them with `git diff --name-only $(git rev-list -1 --before="<window start>" main) main -- src/gateway/ cli/src/otari_agent/ web/src/`,
then find their direct callees from those files.
Do not survey the rest of the tree, and stop once you have at most five
fixes: the job is cancelled after 40 minutes, and opening the PR is the last
step, so leave time for validation and the PR.

## Method

1. Look at recent commits and merged PRs (`git log --oneline --since="<window start>"`
   and `gh pr list --state merged --search "merged:>=<window start>"`).
2. Read the guidance for the area you are in: [ARCHITECTURE.md](../../ARCHITECTURE.md)
   and [DOMAINS.md](../../DOMAINS.md) for where code belongs,
   `.github/skills/backend-standards/SKILL.md` for the gateway and
   `.github/skills/frontend-standards/SKILL.md` for the dashboard.
3. Identify concrete improvement opportunities:
   - over-complex logic
   - duplicated patterns
   - weak boundaries between layers, including code that sits on one of
     `scripts/check_architecture.py`'s baselines and can now come off it
   - naming and structure clarity problems
4. Apply only targeted refactors that preserve behavior.
5. Validate with the commands below.
6. Create a PR as [automation-pr.prompt.md](automation-pr.prompt.md) describes.
   Include every validated fix you made, up to the five the scope allows.

## Guardrails

- Do not change product behavior or the public API.
- Do not refactor unrelated files.
- Do not change a rule in `ARCHITECTURE.md` or `scripts/check_architecture.py`;
  shrinking a baseline is fine, loosening a rule is not.
- Prefer small, reviewable PRs.

## Validation Commands

- `make lint` and `make typecheck`
- `uv run pytest` on the unit and integration test files that cover the
  modules you touched, never the whole of either suite. Integration tests need
  Docker, which the runner has. The rest of the suite runs on the PR in CI.
- For dashboard changes: `pnpm --dir web run lint`, `pnpm --dir web run typecheck`,
  and `pnpm --dir web test <test files>` with the touched components' test
  files; without them it runs the whole Vitest suite.

If no improvements are justified, say so and exit without creating a PR.
