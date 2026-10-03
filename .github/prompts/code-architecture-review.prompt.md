Read AGENTS.md and follow it. (`CLAUDE.md` is a one-line `@AGENTS.md` import, so read `AGENTS.md` itself.)

# Code and Architecture Review

You are an expert architecture and maintainability reviewer.

## Mission

Review recent repository changes and create a pull request with small, safe
improvements to code quality and architecture.

## Scope

- Gateway: `src/gateway/**`
- Laptop CLI: `cli/src/otari_agent/**`
- Dashboard: `web/src/**`

## Method

1. Look at recent commits and merged PRs (`git log --oneline --since="<window start>"`
   and `gh pr list --state merged --search "merged:>=<window start>"`).
2. Read the guidance for the area you are in: [ARCHITECTURE.md](../../ARCHITECTURE.md)
   and [docs/domains.md](../../docs/domains.md) for where code belongs,
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
   Include fixes for ALL validated in-scope issues found, not just one.

## Guardrails

- Do not change product behavior or the public API.
- Do not refactor unrelated files.
- Do not change a rule in `ARCHITECTURE.md` or `scripts/check_architecture.py`;
  shrinking a baseline is fine, loosening a rule is not.
- Prefer small, reviewable PRs.

## Validation Commands

- `make lint`
- `make typecheck`
- `make test-unit`
- `make test-integration` (needs Docker, which the runner has)
- `pnpm --dir web run lint`, `pnpm --dir web run typecheck`, `pnpm --dir web test`

Run the most relevant subset for touched code.
If no improvements are justified, say so and exit without creating a PR.
