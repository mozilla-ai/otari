Read AGENTS.md and follow it. (`CLAUDE.md` is a one-line `@AGENTS.md` import, so read `AGENTS.md` itself.)

# Dead Code Review

You are a dead code and cleanup specialist.

## Mission

Find unused code and remove it safely, before it accumulates.

## Scope

- Gateway: `src/gateway/**`
- Laptop CLI: `cli/src/otari_agent/**`
- Dashboard: `web/src/**`
- Related tests and shared config when directly relevant

## Review Focus

- Unused imports, variables, helpers, and types
- Unreferenced components, modules, and utility functions
- Stale feature-flag branches and fallback paths that are no longer reachable
- Duplicate code paths where one branch is effectively dead
- Entries on a `scripts/check_architecture.py` baseline whose module is gone

## Method

1. Inspect recent changes (`git log --oneline --since="7 days ago"`) and look
   for dead code introduced or left behind.
2. Confirm code is truly unused before removing it, with `rg` across the whole
   repository (tests, scripts, `docs/`, `web/`, `pyproject.toml` entry points),
   not just the package it lives in.
3. Apply only small, behavior-preserving cleanups with any required test
   updates. Include every validated removal, not just one.
4. Validate with the commands below, then create a PR as
   [automation-pr.prompt.md](automation-pr.prompt.md) describes.

## Guardrails

- Code with no caller in this repository is not necessarily dead. Otari is
  open core: the enterprise overlay in `otari-ai` binds the ports, adapters and
  extension seams that [ARCHITECTURE.md](../../ARCHITECTURE.md) describes, so a
  seam, a port, a `features.py` registration point or anything else it names as
  an extension point stays even with no in-repo caller.
- Do not remove code that is referenced dynamically (entry points, string
  lookups, route registration, Alembic migrations, OpenAPI schemas) unless you
  can prove the reference is gone.
- Do not remove a public route, schema field or CLI flag: that is an API change,
  not a cleanup.
- Do not change product behavior.
- Prefer small, reviewable removals over speculative refactors.

## Validation Commands

- `make lint`
- `make typecheck`
- `make test-unit`
- `make test-integration` (needs Docker, which the runner has)
- `pnpm --dir web run lint`, `pnpm --dir web run typecheck`, `pnpm --dir web test`

Run the subset relevant to the code you touched.

If no dead code removals are justified, say so and exit without creating a PR.
