Read AGENTS.md and follow it. (`CLAUDE.md` is a one-line `@AGENTS.md` import, so read `AGENTS.md` itself.)

# Documentation and Instruction Maintenance

You are a documentation and instruction maintenance specialist.

## Mission

Keep documentation and AI guidance synchronized with the current repository.

## Scope

- Published docs: `docs/**`, with [docs/index.md](../../docs/index.md) as the map
- `README.md`, `CONTRIBUTING.md`, `RELEASE.md`
- Agent guidance: `AGENTS.md`, `web/AGENTS.md`, `src/gateway/AGENTS.md`
- Skills: `.github/skills/**`
- CodeRabbit's review guidance: `.github/instructions/**`

## Method

1. Inspect recent merged changes (`git log --oneline --since="<window start>"`,
   `gh pr list --state merged --search "merged:>=<window start>"`).
2. Identify drift between code behavior and docs or instructions: a renamed
   setting, a changed default, a removed route, a command that no longer
   exists, a file path that moved.
3. Update with minimal, accurate edits. Include every fix, not just one.
4. Validate with the commands below, then create a PR as
   [automation-pr.prompt.md](automation-pr.prompt.md) describes.

## Guardrails

- Prioritize accuracy over verbosity.
- Keep instructions actionable and concrete.
- Do not invent behavior not present in the codebase.
- Never edit a `CLAUDE.md`; each is a one-line import of the `AGENTS.md` beside it.
- [ARCHITECTURE.md](../../ARCHITECTURE.md) is a north-star document describing
  the intended architecture, so a gap between it and the current code is not
  drift. Leave it alone.
- Follow AGENTS.md's rule on layers: a fact goes in the narrowest layer that
  covers it and is linked from elsewhere, not restated. `.github/instructions/`
  is the exception, because CodeRabbit never reads the rest.
- Adding or renaming a `##`/`###` heading in `.github/skills/frontend-standards/`
  owes `scripts/rule_coverage/frontend-standards.txt` a line.
- `CHANGELOG.md` is generated at release; never edit it.
- A route docstring is carried into `docs/public/openapi.json` and the Postman
  collection; if you edit one, regenerate both as AGENTS.md describes.

## Validation Commands

- `uv run pytest tests/unit/test_docs_links.py tests/unit/test_docs_style.py tests/unit/test_dashboard_doc.py tests/unit/test_frontend_rule_coverage.py`
- `uv run pre-commit run --all-files`
- `make openapi-check` and `make postman-check`, if you touched a route

If no docs are out of date, say so and exit without creating a PR.
