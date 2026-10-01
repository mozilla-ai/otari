.PHONY: help dev dashboard test test-unit test-integration lint lint-python lint-web check-architecture typecheck typecheck-python typecheck-web openapi-check postman postman-check changelog check-migrations railway-template-check

help:
	@printf "Available targets:\n"
	@printf "  dev  Run Otari with uvicorn --reload using .env\n"
	@printf "  dashboard Build the admin dashboard into src/gateway/static/dashboard\n"
	@printf "  test Run full test suite (unit + integration)\n"
	@printf "  test-unit Run unit tests\n"
	@printf "  test-integration Run integration tests\n"
	@printf "  lint Run every linter, fixing what it can (lint-python + lint-web)\n"
	@printf "  lint-python Run the pre-commit hooks: architecture, migrations, Ruff lint and format\n"
	@printf "  lint-web Run Biome over the dashboard\n"
	@printf "  check-architecture Enforce gateway layer rules (also run by lint)\n"
	@printf "  typecheck Run every type checker (typecheck-python + typecheck-web)\n"
	@printf "  typecheck-python Run mypy\n"
	@printf "  typecheck-web Run tsc over the dashboard\n"
	@printf "  openapi-check Verify the OpenAPI spec is up to date\n"
	@printf "  postman Regenerate the Postman collection from the OpenAPI spec\n"
	@printf "  postman-check Verify the Postman collection is up to date\n"
	@printf "  changelog Preview the generated CHANGELOG.md locally (git-cliff)\n"
	@printf "  railway-template-check Compare deploy/railway with the live Railway template\n"

dev:
	@set -a; \
	if [ -f .env ]; then . ./.env; fi; \
	set +a; \
	uv run --env-file .env uvicorn gateway.dev:create_dev_app --factory --app-dir src --reload --host "$${OTARI_HOST:-0.0.0.0}" --port "$${OTARI_PORT:-8000}" --reload-dir src

# Build the dashboard bundle the gateway serves at "/". Not committed, so run
# this to see the dashboard from a source checkout (without it the gateway
# degrades to the tutorial page) and before building a wheel. The Docker image
# builds it in its own web stage and needs no local Node.
dashboard: web/node_modules/.install-stamp
	pnpm --dir web run build

# Reinstalling on every rebuild is the wrong price to pay now that the READMEs
# send developers to `make dashboard` to see the dashboard locally. Gate it on a
# stamp of our own rather than on one of pnpm's metadata files: pnpm writes both
# `.modules.yaml` and `.pnpm/lock.yaml`, but it leaves their mtimes alone when
# an install turns out to be a no-op, so either one can sit older than the
# lockfile forever and the gate would never close. Touching the stamp ourselves
# records what the rule actually cares about, which is that an install ran
# against this lockfile. `--frozen-lockfile` is the `npm ci` of pnpm: it
# installs exactly the lockfile and fails rather than rewriting it when
# package.json has moved on.
web/node_modules/.install-stamp: web/pnpm-lock.yaml web/package.json web/pnpm-workspace.yaml
	pnpm --dir web install --frozen-lockfile
	@touch $@

test:
	uv run pytest -v tests/unit tests/integration

test-unit:
	uv run pytest -v tests/unit

test-integration:
	uv run pytest -v tests/integration

# NOTE: The linters fix what they can in place, so a run can leave changes to commit.
lint: lint-python lint-web

lint-python:
	uv run pre-commit run --all-files --show-diff-on-failure

lint-web: web/node_modules/.install-stamp
	pnpm --dir web run lint:fix

# Enforce gateway layer rules. Pure stdlib; runs as part of `make lint-python` (which
# otari-lint.yml calls on every PR) and stays independently runnable.
check-architecture:
	uv run python scripts/check_architecture.py

# Refuse a branched revision graph. Cheap, and it runs before the suites so a
# rebase-away-from-correct branch says so in one line instead of failing every
# test that builds a schema.
check-migrations:
	uv run python scripts/check_alembic_heads.py

typecheck: typecheck-python typecheck-web

typecheck-python:
	uv run mypy

typecheck-web: web/node_modules/.install-stamp
	pnpm --dir web run typecheck

openapi-check:
	uv run python scripts/generate_openapi.py --check

postman:
	uv run python scripts/generate_postman.py

postman-check:
	uv run python scripts/generate_postman.py --check

# Local preview only. CHANGELOG.md is generated at release time by the
# otari-release / otari-tag-release workflows; this target is for eyeballing
# what the next release notes will look like. Set GITHUB_TOKEN to resolve PR
# and author links. Pin git-cliff so local output matches CI.
changelog:
	uvx git-cliff@2.13.1 --config cliff.toml

# Local preview of the Homebrew formula otari-homebrew.yml renders at release,
# from the committed lock and a freshly built sdist. The tap's copy is never
# edited by hand; change packaging/homebrew/otari.rb.tmpl instead.
homebrew-formula:
	rm -rf build/homebrew
	uv build --package otari-agent --sdist -o build/homebrew
# The sdist carries the version setuptools-scm derived from the checkout, so
# the name is only known after the build; the release workflow globs it too.
	uv run python scripts/homebrew_formula.py --version "$$(ls build/homebrew/otari_agent-*.tar.gz | sed -E 's|.*/otari_agent-(.*)\.tar\.gz|\1|')" --sdist "$$(ls build/homebrew/otari_agent-*.tar.gz)" --output build/homebrew/otari.rb
	@echo "Rendered build/homebrew/otari.rb"

# Reads the live template from Railway's public API, so it needs network but no token.
railway-template-check:
	python3 scripts/check_railway_template.py
