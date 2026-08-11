.PHONY: help dev dashboard test test-unit test-integration lint check-architecture typecheck openapi-check postman postman-check changelog

help:
	@printf "Available targets:\n"
	@printf "  dev  Run Otari with uvicorn --reload using .env\n"
	@printf "  dashboard Build the admin dashboard into src/gateway/static/dashboard\n"
	@printf "  test Run full test suite (unit + integration)\n"
	@printf "  test-unit Run unit tests\n"
	@printf "  test-integration Run integration tests\n"
	@printf "  lint Run Ruff lint checks and the architecture check\n"
	@printf "  check-architecture Enforce gateway layer rules (also run by lint)\n"
	@printf "  typecheck Run mypy type checks\n"
	@printf "  openapi-check Verify the OpenAPI spec is up to date\n"
	@printf "  postman Regenerate the Postman collection from the OpenAPI spec\n"
	@printf "  postman-check Verify the Postman collection is up to date\n"
	@printf "  changelog Preview the generated CHANGELOG.md locally (git-cliff)\n"

dev:
	@set -a; \
	if [ -f .env ]; then . ./.env; fi; \
	set +a; \
	uv run --env-file .env uvicorn gateway.dev:create_dev_app --factory --app-dir src --reload --host "$${OTARI_HOST:-0.0.0.0}" --port "$${OTARI_PORT:-8000}" --reload-dir src

# Build the dashboard bundle the gateway serves at "/". Not committed, so run
# this to see the dashboard from a source checkout (without it the gateway
# degrades to the tutorial page) and before building a wheel. The Docker image
# builds it in its own web stage and needs no local Node.
dashboard: web/node_modules/.package-lock.json
	npm --prefix web run build

# `npm ci` deletes and reinstalls node_modules, which is the wrong price to pay on
# every rebuild now that the READMEs send developers to `make dashboard` to see the
# dashboard locally. Gate it on npm's own install stamp, which it writes inside
# node_modules: absent (or older than the lockfile) means the tree is missing or
# stale, and otherwise the install is already the one the lockfile asks for. Kept
# as `ci` rather than `install` so the lockfile is never rewritten as a side effect.
web/node_modules/.package-lock.json: web/package-lock.json
	npm --prefix web ci

test:
	uv run pytest -v tests/unit tests/integration

test-unit:
	uv run pytest -v tests/unit

test-integration:
	uv run pytest -v tests/integration

lint: check-architecture
	uv run ruff check src tests scripts

# Enforce gateway layer rules. Pure stdlib; runs as part of `make lint` (which
# otari-lint.yml calls on every PR) and stays independently runnable.
check-architecture:
	uv run python scripts/check_architecture.py

typecheck:
	uv run mypy

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
