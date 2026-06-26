.PHONY: help dev test test-unit test-integration lint typecheck openapi-check postman postman-check changelog

help:
	@printf "Available targets:\n"
	@printf "  dev  Run Otari with uvicorn --reload using .env\n"
	@printf "  test Run full test suite (unit + integration)\n"
	@printf "  test-unit Run unit tests\n"
	@printf "  test-integration Run integration tests\n"
	@printf "  lint Run Ruff lint checks\n"
	@printf "  typecheck Run mypy type checks\n"
	@printf "  openapi-check Verify the OpenAPI spec is up to date\n"
	@printf "  postman Regenerate the Postman collection from the OpenAPI spec\n"
	@printf "  postman-check Verify the Postman collection is up to date\n"
	@printf "  changelog Preview the generated CHANGELOG.md locally (git-cliff)\n"

dev:
	@set -a; \
	if [ -f .env ]; then . ./.env; fi; \
	set +a; \
	uv run --env-file .env uvicorn gateway.dev:create_dev_app --factory --app-dir src --reload --host "$${OTARI_HOST:-$${GATEWAY_HOST:-0.0.0.0}}" --port "$${OTARI_PORT:-$${GATEWAY_PORT:-8000}}" --reload-dir src

test:
	uv run pytest -v tests/unit tests/integration

test-unit:
	uv run pytest -v tests/unit

test-integration:
	uv run pytest -v tests/integration

lint:
	uv run ruff check src tests scripts

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
