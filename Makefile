.PHONY: help dev test test-unit test-integration lint typecheck openapi-check

help:
	@printf "Available targets:\n"
	@printf "  dev  Run gateway with uvicorn --reload using .env\n"
	@printf "  test Run full test suite (unit + integration)\n"
	@printf "  test-unit Run unit tests\n"
	@printf "  test-integration Run integration tests\n"
	@printf "  lint Run Ruff lint checks\n"
	@printf "  typecheck Run mypy type checks\n"
	@printf "  openapi-check Verify the OpenAPI spec is up to date\n"

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
