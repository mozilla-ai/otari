.PHONY: help dev test

help:
	@printf "Available targets:\n"
	@printf "  dev  Run gateway with uvicorn --reload using .env\n"
	@printf "  test Run gateway test suite\n"

dev:
	@set -a; \
	if [ -f .env ]; then . ./.env; fi; \
	set +a; \
	uv run --env-file .env uvicorn gateway.dev:create_dev_app --factory --app-dir src --reload --host "$${GATEWAY_HOST:-0.0.0.0}" --port "$${GATEWAY_PORT:-8000}" --reload-dir src

test:
	uv run pytest -v tests/integration tests/unit/test_gateway_*.py
