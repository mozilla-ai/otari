FROM python:3.14-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip uv

COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.14-slim AS runtime

WORKDIR /app

RUN useradd -m -u 1000 otari && chown otari:otari /app

COPY --from=builder --chown=otari:otari /app/.venv /app/.venv
COPY --chown=otari:otari src ./src
COPY --chown=otari:otari alembic ./alembic
COPY --chown=otari:otari alembic.ini ./alembic.ini

USER otari

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ARG OTARI_VERSION=0.0.0-dev
ENV OTARI_VERSION=${OTARI_VERSION}
ENV OTARI_HOST=0.0.0.0
ENV OTARI_PORT=8000

CMD ["otari", "serve"]
