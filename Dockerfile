# The admin dashboard bundle is not committed (see .gitignore), so build it here.
#
# Pinned to BUILDPLATFORM: otari-docker.yml publishes linux/amd64 and linux/arm64,
# so without this the whole install + vite build runs a second time under QEMU for
# the non-native arch on every publish. The output is JavaScript, CSS, and PNGs,
# which are architecture-independent, so one native build serves both images.
FROM --platform=$BUILDPLATFORM node:26-slim AS web

WORKDIR /app/web

# The dashboard is a pnpm project. Corepack reads the `packageManager` field in
# web/package.json, so the image installs with exactly the pnpm the lockfile was
# written by, and the version is pinned in one place rather than here as well.
# Node 26 no longer bundles corepack, hence the install. The prompt would block
# a non-interactive build.
ENV COREPACK_ENABLE_DOWNLOAD_PROMPT=0
RUN npm install --global corepack@latest && corepack enable

# Install from the lockfile alone, so the dependency layer is reused whenever
# only dashboard sources changed. pnpm-workspace.yaml carries the build-script
# approvals (esbuild), without which the install refuses to link Vite's binary.
COPY web/package.json web/pnpm-lock.yaml web/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile

COPY web ./
# DocsPage.tsx imports the operator guide with Vite's `?raw`, so the build reads
# it from outside web/. .dockerignore excludes docs/ but re-includes this file.
COPY docs/dashboard.md /app/docs/dashboard.md

# web/vite.config.ts writes to ../src/gateway/static/dashboard, so the bundle
# lands at /app/src/gateway/static/dashboard for the runtime stage to copy.
RUN pnpm run build

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
COPY --from=web --chown=otari:otari /app/src/gateway/static/dashboard ./src/gateway/static/dashboard
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
