#!/usr/bin/env bash
#
# Liveness check for the gateway Docker image.
# Starts PostgreSQL, runs the gateway container, and validates health endpoints.
#
# Usage: ./scripts/docker_liveness_check.sh <image_tag>
# Example: ./scripts/docker_liveness_check.sh otari-test:abc123
#

set -euo pipefail

IMAGE_TAG="${1:-}"

if [[ -z "$IMAGE_TAG" ]]; then
    echo "Usage: $0 <image_tag>"
    echo "Example: $0 otari-test:latest"
    exit 1
fi

POSTGRES_CONTAINER="test-postgres"
OTARI_CONTAINER="test-otari"
CONFIG_FILE="/tmp/otari-config.yml"

cleanup() {
    echo "Cleaning up containers..."
    docker rm -f "$OTARI_CONTAINER" "$POSTGRES_CONTAINER" 2>/dev/null || true
}

trap cleanup EXIT

echo "Starting PostgreSQL container..."
docker run -d --name "$POSTGRES_CONTAINER" \
    -e POSTGRES_USER=otari \
    -e POSTGRES_PASSWORD=otari \
    -e POSTGRES_DB=otari \
    -p 5432:5432 \
    postgres:16-alpine

echo "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if docker exec "$POSTGRES_CONTAINER" pg_isready -U otari > /dev/null 2>&1; then
        echo "PostgreSQL is ready"
        break
    fi
    if [[ $i -eq 30 ]]; then
        echo "PostgreSQL failed to become ready"
        docker logs "$POSTGRES_CONTAINER"
        exit 1
    fi
    sleep 1
done

echo "Creating gateway config..."
cat > "$CONFIG_FILE" << 'EOF'
database_url: "postgresql://otari:otari@host.docker.internal:5432/otari"
host: "0.0.0.0"
port: 8000
master_key: "test-master-key"
EOF

echo "Starting gateway container with image: $IMAGE_TAG"
docker run -d --name "$OTARI_CONTAINER" \
    --add-host=host.docker.internal:host-gateway \
    -p 8000:8000 \
    -v "$CONFIG_FILE":/app/config.yml \
    "$IMAGE_TAG" \
    otari serve --config /app/config.yml

echo "Waiting for gateway to be healthy..."
for i in {1..60}; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "Gateway is responding"
        break
    fi
    if [[ $i -eq 60 ]]; then
        echo "Gateway failed to become healthy within 60 seconds"
        echo "=== Gateway logs ==="
        docker logs "$OTARI_CONTAINER"
        exit 1
    fi
    sleep 1
done

echo "Testing /health endpoint..."
curl -sf http://localhost:8000/health
echo ""

echo "Testing /health/liveness endpoint..."
curl -sf http://localhost:8000/health/liveness
echo ""

echo "Testing /health/readiness endpoint..."
curl -sf http://localhost:8000/health/readiness
echo ""

# The dashboard bundle is not committed: the Dockerfile's web stage builds it and
# the runtime stage copies it in. A bundle that never reached the image is not an
# error at boot, it silently degrades to the get-started tutorial at "/", so
# assert on the page the browser would actually get.
# Captured rather than piped into grep: with pipefail, grep -q closing the pipe
# early can surface curl's SIGPIPE as a failure of the check itself.
echo "Testing the built dashboard is served at /..."
ROOT_PAGE="$(curl -sf http://localhost:8000/ || true)"
if ! grep -q "<title>Otari Dashboard</title>" <<< "$ROOT_PAGE"; then
    echo "The root path did not serve the admin dashboard."
    echo "The image builds the bundle in its web stage, so check that stage and the"
    echo "COPY --from=web in the runtime stage. Serving the get-started tutorial"
    echo "instead means the bundle never reached the image."
    echo "=== Page served at / ==="
    head -40 <<< "$ROOT_PAGE"
    echo "=== Gateway logs ==="
    docker logs "$OTARI_CONTAINER"
    exit 1
fi

echo "All liveness checks passed!"
