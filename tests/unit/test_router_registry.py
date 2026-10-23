"""Which core routers the gateway mounts.

A router moving in or out changes what the product serves, so the paths are named
here rather than left to be discovered.
"""

from fastapi import APIRouter
from fastapi.routing import iter_route_contexts

from gateway.api import main as api_main
from gateway.core.config import GatewayConfig

# Inference paths that read management state too.
_INFERENCE_READING_MANAGEMENT = {
    "/batches",
    "/decisions",
    "/files",
    "/routing/recommend",
    "/search",
    "/systemone",
}

# The pass-through paths, served with the deployment's own provider credentials.
_PASSTHROUGH = {
    "/audio/speech",
    "/audio/transcriptions",
    "/embeddings",
    "/images/generations",
    "/moderations",
    "/rerank",
}


def _served(config: GatewayConfig) -> set[str]:
    """The paths ``config``'s deployment mounts, without path parameters."""
    api = APIRouter()
    api_main._register_core_routers(api, config, ())
    served = set()
    for route in iter_route_contexts(api.routes):
        path = route.path
        if path is not None:
            served.add(path.split("{")[0].rstrip("/") or path)
    return served


def test_the_gateway_serves_inference_and_management() -> None:
    served = _served(GatewayConfig(require_pricing=False))

    assert "/chat/completions" in served
    assert "/messages" in served
    assert "/keys" in served
    assert _INFERENCE_READING_MANAGEMENT | _PASSTHROUGH <= served


def test_the_passkey_routes_are_absent_where_passkeys_are_off() -> None:
    assert any(path.startswith("/auth/webauthn") for path in _served(GatewayConfig(require_pricing=False)))
    assert not any(
        path.startswith("/auth/webauthn")
        for path in _served(GatewayConfig(require_pricing=False, passkeys_enabled=False))
    )
