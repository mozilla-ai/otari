"""Which core routers each deployment mounts.

A router moving between deployments changes what the product serves, so the
paths each deployment answers are named here rather than left to be discovered.
"""

from fastapi import APIRouter

from gateway.api import main as api_main
from gateway.core.config import GatewayConfig
from gateway.core.deployment import Deployment, Plane, RouterMount, deployment_for

# Data-plane paths that need a control plane as well. None of them has hybrid
# handling, so mounting one on a hybrid gateway would fail at request time.
_DATA_PLANE_NEEDING_CONTROL = {
    "/audio/speech",
    "/audio/transcriptions",
    "/batches",
    "/embeddings",
    "/files",
    "/images/generations",
    "/moderations",
    "/rerank",
    "/search",
}


def _served(config: GatewayConfig) -> set[str]:
    """The paths ``config``'s deployment mounts, without path parameters."""
    api = APIRouter()
    api_main._register_core_routers(api, config, ())
    served = set()
    for route in api.routes:
        path = getattr(route, "path", None)
        if path is not None:
            served.add(path.split("{")[0].rstrip("/") or path)
    return served


def _standalone() -> set[str]:
    # Named rather than left to default: an ambient OTARI_AI_TOKEN selects
    # hybrid when no mode is set, and these tests would then pass on hybrid.
    return _served(GatewayConfig(require_pricing=False, mode="standalone"))


def _hosted() -> set[str]:
    return _served(GatewayConfig(require_pricing=False, mode="hosted"))


def _hybrid() -> set[str]:
    return _served(GatewayConfig(require_pricing=False, mode="hybrid"))


def test_a_deployment_serves_the_planes_its_settings_describe() -> None:
    assert deployment_for(GatewayConfig(mode="standalone")).planes == Plane.CONTROL | Plane.DATA
    assert deployment_for(GatewayConfig(mode="hosted")).planes == Plane.CONTROL
    assert deployment_for(GatewayConfig(mode="hybrid")).planes == Plane.DATA


def test_a_router_needing_no_plane_mounts_everywhere() -> None:
    mounted = RouterMount(APIRouter())
    config = GatewayConfig()
    assert all(mounted.applies_to(Deployment(planes), config) for planes in (Plane.CONTROL, Plane.DATA, Plane(0)))


def test_a_router_mounts_only_where_every_plane_it_needs_is_served() -> None:
    both = RouterMount(APIRouter(), Plane.DATA | Plane.CONTROL)
    config = GatewayConfig()
    assert both.applies_to(Deployment(Plane.CONTROL | Plane.DATA), config)
    assert not both.applies_to(Deployment(Plane.DATA), config)
    assert not both.applies_to(Deployment(Plane.CONTROL), config)


def test_a_predicate_narrows_a_router_further_than_its_planes() -> None:
    config = GatewayConfig()
    everywhere = Deployment(Plane.CONTROL | Plane.DATA)
    assert RouterMount(APIRouter(), when=lambda _: True).applies_to(everywhere, config)
    assert not RouterMount(APIRouter(), when=lambda _: False).applies_to(everywhere, config)


def test_standalone_serves_both_planes() -> None:
    standalone = _standalone()

    assert "/chat/completions" in standalone, "the data plane"
    assert "/keys" in standalone, "the control plane"
    assert _DATA_PLANE_NEEDING_CONTROL <= standalone


def test_a_hosted_control_plane_serves_no_data_plane() -> None:
    """A completion served here would skip the usage report that debits the wallet."""
    hosted, standalone = _hosted(), _standalone()

    assert "/keys" in hosted
    assert "/chat/completions" not in hosted
    assert "/messages" not in hosted
    assert not _DATA_PLANE_NEEDING_CONTROL & hosted
    # Everything a hosted deployment drops is data plane; it keeps every
    # management route a standalone one has.
    assert standalone - hosted == _DATA_PLANE_NEEDING_CONTROL | {
        "/chat/completions",
        "/messages",
        "/messages/count_tokens",
        "/responses",
        "/mcp/execute",
        "/mcp/servers",
    }


def test_a_hybrid_gateway_serves_no_control_plane() -> None:
    """An entry that omits ``requires`` mounts everywhere.
    Only an exact set catches a management router added without one.
    """
    assert _hybrid() == {
        # Read nothing a control plane owns.
        "/bootstrap",
        "/health",
        "/health/liveness",
        "/health/readiness",
        "/hooks/check",
        # Dispatch, with the platform answering for tenancy and policy.
        "/chat/completions",
        "/mcp/execute",
        "/mcp/servers",
        "/messages",
        "/messages/count_tokens",
        "/responses",
    }
