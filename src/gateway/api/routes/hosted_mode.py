"""Data-plane stubs for a hosted control plane, which serves management only.

The mirror image of :mod:`gateway.api.routes.hybrid_mode`. There, a gateway
attached to otari.ai holds no management plane and stubs those prefixes; here, a
hosted control plane holds no data plane and stubs these. Both say the same
thing from opposite ends: this process does not host that plane, and here is the
one that does.

Hosted mode owns organizations, wallets and credentials for many tenants. The
inference itself belongs on a data-plane gateway running in hybrid mode, which
resolves this control plane's credentials per request and reports its usage
back, and that report is what debits the wallet. A request served here would
skip the report and so run unbilled, which is what otari#822 found live.

Stubbed rather than simply left unmounted for the reason the hybrid stubs are:
a bare 404 tells whoever hit it nothing, and the two audiences for one both need
more than that. A customer who pointed an SDK at the control plane host needs to
learn that the host is wrong, not that the endpoint is gone; an operator who
mis-set ``OTARI_MODE`` on a box meant to serve inference needs to learn that the
mode is why.

Naming the address is the other half of that. "Send it to your Otari gateway" is
what the caller already believed they were doing, so where the deployment knows
its data plane (``data_plane_url``, the same value ``GET /v1/bootstrap`` hands
the dashboard) the refusal says which host to use. Left unset it falls back to
the generic sentence, matching what bootstrap already treats as unconfigured.

The status stays 404 to match the hybrid stubs, because it is the same fact
being reported. It is not an authorization decision, so 403 would invite a retry
with a different key that cannot help, and Otari does implement these endpoints,
so 501 would misreport a deployment posture as a missing feature.

The refusal covers the reads and deletes, not only the calls that dispatch. A
control plane that was serving this traffic before the gate went in may already
hold files and batches, and their owners lose the API to them here rather than
keeping a read-only window onto rows the plane no longer serves. That is
deliberate: a plane is either on a deployment or it is not, and a half-mounted
one is the state nobody can reason about later. Nothing stored is deleted, so
the rows outlive the refusal and a data-plane gateway is where they belong.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from gateway.api.deps import get_config
from gateway.core.config import GatewayConfig

_GENERIC_TARGET = "your Otari gateway"

_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

# Every prefix a hosted deployment refuses, with why it is data plane rather
# than management. Kept as data so a prefix added later cannot pick up a
# different method list from the one beside it, and so the reasoning stays
# attached to the prefix instead of floating between near-identical blocks.
#
# ``tests/integration/test_hosted_mode_surface`` derives its coverage from this
# tuple against the routers `main._register_core_routers` actually gates, so a
# router dropped there without a prefix added here fails rather than shipping
# the bare 404 this module exists to avoid.
DATA_PLANE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("/v1/chat", "OpenAI-compatible completions, the path otari#822 was found on"),
    (
        "/v1/messages",
        "Anthropic-shaped completions. The catch-all takes /v1/messages/count_tokens "
        "with it: that one bills nothing and contacts no provider, but it sizes a "
        "prompt for a completion this deployment will not serve",
    ),
    ("/v1/responses", "the Responses API surface"),
    ("/v1/embeddings", "priced per token like a completion"),
    ("/v1/images", "priced per image"),
    ("/v1/audio", "transcription and speech, priced per second or per character"),
    ("/v1/rerank", "priced per request"),
    ("/v1/moderations", "dispatches upstream even where the upstream charges nothing"),
    (
        "/v1/search",
        "not /v1/search-tools, which is the management catalog this dispatches "
        "against and stays mounted. The catch-all cannot reach it either, since it "
        "only matches under /v1/search/",
    ),
    ("/v1/batches", "queues completions, so it is dispatch deferred rather than avoided"),
    (
        "/v1/files",
        "dispatches to no provider and costs nothing to serve, so not the leak "
        "itself. It exists only to be referenced from a completion or a batch, and "
        "with both refused here an upload has no consumer on this deployment",
    ),
)

router = APIRouter(tags=["hosted-mode"])


def _detail(data_plane_url: str | None) -> str:
    target = data_plane_url or _GENERIC_TARGET
    return (
        "This deployment is a control plane and does not serve inference. "
        f"Send inference requests to {target} instead."
    )


def _register(prefix: str) -> None:
    """Mount the base path and its catch-all for one refused prefix."""

    async def refuse(config: Annotated[GatewayConfig, Depends(get_config)]) -> None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_detail(config.data_plane_url))

    # A distinct name per prefix, so the generated operation ids stay as
    # readable as the hand-written siblings' in hybrid_mode.
    refuse.__name__ = f"{prefix.removeprefix('/v1/').replace('/', '_')}_disabled"
    for path in (prefix, f"{prefix}/{{path:path}}"):
        router.api_route(path, methods=_METHODS)(refuse)


for _prefix, _why in DATA_PLANE_PREFIXES:
    _register(_prefix)
