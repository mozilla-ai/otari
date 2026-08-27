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

The status stays 404 to match the hybrid stubs, because it is the same fact
being reported. It is not an authorization decision, so 403 would invite a retry
with a different key that cannot help, and Otari does implement these endpoints,
so 501 would misreport a deployment posture as a missing feature.
"""

from fastapi import APIRouter, HTTPException, status

_DISABLED_DETAIL = (
    "This deployment is a control plane and does not serve inference. "
    "Send inference requests to your Otari gateway instead."
)

_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

router = APIRouter(tags=["management-only"])


def _raise_disabled() -> None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_DISABLED_DETAIL)


@router.api_route("/v1/chat/{path:path}", methods=_METHODS)
@router.api_route("/v1/chat", methods=_METHODS)
async def chat_disabled() -> None:
    _raise_disabled()


# The two non-OpenAI completion shapes. Both carry their sub-paths through the
# catch-all arm, which is what takes /v1/messages/count_tokens with them. That
# one bills nothing itself, since it estimates locally and contacts no provider,
# but it sizes a prompt for a completion this deployment will not serve, so it
# belongs to the plane it answers for rather than staying behind alone.
@router.api_route("/v1/messages/{path:path}", methods=_METHODS)
@router.api_route("/v1/messages", methods=_METHODS)
async def messages_disabled() -> None:
    _raise_disabled()


@router.api_route("/v1/responses/{path:path}", methods=_METHODS)
@router.api_route("/v1/responses", methods=_METHODS)
async def responses_disabled() -> None:
    _raise_disabled()


@router.api_route("/v1/embeddings/{path:path}", methods=_METHODS)
@router.api_route("/v1/embeddings", methods=_METHODS)
async def embeddings_disabled() -> None:
    _raise_disabled()


@router.api_route("/v1/images/{path:path}", methods=_METHODS)
@router.api_route("/v1/images", methods=_METHODS)
async def images_disabled() -> None:
    _raise_disabled()


@router.api_route("/v1/audio/{path:path}", methods=_METHODS)
@router.api_route("/v1/audio", methods=_METHODS)
async def audio_disabled() -> None:
    _raise_disabled()


@router.api_route("/v1/rerank/{path:path}", methods=_METHODS)
@router.api_route("/v1/rerank", methods=_METHODS)
async def rerank_disabled() -> None:
    _raise_disabled()


@router.api_route("/v1/moderations/{path:path}", methods=_METHODS)
@router.api_route("/v1/moderations", methods=_METHODS)
async def moderations_disabled() -> None:
    _raise_disabled()


# /v1/search, not /v1/search-tools: the tool catalog is management and stays.
# The catch-all arm cannot reach it either, since it only matches under
# /v1/search/.
@router.api_route("/v1/search/{path:path}", methods=_METHODS)
@router.api_route("/v1/search", methods=_METHODS)
async def search_disabled() -> None:
    _raise_disabled()


@router.api_route("/v1/batches/{path:path}", methods=_METHODS)
@router.api_route("/v1/batches", methods=_METHODS)
async def batches_disabled() -> None:
    _raise_disabled()


# Files dispatch to no provider and cost nothing to serve, so they are not
# themselves the leak. They are stubbed because they exist only to be referenced
# from a completion or a batch: with both refused here, an upload has no
# consumer on this deployment and would leave a multi-tenant control plane
# storing blobs nothing can ever read back.
@router.api_route("/v1/files/{path:path}", methods=_METHODS)
@router.api_route("/v1/files", methods=_METHODS)
async def files_disabled() -> None:
    _raise_disabled()
