"""``GET /api/v1/key-identity``: validate a workspace API key for another service and say whose it is.

For a service that accepts an Otari workspace API key from its own callers and
needs to know who they are (a router or an aggregator running beside Otari,
say). It forwards the key unchanged, in any header Otari reads a key from, and
gets back the key's id, its owner, and the workspace and organization it
belongs to. It is not a dashboard API: a session cookie is never read, and
the master key names no key, so it is refused.

The lookup answers the question and does nothing else. It reserves no budget
and writes no usage row; the only write is the throttled ``last_used_at`` stamp
every key verification makes. The security middleware marks every answer
``private, no-store``.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from gateway.api.deps import FORWARDED_KEY_REFUSED, ApiKeyServiceDep, verify_forwarded_api_key
from gateway.core.database import DATABASE_ERRORS
from gateway.models.api_keys import APIKey
from gateway.schemas.api_keys import KeyIdentity

# Declared on the router as well as taken by the handler, so a route added here
# later starts gated; FastAPI resolves it once per request either way.
router = APIRouter(tags=["key-identity"], dependencies=[Depends(verify_forwarded_api_key)])


@router.get(
    "/key-identity",
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "The key is not live, or its owner may not use it."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "The key could not be checked. Retry."},
    },
)
async def read_key_identity(
    api_key: Annotated[APIKey, Depends(verify_forwarded_api_key)],
    service: ApiKeyServiceDep,
) -> KeyIdentity:
    """Validate a workspace API key and return who owns it.

    Every refusal is the same 401: a key that is missing, malformed, unknown,
    inactive, expired or another deployment's, and a key whose owner is
    deleted or blocked. A 503 means the key was not judged.
    """
    try:
        identity = await service.identify(api_key.id)
    except DATABASE_ERRORS as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication temporarily unavailable, please retry",
        ) from exc
    if identity is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=FORWARDED_KEY_REFUSED)
    return identity
