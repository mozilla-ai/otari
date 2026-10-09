"""The API's old root, answered with a 404 that names the address to use instead.

The API moved from ``/v1`` to ``API_ROOT`` (otari#1026), and the old root is
deliberately not served as an alias. OpenAI- and Anthropic-style clients are
commonly configured with a base URL ending in ``/v1``, or append it themselves,
so a caller pointed at the host alone lands here. A bare 404 would not say why;
this one names the equivalent path under the current root.

Mounted beside the API root rather than under it, in every mode, and left out of
the document: it serves nothing a client can call.
"""

from fastapi import APIRouter, HTTPException, Request, status

from gateway.core.config import API_ROOT

RETIRED_ROOT = "/v1"

# ``HEAD`` and ``OPTIONS`` listed for the reason ``hosted_mode`` gives: a 405
# would report the path as served here with only the verb wrong.
_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

router = APIRouter(tags=["retired-root"], include_in_schema=False)


async def retired_root(request: Request) -> None:
    rest = request.path_params.get("path", "")
    target = f"{API_ROOT}/{rest}" if rest else API_ROOT
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Otari serves its API under {API_ROOT}, not {RETIRED_ROOT}. Use {target} instead.",
    )


for _path in (RETIRED_ROOT, f"{RETIRED_ROOT}/{{path:path}}"):
    router.api_route(_path, methods=_METHODS)(retired_root)
