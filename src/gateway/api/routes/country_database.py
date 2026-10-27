"""The country file stored in the database (``/api/v1/settings/country-database``). Operator-gated.

An operator uploads a MaxMind DB country file from the dashboard instead of
mounting one into every container. The replica that served the upload reads it
at once, and every other replica within 30 seconds.
"""

from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from starlette.datastructures import UploadFile
from starlette.requests import ClientDisconnect
from starlette.types import Message

from gateway.api.deps import InferenceServiceDep, require_deployment_operator
from gateway.schemas.inference import CountryDatabaseStatus

router = APIRouter(
    prefix="/settings/country-database",
    tags=["settings"],
    dependencies=[Depends(require_deployment_operator)],
)

_READ_CHUNK_BYTES = 1024 * 1024
# The service's 32 MiB file limit, plus room for the multipart framing around the file.
_MAX_BODY_BYTES = 33 * 1024 * 1024
_TOO_LARGE = "The file is larger than 32 MiB"
_UPLOAD_SCHEMA = {
    "type": "object",
    "properties": {
        "file": {
            "type": "string",
            "contentMediaType": "application/octet-stream",
            "description": "A MaxMind DB country file, such as GeoLite2 Country or DB-IP Lite Country.",
        }
    },
    "required": ["file"],
}


async def _upload_chunks(file: UploadFile) -> AsyncIterator[bytes]:
    while chunk := await file.read(_READ_CHUNK_BYTES):
        yield chunk


def _bounded(request: Request) -> Request:
    """``request``, refusing its body once more than ``_MAX_BODY_BYTES`` arrive, whatever ``Content-Length`` says."""
    received = 0

    async def receive() -> Message:
        nonlocal received
        message = await request.receive()
        received += len(message.get("body", b""))
        if received > _MAX_BODY_BYTES:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, _TOO_LARGE)
        return message

    return Request(request.scope, receive)


@router.get("")
async def get_country_database(service: InferenceServiceDep) -> CountryDatabaseStatus:
    """Describe the stored country file, and say whether ``client_country_database`` is read instead."""
    return await service.country_database_status()


@router.put(
    "",
    responses={
        413: {"description": "The file is larger than 32 MiB."},
        422: {"description": "The body holds no file in the field ``file``."},
    },
    openapi_extra={"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": _UPLOAD_SCHEMA}}}},
)
async def upload_country_database(request: Request, service: InferenceServiceDep) -> CountryDatabaseStatus:
    """Store a country file in place of the one stored before.

    A file larger than 32 MiB answers 413, and one that is no MaxMind DB
    country file answers 400. Either way the file stored before stays. While
    ``client_country_database`` is set, the stored file is kept but not read.
    """
    # Read here rather than as a File parameter, which FastAPI parses before the operator gate runs.
    try:
        async with _bounded(request).form(max_files=1, max_fields=0) as form:
            file = form.get("file")
            if not isinstance(file, UploadFile):
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Send the country file in the field 'file'.")
            return await service.store_country_database(_upload_chunks(file))
    except ClientDisconnect:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The upload was interrupted.") from None


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def delete_country_database(service: InferenceServiceDep) -> Response:
    """Remove the stored country file.

    Answers 404 when none is stored, and 409 when it is the only country source
    for ``provider_supported_countries`` or for the supported countries of an
    enabled hosted provider.
    """
    await service.remove_country_database()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
