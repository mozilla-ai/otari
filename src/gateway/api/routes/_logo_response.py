"""How a logo is sent: as an image the browser may cache, which cannot run anything if opened on its own."""

from fastapi import Request, Response, status

from gateway.services.mcp_catalog import Logo

# A logo is served only to a signed-in dashboard, so no shared cache keeps it.
_VERSIONED = "private, max-age=604800, immutable"
_UNVERSIONED = "private, max-age=300"
# An SVG opened directly must not run script, nor load anything.
_IMAGE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; sandbox",
    "Content-Disposition": "inline",
}

LOGO_RESPONSES: dict[int | str, dict[str, object]] = {
    status.HTTP_200_OK: {
        "content": {"image/*": {}},
        "description": "The logo, as PNG, JPEG, GIF, WebP, ICO or SVG.",
    },
    status.HTTP_404_NOT_FOUND: {"description": "No logo is known, or none was found yet."},
}


def logo_response(request: Request, logo: Logo | None) -> Response:
    """The image, versioned for a long cache when the URL names its digest, or an empty 404 nobody caches."""
    if logo is None:
        return Response(status_code=status.HTTP_404_NOT_FOUND, headers={"Cache-Control": "no-store"})
    etag = f'"{logo.etag}"'
    # A 304 carries the policy its 200 would, since a cache refreshes what it kept from it.
    cache_control = _VERSIONED if request.query_params.get("v") == logo.etag else _UNVERSIONED
    headers = {"Cache-Control": cache_control, "ETag": etag, **_IMAGE_HEADERS}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=headers)
    return Response(content=logo.content, media_type=logo.content_type, headers=headers)
