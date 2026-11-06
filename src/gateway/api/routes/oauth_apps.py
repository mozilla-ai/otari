"""OAuth app consent and credential delivery."""

from collections.abc import Callable, Coroutine
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.routing import APIRoute
from starlette.datastructures import MutableHeaders

from gateway.api.deps import CurrentIdentity, OAuthAppServiceDep, OrganizationServiceDep, verify_master_key
from gateway.api.routes._helpers import bounded_body
from gateway.api.routes._public_auth import throttle_public_auth
from gateway.core.feature import CoreFeature
from gateway.models.tenancy import WorkspacesPublic
from gateway.schemas.api_keys import (
    AppAuthorizationParameters,
    AppAuthorizationRequest,
    AppAuthorizationResponse,
    AppDetails,
    AppTokenRequest,
    AppTokenResponse,
    OAuthErrorResponse,
)
from gateway.services.api_keys import OAuthAppGrantError

_NO_STORE = {"Cache-Control": "no-store", "Pragma": "no-cache"}
# Fits the bounded fields even when Unicode values are UTF-8 percent-encoded.
_TOKEN_BODY_LIMIT = 32 * 1024


class OAuthAppRoute(APIRoute):
    """OAuth protocol refusals use RFC 6749 error bodies, including invalid form parameters."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handle = super().get_route_handler()

        async def oauth_response(request: Request) -> Response:
            throttle_public_auth(request)
            try:
                if request.method == "POST":
                    media_type, separator, media_parameters = request.headers.get("content-type", "").partition(";")
                    media_type = media_type.strip().lower()
                    if media_type != "application/x-www-form-urlencoded":
                        raise OAuthAppGrantError("invalid_request", "Use application/x-www-form-urlencoded")
                    # Starlette's form parser compares the media type case-sensitively.
                    headers = MutableHeaders(headers=request.headers)
                    headers["content-type"] = media_type + separator + media_parameters
                    request = bounded_body(
                        Request({**request.scope, "headers": headers.raw}, request.receive),
                        _TOKEN_BODY_LIMIT,
                        "OAuth token request is too large",
                    )
                parameters = await request.form() if request.method == "POST" else request.query_params
                if len(parameters.multi_items()) != len(parameters):
                    raise OAuthAppGrantError("invalid_request", "OAuth parameters must not be repeated")
                return await handle(request)
            except RequestValidationError as exc:
                unsupported = any(
                    error["loc"] == ("body", "grant_type") and error["type"] == "literal_error"
                    for error in exc.errors()
                )
                error = "unsupported_grant_type" if unsupported else "invalid_request"
                body = OAuthErrorResponse(error=error, error_description="Invalid OAuth request parameters")
            except OAuthAppGrantError as exc:
                body = OAuthErrorResponse(error=exc.error, error_description=exc.description)
            return JSONResponse(body.model_dump(), status_code=status.HTTP_400_BAD_REQUEST, headers=_NO_STORE)

        return oauth_response


router = APIRouter(
    prefix="/oauth/apps", tags=["oauth-apps"], dependencies=[Depends(verify_master_key), Depends(throttle_public_auth)]
)
public_router = APIRouter(prefix="/oauth/apps", tags=["oauth-apps"], route_class=OAuthAppRoute)


@public_router.get(
    "/authorize",
    status_code=status.HTTP_302_FOUND,
    response_class=RedirectResponse,
    responses={
        status.HTTP_302_FOUND: {
            "description": "Continue to dashboard sign-in and consent.",
            "headers": {"Location": {"schema": {"type": "string", "format": "uri-reference"}}},
        },
        status.HTTP_400_BAD_REQUEST: {"model": OAuthErrorResponse},
    },
)
async def authorization_redirect(
    params: Annotated[AppAuthorizationParameters, Query()], service: OAuthAppServiceDep
) -> RedirectResponse:
    """Start authorization with standard OAuth query parameters, preserving them through dashboard sign-in."""
    return RedirectResponse(service.authorization_url(params), status_code=status.HTTP_302_FOUND, headers=_NO_STORE)


@public_router.get("/clients")
async def app_details(
    client_id: Annotated[str, Query(min_length=1, max_length=128)],
    redirect_uri: Annotated[str, Query(max_length=2048)],
    service: OAuthAppServiceDep,
) -> AppDetails:
    """Validate the app and redirect before the dashboard offers consent."""
    return service.details(client_id, redirect_uri)


@router.post("/authorize")
async def authorize(
    body: AppAuthorizationRequest, identity: CurrentIdentity, service: OAuthAppServiceDep, response: Response
) -> AppAuthorizationResponse:
    """Approve access to the caller's selected workspace, with normal budget enforcement."""
    response.headers.update(_NO_STORE)
    return AppAuthorizationResponse(redirect_uri=await service.authorize(identity, body))


@router.get("/workspaces")
async def app_workspaces(
    identity: CurrentIdentity,
    service: OrganizationServiceDep,
    search: Annotated[str, Query(max_length=200)] = "",
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 50,
) -> WorkspacesPublic:
    """List workspaces where the signed-in identity may authorize app access."""
    return await service.list_app_workspaces(user=identity, search=search, skip=skip, limit=limit)


@public_router.post("/token", response_model=AppTokenResponse, responses={400: {"model": OAuthErrorResponse}})
async def exchange(body: Annotated[AppTokenRequest, Form()], service: OAuthAppServiceDep) -> JSONResponse:
    """Exchange a form-encoded single-use authorization code and S256 verifier for a revocable API key."""
    token = AppTokenResponse(access_token=await service.exchange(body))
    return JSONResponse(token.model_dump(), headers=_NO_STORE)


FEATURE = CoreFeature(
    name="oauth_apps",
    surface=None,
    enabled=lambda config: bool(config.oauth_apps),
    routers=lambda config: (router, public_router),
)
