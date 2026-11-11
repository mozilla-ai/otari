"""The credential check holds a key to the scope its route declares, and refuses it where the route declares none."""

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute
from starlette.requests import Request

from gateway.api.deps import _require_route_scope
from gateway.api.key_scopes import key_scope
from gateway.models.api_keys import APIKey
from gateway.types.key_scope import KeyScope


async def _endpoint() -> None:
    return None


def _request(route: APIRoute) -> Request:
    return Request({"type": "http", "method": "GET", "path": route.path, "headers": [], "route": route})


def _route(**openapi_extra: object) -> APIRoute:
    return APIRoute("/thing", _endpoint, openapi_extra=dict(openapi_extra) or None)


def _key(*scopes: KeyScope) -> APIKey:
    return APIKey(id="key-1", scopes=[scope.value for scope in scopes])


def test_a_key_holding_the_scope_passes() -> None:
    _require_route_scope(_request(_route(**key_scope(KeyScope.FILES_READ))), _key(KeyScope.FILES_READ))


def test_a_key_lacking_the_scope_is_refused_with_its_name() -> None:
    with pytest.raises(HTTPException) as refused:
        _require_route_scope(_request(_route(**key_scope(KeyScope.FILES_WRITE))), _key(KeyScope.FILES_READ))

    assert refused.value.status_code == 403
    assert "'files:write'" in str(refused.value.detail)


def test_a_route_open_to_every_key_passes_any_key() -> None:
    _require_route_scope(_request(_route(**key_scope(None))), _key(KeyScope.USAGE_READ))


@pytest.mark.parametrize("route", [_route(), _route(requestBody={"required": True})], ids=["bare", "other-extras"])
def test_a_route_declaring_no_scope_refuses_every_key(route: APIRoute) -> None:
    with pytest.raises(HTTPException) as refused:
        _require_route_scope(_request(route), _key(*KeyScope))

    assert refused.value.status_code == 403
    assert refused.value.detail == "This endpoint does not accept API keys."
