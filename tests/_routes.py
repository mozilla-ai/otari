"""Read the routes an app mounts, with the prefixes they are included under.

Since FastAPI 0.137 an included router stays a router inside ``app.routes``
rather than copying its routes in, so the list is a tree and a route in it
carries only the path declared on it. FastAPI walks that tree for the OpenAPI
document; the tests read it the same way.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from fastapi.routing import APIRoute, iter_route_contexts
from starlette.routing import BaseRoute


@dataclass(frozen=True)
class MountedRoute:
    """A path operation and the path it answers on."""

    path: str
    route: APIRoute

    @property
    def methods(self) -> frozenset[str]:
        return frozenset(self.route.methods or ())

    @property
    def endpoint(self) -> Callable[..., Any]:
        return self.route.endpoint


def mounted_paths(routes: Sequence[BaseRoute]) -> set[str]:
    """Every path ``routes`` answers on, path operations and mounts alike."""
    return {context.path for context in iter_route_contexts(routes) if isinstance(context.path, str)}


def mounted_api_routes(routes: Sequence[BaseRoute]) -> list[MountedRoute]:
    """The path operations in ``routes``, in the order they are tried against a request."""
    mounted = []
    for context in iter_route_contexts(routes):
        route = context.original_route
        if isinstance(route, APIRoute) and isinstance(context.path, str):
            mounted.append(MountedRoute(context.path, route))
    return mounted
