"""Locate the built admin dashboard bundle.

The dashboard is a React + HeroUI single-page app that lives in ``web/`` and is
built (``make dashboard``) into ``gateway/static/dashboard`` so it ships inside
the Python package. The built bundle is gitignored rather than committed, so a
missing one is expected, not a defect: the Docker image builds it in a Node
stage, while a source checkout or a wheel built without that step has none. The
gateway serves it at ``/``, and at every other page path a browser opens, and
falls back to the root tutorial page when it is absent.
"""

import hashlib
from importlib import resources
from pathlib import Path

from starlette.datastructures import Headers
from starlette.responses import FileResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from gateway.core.config import OTLP_ROOT

DASHBOARD_PACKAGE_PATH = "static/dashboard"

# Namespaces whose unmatched paths stay a plain 404: a client of the API never
# gets the dashboard's page back for a mistyped path.
_NOT_DASHBOARD_PREFIXES = ("/api", OTLP_ROOT)


def get_dashboard_dir() -> Path | None:
    """Return the built dashboard directory, or ``None`` if it is not present.

    Returns ``None`` rather than raising so the app can degrade gracefully to
    the tutorial page when the frontend has not been built.
    """
    try:
        base = resources.files("gateway").joinpath(DASHBOARD_PACKAGE_PATH)
        index = base.joinpath("index.html")
        if not index.is_file():
            return None
        path = Path(str(base))
    except (FileNotFoundError, ModuleNotFoundError, NotADirectoryError):
        return None
    return path if (path / "index.html").is_file() else None


def get_dashboard_build_id(dashboard_dir: Path) -> str:
    """Identify the built bundle, so a running tab can tell it went stale.

    Digests ``index.html`` rather than the assets it points at: Vite stamps the
    content-hashed asset names into that page, so it changes only when the app
    the browser would load changes, and stays put when only the gateway around it
    is upgraded. A tab holding an older id is running code the server no longer
    serves and should offer to reload.
    """
    return hashlib.sha256((dashboard_dir / "index.html").read_bytes()).hexdigest()[:16]


def is_dashboard_navigation(scope: Scope) -> bool:
    """Whether a request is a browser opening a page of the dashboard.

    A ``GET`` or ``HEAD`` that accepts HTML, for a path outside the API's
    namespaces. The dashboard routes in the browser, so every one of its pages
    is the same ``index.html``, and a reload or a pasted link to one of them
    reaches the server as an ordinary path.
    """
    if scope["type"] != "http" or scope["method"] not in {"GET", "HEAD"}:
        return False
    path: str = scope["path"]
    if any(path == prefix or path.startswith(f"{prefix}/") for prefix in _NOT_DASHBOARD_PREFIXES):
        return False
    return "text/html" in Headers(scope=scope).get("accept", "")


def dashboard_fallback(index_file: Path, not_found: ASGIApp) -> ASGIApp:
    """The router's answer when no route matches: the dashboard's page, or ``not_found``.

    Installed as the router's ``default`` rather than as a catch-all route, so it
    runs only after every route, mount and overlay contribution has declined the
    path. A route added later can therefore never be shadowed by it.
    """

    async def fallback(scope: Scope, receive: Receive, send: Send) -> None:
        if is_dashboard_navigation(scope):
            await FileResponse(index_file, media_type="text/html")(scope, receive, send)
            return
        await not_found(scope, receive, send)

    return fallback
