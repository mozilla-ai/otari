"""Locate the built admin dashboard bundle.

The dashboard is a React + HeroUI single-page app that lives in ``web/`` and is
built (``npm --prefix web run build``) into ``gateway/static/dashboard`` so it
ships inside the Python package. The gateway serves it at ``/`` in standalone
mode; if the bundle is missing (e.g. a source checkout that has not run the
build) the caller falls back to the root tutorial page.
"""

from importlib import resources
from pathlib import Path

DASHBOARD_PACKAGE_PATH = "static/dashboard"


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
