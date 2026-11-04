"""The dashboard surface a route group serves."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Surface:
    """A dashboard surface a route group serves.

    ``name`` is what the dashboard gates pages on; several pages can share one.
    Publishing a surface does not authorize a caller to use it.
    """

    name: str
