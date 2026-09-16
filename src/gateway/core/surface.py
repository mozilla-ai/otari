"""The dashboard surface a route group serves."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Surface:
    """A dashboard surface a route group serves, and the deployments that publish it.

    ``name`` is what the dashboard gates pages on; several pages can share one.
    Publishing a surface does not authorize a caller to use it.
    Raises ``ValueError`` when no deployment publishes it.
    """

    name: str
    standalone: bool = True
    hosted: bool = True

    def __post_init__(self) -> None:
        if not (self.standalone or self.hosted):
            msg = f"Surface {self.name!r} is published by no deployment"
            raise ValueError(msg)
