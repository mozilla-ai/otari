"""What a caller sends to rate a routed response."""

from pydantic import BaseModel, ConfigDict, Field


class RoutingFeedbackRequest(BaseModel):
    """A caller's rating of one response, named by the ``Otari-Request-ID`` it came back with."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    request_id: str = Field(
        min_length=1,
        max_length=255,
        description="The `Otari-Request-ID` response header of the request being rated.",
    )
    score: float = Field(ge=0.0, le=1.0, description="How good the response was, from 0 (worst) to 1 (best).")
