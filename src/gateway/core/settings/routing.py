"""Automatic provider selection settings."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from gateway.core.settings_view import OMITTED

Region = Literal["us", "eu"]


class AutomaticOffering(BaseModel):
    """Operator-verified endpoint location and relative serving capacity."""

    model_config = ConfigDict(extra="forbid")
    region: Region | None = None
    capacity: int = Field(default=1, ge=1, le=100000)


class AutomaticRouting(BaseModel):
    """Request-aware ordering of interchangeable catalog offerings."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    session_ttl_seconds: int = Field(default=3600, ge=1, le=86400)
    default_region: Region | None = None
    country_header: str | None = Field(default=None, pattern=r"^[A-Za-z0-9-]+$")
    offerings: dict[str, AutomaticOffering] = Field(default_factory=dict, max_length=10000)


class RoutingSettings(BaseModel):
    automatic_routing: Annotated[AutomaticRouting, OMITTED] = Field(default_factory=AutomaticRouting)
