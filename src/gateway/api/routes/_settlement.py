"""The settled cost a completion route can attach to its response."""

from __future__ import annotations

from pydantic import BaseModel, Field

# What the gateway attaches as a settled cost, checked at this edge: a
# negative amount or a source name of any shape reaches a client otherwise.
_SETTLED_COST_PATTERN = r"^\d+\.\d{6}$"
_PRICING_SOURCE_PATTERN = r"^[a-z_]{1,32}$"


class SettledCost(BaseModel):
    """An attachable settlement: a priced cost and the basis for it."""

    cost_usd: str = Field(pattern=_SETTLED_COST_PATTERN)
    pricing_source: str = Field(pattern=_PRICING_SOURCE_PATTERN)
