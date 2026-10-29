"""Request and response models of the routing domain."""

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from gateway.core.config import MAX_AGENT_MODEL_CANDIDATES

Harness = Literal["claude-code", "pi"]
"""The agent harnesses Otari recommends a subagent's model for."""

OtariModelId = Annotated[str, StringConstraints(max_length=256, pattern=r"^[^:\s]+:\S+$")]
"""A model as Otari names it, ``<provider>:<model>``."""


class AgentModelRecommendationRequest(BaseModel):
    """A coding agent about to start a subagent, asking which model it should run on.

    Every field but ``user`` is a fact the harness already holds at spawn time.
    The prompt is the task the subagent is given; it is read for the
    recommendation and never stored or logged.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "harness": "claude-code",
                "session_id": "bf05abe2-5ff2-4eb0-8459-ab578d5c9468",
                "tool_use_id": "toolu_01Ab3dEfGh",
                "agent_type": "Explore",
                "description": "Find the budget reservation code",
                "prompt": "Find where budgets are reserved before dispatch and report the call chain.",
                "parent_model": "claude-opus-5",
                "requested_model": None,
            }
        },
    )

    harness: Harness = Field(
        description=(
            "The agent harness asking. It decides the candidates: `agent_recommender_candidates`, Claude Code's "
            "own aliases, for `claude-code`, and the request's own `candidates` for `pi`."
        )
    )
    candidates: dict[OtariModelId, Annotated[str, StringConstraints(max_length=1024)] | None] | None = Field(
        default=None,
        min_length=2,
        max_length=MAX_AGENT_MODEL_CANDIDATES,
        description=(
            "The models the subagent may be sent to, as Otari model ids (`<provider>:<model>`), each with what "
            "it is for (or null when the name says enough). Required for `pi`, refused for `claude-code`."
        ),
    )
    session_id: str = Field(
        min_length=1, max_length=256, description="The harness's own id for the session the spawn happens in."
    )
    tool_use_id: str = Field(
        min_length=1, max_length=256, description="The harness's own id for the tool call that spawns the subagent."
    )
    agent_type: str = Field(
        min_length=1,
        max_length=128,
        description="The subagent type: a built-in such as `Explore` or `Plan`, or a custom agent's name.",
    )
    description: str = Field(default="", max_length=1024, description="The caller's short description of the task.")
    prompt: str = Field(min_length=1, max_length=65_536, description="The task the subagent is given.")
    parent_model: str = Field(
        min_length=1, max_length=256, description="The model the parent conversation runs on, as the harness names it."
    )
    requested_model: str | None = Field(
        default=None,
        max_length=256,
        description="The model the caller asked for, if any. Sent as a fact; the recommendation is the gateway's.",
    )
    user: str | None = Field(
        default=None,
        description="User ID the decision is billed to when asking with the master key; not sent upstream.",
    )


class AgentModelRecommendation(BaseModel):
    """The model recommended for the subagent."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "sonnet",
                "reason": "sonnet with 72%",
                "probabilities": {"haiku": 0.21, "sonnet": 0.72, "opus": 0.07},
            }
        }
    )

    model: str = Field(description="A model alias or id the harness can start the subagent on.")
    reason: str | None = Field(default=None, description="Why, in one short sentence the harness may show.")
    probabilities: dict[str, float] | None = Field(
        default=None, description="The decision model's probability for each candidate, when it reports them."
    )


class FallbackOfferingPublic(BaseModel):
    """One offering a catalog ID tries, and who runs the built-in tools on it."""

    model_config = ConfigDict(from_attributes=True)

    selector: str = Field(description="The offering, in `instance:model` form.")
    provider: str = Field(description="The provider instance the selector names.")
    provider_type: str = Field(description="The any-llm implementation behind the instance.")
    own_tools: list[str] = Field(
        description=(
            "Built-in tools this provider runs itself, for a caller that declares them in the provider's own words "
            "and leaves the choice of executor to the deployment."
        )
    )
    otari_tools: list[str] = Field(description="Built-in tools Otari runs in this provider's place, for that caller.")


class AutomaticFallbackPublic(BaseModel):
    """One model the caller's organization used in the window, and the fallback a catalog ID gives it."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="The catalog ID a caller sends to get the fallback, e.g. `anthropic/claude-sonnet-5`.")
    name: str
    offerings: list[FallbackOfferingPublic] = Field(
        description="What a request sent by catalog ID tries, first first. One entry means no backup."
    )
    requests: int = Field(description="Requests served in the window, each counted once however many attempts.")
    fell_back: int = Field(description="Of `requests`, the successful ones a later offering served.")
    pinned: int = Field(
        description="Of `requests`, those that named an offering or a policy rather than the catalog ID, "
        "and so got no automatic fallback."
    )
    pinned_provider: str | None = Field(description="The instance most pinned requests reached.")


class AutomaticFallbacksResponse(BaseModel):
    """The automatic fallbacks the caller's organization used."""

    window_days: int = Field(description="How many days of traffic the counts cover.")
    models: list[AutomaticFallbackPublic] = Field(description="Most requested first.")


# Subagent routers: the stored decision models that recommend a subagent's model
#
# No shape here carries key material: a stored key is known by whether there is
# one and its last four characters, which a key of four or fewer goes without.
# Prices are USD per million of the decision model's tokens, as floats, the way
# every other rate on the wire is; the store keeps them exact.

Rate = Annotated[float, Field(ge=0, le=1_000_000)]
"""USD per million tokens. The ceiling keeps a typo inside the stored column."""


class AgentRecommenderCreate(BaseModel):
    """Add a subagent router. The first one added is switched on; a later one waits to be."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(
        min_length=1, max_length=32, description="The decision provider: typesafe, openrouter or llamacpp."
    )
    model: str = Field(min_length=1, max_length=255, description="The decision model, as the provider names it.")
    api_base: str | None = Field(
        default=None, max_length=1024, description="Root URL of the provider; blank for its own."
    )
    api_key: str | None = Field(
        default=None, min_length=1, max_length=1024, description="Stored encrypted, never returned."
    )
    input_price_per_million: Rate = 0.0
    output_price_per_million: Rate = 0.0


class AgentRecommenderUpdate(BaseModel):
    """Change a subagent router. An omitted field keeps its stored value.

    Omitting ``api_key`` keeps the stored key while the provider and the API
    base stay the same, so a price change never has to re-send a secret. A new
    provider or address needs its own key, unless it is a self-hosted server
    that takes none. A blank ``api_base`` goes back to the provider's own
    address. Switching one on switches every other one off.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str | None = Field(default=None, min_length=1, max_length=32)
    model: str | None = Field(default=None, min_length=1, max_length=255)
    api_base: str | None = Field(default=None, max_length=1024)
    api_key: str | None = Field(default=None, min_length=1, max_length=1024)
    input_price_per_million: Rate | None = None
    output_price_per_million: Rate | None = None
    enabled: bool | None = None


class AgentRecommenderPublic(BaseModel):
    """One stored subagent router, known by its provider, model and the tail of its key."""

    id: uuid.UUID
    provider: str
    model: str
    api_base: str | None
    api_key_last4: str | None
    has_api_key: bool
    input_price_per_million: float
    output_price_per_million: float
    enabled: bool
    created_at: datetime
    updated_at: datetime | None


class ConfiguredRecommenderPublic(BaseModel):
    """The decision model ``agent_recommender_model`` names, which answers unpriced while no router is stored."""

    selector: str
    provider: str
    model: str


class AgentRecommendersPublic(BaseModel):
    """Every stored subagent router, oldest first, and the configured decision model, where it resolves."""

    data: list[AgentRecommenderPublic]
    configured: ConfiguredRecommenderPublic | None
