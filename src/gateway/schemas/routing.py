"""Request and response models of the routing domain."""

from pydantic import BaseModel, ConfigDict, Field


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

    harness: str = Field(min_length=1, max_length=64, description="The agent harness asking, such as `claude-code`.")
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
                "reason": "jev-1.13.0 chose sonnet with 72%",
                "probabilities": {"haiku": 0.21, "sonnet": 0.72, "opus": 0.07},
            }
        }
    )

    model: str = Field(description="A model alias or id the harness can start the subagent on.")
    reason: str | None = Field(default=None, description="Why, in one short sentence the harness may show.")
    probabilities: dict[str, float] | None = Field(
        default=None, description="The decision model's probability for each candidate, when it reports them."
    )
