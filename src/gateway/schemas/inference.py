"""Request and response models of the inference domain's decisions, rerank and embeddings endpoints.

The decisions shape is TypeSafe's ``/v1/systemone``, which OpenRouter's alpha Decisions
API also takes, so either provider's SDK can point at the gateway. Questions and
answers allow extra fields, so a field a provider adds later reaches the caller
instead of being dropped here.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Instructions, criteria and state may each be text or structured JSON.
DecisionContent = str | dict[str, Any] | list[Any]

# A vision decision model reads a few images, not an album; the model may cap it lower.
MAX_DECISION_IMAGES = 16


class NoulCriteria(BaseModel):
    """What a yes and a no each mean, for a ``noul`` question."""

    model_config = ConfigDict(extra="allow")

    true: DecisionContent | None = None
    false: DecisionContent | None = None


class NoulQuestion(BaseModel):
    """A yes-or-no question, answered with the probability of yes."""

    model_config = ConfigDict(extra="allow")

    type: Literal["noul"]
    instructions: DecisionContent
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    """A question answered with one of the named options."""

    model_config = ConfigDict(extra="allow")

    type: Literal["choice"]
    instructions: DecisionContent
    criteria: dict[str, DecisionContent | None] = Field(
        min_length=2, max_length=255, description="Option name to what it means"
    )


class ScoreQuestion(BaseModel):
    """A question answered with a level on an ordered scale."""

    model_config = ConfigDict(extra="allow")

    type: Literal["score"]
    instructions: DecisionContent
    criteria: list[DecisionContent] = Field(min_length=2, max_length=10, description="Level descriptions, lowest first")


DecisionQuestion = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class DecisionRequest(BaseModel):
    """A decisions request: typed questions to answer about one state."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "typesafe:jev-latest",
                "state": "The Stripe integration has failed for 3 days and I'm losing sales.",
                "questions": {
                    "urgency": {"type": "noul", "instructions": "Does this message express urgency?"},
                },
            }
        },
    )

    model: str = Field(description="Decision provider and model, e.g. 'typesafe:jev-latest'")
    state: DecisionContent = Field(description="The content the questions are about")
    questions: dict[str, DecisionQuestion] = Field(min_length=1, description="Questions, keyed by answer name")
    images: list[str] | None = Field(
        default=None,
        max_length=MAX_DECISION_IMAGES,
        description=(
            "Images for a vision decision model, as data URLs (data:image/...;base64,...). "
            "An extension llama-server supports; a provider that does not refuses the request."
        ),
    )
    user: str | None = Field(default=None, description="User ID for usage attribution; not sent upstream")

    @field_validator("images")
    @classmethod
    def _images_are_data_urls(cls, images: list[str] | None) -> list[str] | None:
        """Accept inline images only, so the gateway never forwards a URL for a provider to fetch."""
        if images is not None and not all(image.startswith("data:image/") for image in images):
            msg = "each image must be a data URL (data:image/...;base64,...)"
            raise ValueError(msg)
        return images


class DecisionAnswer(BaseModel):
    """One question's answer. Which value field is set follows ``type``."""

    model_config = ConfigDict(extra="allow")

    type: str
    noul: float | None = Field(default=None, description="Probability of yes, for a noul question")
    choice: str | None = Field(default=None, description="The chosen option, for a choice question")
    score: float | None = Field(default=None, description="The level, for a score question")
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    legend: dict[str, Any] | None = None


class DecisionUsage(BaseModel):
    """Token counts the provider reported, plus its own cost where it reports one."""

    model_config = ConfigDict(extra="allow")

    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = Field(default=None, ge=0, description="The provider's own charge in USD, when it reports one")


class DecisionResponse(BaseModel):
    """The provider's answers, keyed like the request's questions."""

    model_config = ConfigDict(extra="allow")

    model: str
    answers: dict[str, DecisionAnswer]
    usage: DecisionUsage | None = None


def _is_none(value: object) -> bool:
    return value is None


class InlineCost(BaseModel):
    """A pass-through response's ``cost_usd`` and ``pricing_source``, the fields a completion's usage carries.

    Both are present when the request was priced and both are left out when it
    was not, so a caller tells an unpriced request by their absence, as on a
    completion.
    """

    cost_usd: str | None = Field(
        default=None,
        pattern=r"^-?\d+\.\d{6}$",
        exclude_if=_is_none,
        description="What the request cost, as a six-decimal USD string; absent when it was not priced",
    )
    # A string rather than ``PriceSource``: a hybrid gateway carries the control
    # plane's own name for the rate, as a completion's usage does.
    pricing_source: str | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "Which rate priced the request: `organization`, `deployment` or `defaults` on a standalone "
            "gateway, and the control plane's own name for it on a hybrid one; absent when it was not priced"
        ),
    )


class RerankResult(BaseModel):
    """One document's relevance to the query."""

    index: int = Field(description="Zero-based index into the request's documents")
    relevance_score: float = Field(description="Relevance score, higher is more relevant")


class RerankMeta(BaseModel):
    """The provider's own billing report, passed through as it sent it."""

    billed_units: dict[str, float] | None = None
    tokens: dict[str, int] | None = None


class RerankUsage(InlineCost):
    """What a rerank request was metered at, and what it cost."""

    total_tokens: int | None = Field(default=None, description="Tokens the provider reported, or null")
    search_units: int | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "Search units the request was billed for, present only when the model is priced per request: "
            "the provider's count rounded up, or one when it reported none"
        ),
    )


class RerankResponse(BaseModel):
    """Documents ordered by relevance, with the gateway's metered usage."""

    id: str | None = Field(default=None, description="Provider-assigned response ID")
    model: str | None = Field(default=None, description="The model that served the request")
    results: list[RerankResult] = Field(description="Results sorted by relevance_score, highest first")
    meta: RerankMeta | None = Field(default=None, description="The provider's own billing report, unchanged")
    usage: RerankUsage


class EmbeddingData(BaseModel):
    """One input's embedding, in OpenAI's shape."""

    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float] | str = Field(
        description="The vector, or its base64 encoding when the request asked for `encoding_format: base64`"
    )


class EmbeddingUsage(InlineCost):
    """What an embeddings request was metered at, and what it cost."""

    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """An OpenAI-compatible embeddings response, with the gateway's inline cost on ``usage``."""

    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage | None = Field(
        default=None, exclude_if=_is_none, description="Absent when the provider reported no usage"
    )
