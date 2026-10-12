"""What a pass-through endpoint's provider result reports as billable: its tokens and its counts by meter."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from any_llm.types.audio import Transcription
from any_llm.types.completion import CreateEmbeddingResponse
from any_llm.types.image import ImagesResponse
from any_llm.types.rerank import RerankResponse

from gateway.services.pricing import billed_search_units


@dataclass(frozen=True)
class ReportedUsage:
    """The tokens and the counts one provider result reports, which is all a rate can price it on."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    units: Mapping[str, int] = field(default_factory=dict)


def embedding_usage(result: CreateEmbeddingResponse) -> ReportedUsage:
    """Input tokens only: an embedding generates no output."""
    usage = result.usage
    if usage is None:
        return ReportedUsage()
    return ReportedUsage(input_tokens=usage.prompt_tokens, output_tokens=0, total_tokens=usage.total_tokens)


def rerank_usage(result: RerankResponse) -> ReportedUsage:
    """The search units the provider billed, at least one, and its tokens where it reports them."""
    total = result.usage.total_tokens if result.usage else None
    search_units = billed_search_units(result.meta.billed_units if result.meta else None)
    return ReportedUsage(input_tokens=total, output_tokens=0, total_tokens=total, units={"search_units": search_units})


def image_usage(result: ImagesResponse) -> ReportedUsage:
    """The images returned, and the tokens a token-billed image model reports beside them."""
    units = {"images": len(result.data or [])}
    usage = result.usage
    if usage is None:
        return ReportedUsage(input_tokens=0, output_tokens=0, total_tokens=0, units=units)
    return ReportedUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        units=units,
    )


def transcription_usage(result: Transcription) -> ReportedUsage:
    """One request, and the tokens a token-billed transcription model reports beside it."""
    units = {"requests": 1}
    usage = result.usage
    if usage is None or usage.type != "tokens":
        return ReportedUsage(input_tokens=0, output_tokens=0, total_tokens=0, units=units)
    return ReportedUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        units=units,
    )


def one_request_usage(_result: object) -> ReportedUsage:
    """One request and no tokens: speech returns bytes."""
    return ReportedUsage(input_tokens=0, output_tokens=0, total_tokens=0, units={"requests": 1})


def moderation_usage(_result: object) -> ReportedUsage:
    """One request, and no input token count: a moderation reports nothing it was billed on."""
    return ReportedUsage(output_tokens=0, units={"requests": 1})


@dataclass(frozen=True)
class PassthroughBilling:
    """How a pass-through endpoint is billed: what its result reports, and what that means unpriced.

    ``meter`` and ``reports_tokens`` say what the endpoint can report before it
    is called, which is what the ``require_pricing`` gate checks a rate against.
    ``unpriced_is_free`` settles an attempt no rate prices at $0 rather than
    leaving it unpriced, the documented convention of the per-request endpoints.
    """

    report: Callable[[Any], ReportedUsage]
    meter: str | None
    reports_tokens: bool
    unpriced_is_free: bool = False


EMBEDDINGS_BILLING = PassthroughBilling(report=embedding_usage, meter=None, reports_tokens=True)
RERANK_BILLING = PassthroughBilling(report=rerank_usage, meter="search_units", reports_tokens=True)
IMAGES_BILLING = PassthroughBilling(report=image_usage, meter="images", reports_tokens=False)
TRANSCRIPTION_BILLING = PassthroughBilling(
    report=transcription_usage, meter="requests", reports_tokens=False, unpriced_is_free=True
)
ONE_REQUEST_BILLING = PassthroughBilling(
    report=one_request_usage, meter="requests", reports_tokens=False, unpriced_is_free=True
)
MODERATION_BILLING = PassthroughBilling(
    report=moderation_usage, meter="requests", reports_tokens=False, unpriced_is_free=True
)
