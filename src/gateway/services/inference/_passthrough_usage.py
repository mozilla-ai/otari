"""What a pass-through endpoint's provider result reports as billable: its tokens, its counts and what it measured."""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from any_llm.types.audio import Transcription
from any_llm.types.completion import CreateEmbeddingResponse
from any_llm.types.image import ImagesResponse
from any_llm.types.rerank import RerankResponse
from any_llm.types.video import VideoJob

from gateway.services.pricing import Quantities, Slot, billed_search_units
from gateway.types.price_unit import PriceUnit

_PIXELS_PER_MEGAPIXEL = Decimal(1_000_000)
_SECONDS_PER_MINUTE = Decimal(60)
_IMAGE_SIZE = re.compile(r"^\s*(\d{1,5})\s*[xX×]\s*(\d{1,5})\s*$")


@dataclass(frozen=True)
class ReportedUsage:
    """The tokens, the counts and the quantities one provider result reports, which is all a rate can price it on.

    ``units`` holds the counts by meter (``images``, ``requests``,
    ``search_units``), and ``quantities`` what the endpoint measured, in each
    unit it can measure, which the slot the endpoint bills under is priced on.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    units: Mapping[str, int] = field(default_factory=dict)
    quantities: Quantities = field(default_factory=dict)


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


def image_megapixels(size: str | None) -> Decimal | None:
    """The megapixels of one image of ``size`` (``1024x1024``), or ``None`` for a size that names no dimensions."""
    found = _IMAGE_SIZE.match(size or "")
    if found is None:
        return None
    width, height = (int(found.group(1)), int(found.group(2)))
    return Decimal(width * height) / _PIXELS_PER_MEGAPIXEL if width and height else None


def image_quantities(size: str | None, count: int) -> Quantities:
    """What ``count`` images of ``size`` measure: the count, and their megapixels where the size names them."""
    quantities: dict[PriceUnit, int | Decimal] = {"image": count}
    megapixels = image_megapixels(size)
    if megapixels is not None:
        quantities["megapixel"] = megapixels * count
    return quantities


def image_usage(result: ImagesResponse, *, size: str | None = None) -> ReportedUsage:
    """The images returned, their megapixels where a size is known, and the tokens a token-billed model reports.

    The size is the response's where it says one, else the ``size`` the request
    asked for.
    """
    count = len(result.data or [])
    quantities = image_quantities(result.size or size, count)
    units = {"images": count}
    usage = result.usage
    if usage is None:
        return ReportedUsage(input_tokens=0, output_tokens=0, total_tokens=0, units=units, quantities=quantities)
    return ReportedUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        units=units,
        quantities=quantities,
    )


def transcription_usage(result: Transcription) -> ReportedUsage:
    """One request, the audio's length where the provider reports it, and the tokens a token-billed model reports."""
    units = {"requests": 1}
    usage = result.usage
    if usage is None:
        return ReportedUsage(input_tokens=0, output_tokens=0, total_tokens=0, units=units)
    if usage.type == "duration":
        seconds = Decimal(str(usage.seconds))
        quantities: dict[PriceUnit, int | Decimal] = {"second": seconds, "minute": seconds / _SECONDS_PER_MINUTE}
        return ReportedUsage(input_tokens=0, output_tokens=0, total_tokens=0, units=units, quantities=quantities)
    return ReportedUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        units=units,
    )


def speech_usage(characters: int) -> Callable[[Any], ReportedUsage]:
    """One request and the ``characters`` of the text spoken: a speech result is bytes and reports nothing."""

    def report(_result: object) -> ReportedUsage:
        return ReportedUsage(
            input_tokens=0, output_tokens=0, total_tokens=0, units={"requests": 1}, quantities={"character": characters}
        )

    return report


def moderation_usage(_result: object) -> ReportedUsage:
    """One request, and no input token count: a moderation reports nothing it was billed on."""
    return ReportedUsage(output_tokens=0, units={"requests": 1})


def video_usage(_job: VideoJob) -> ReportedUsage:
    """No tokens and no count: a video is billed by its length when it completes, not when it starts."""
    return ReportedUsage(input_tokens=0, output_tokens=0, total_tokens=0)


@dataclass(frozen=True)
class PassthroughBilling:
    """How a pass-through endpoint is billed: what its result reports, and under which slot of a rate.

    ``meter`` and ``reports_tokens`` say what the endpoint can report before it
    is called, and ``slot`` and ``measures`` under which slot of the rate's
    block it bills and which units it measures a quantity in, which is what the
    ``require_pricing`` gate checks a rate against. ``bills_on_completion``
    says the result is a job priced when it completes, so a start that prices
    nothing is not a request served free. ``bills`` says in words what the
    endpoint bills, for the refusal of a rate in a unit it cannot bill on;
    ``None`` keeps the generic refusal.
    """

    report: Callable[[Any], ReportedUsage]
    meter: str | None
    reports_tokens: bool
    slot: Slot | None = None
    measures: frozenset[PriceUnit] = frozenset()
    bills_on_completion: bool = False
    bills: str | None = None


def image_billing(size: str | None) -> PassthroughBilling:
    """How an image request is billed: per image made, or per megapixel where ``size`` names the dimensions."""
    measures: set[PriceUnit] = {"image"}
    if image_megapixels(size) is not None:
        measures.add("megapixel")
    return PassthroughBilling(
        report=lambda result: image_usage(result, size=size),
        meter="images",
        reports_tokens=True,
        slot=("image", "output"),
        measures=frozenset(measures),
    )


def speech_billing(characters: int) -> PassthroughBilling:
    """How a speech request is billed: per character of the text spoken, or per request."""
    return PassthroughBilling(
        report=speech_usage(characters),
        meter="requests",
        reports_tokens=False,
        slot=("text", "input"),
        measures=frozenset({"character"}),
        bills="the characters of the text spoken and reports no tokens: price it per character or per request",
    )


EMBEDDINGS_BILLING = PassthroughBilling(report=embedding_usage, meter=None, reports_tokens=True)
RERANK_BILLING = PassthroughBilling(report=rerank_usage, meter="search_units", reports_tokens=True)
IMAGES_BILLING = image_billing(None)
TRANSCRIPTION_BILLING = PassthroughBilling(
    report=transcription_usage,
    meter="requests",
    reports_tokens=True,
    slot=("audio", "input"),
    measures=frozenset({"second", "minute"}),
)
MODERATION_BILLING = PassthroughBilling(
    report=moderation_usage, meter="requests", reports_tokens=False, slot=("text", "input")
)
VIDEO_BILLING = PassthroughBilling(report=video_usage, meter=None, reports_tokens=False, bills_on_completion=True)
