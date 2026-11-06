"""What each pass-through endpoint reads off its provider result as billable."""

from decimal import Decimal

import pytest
from any_llm.types.audio import Transcription
from any_llm.types.completion import CreateEmbeddingResponse, Embedding, Usage
from any_llm.types.image import Image, ImagesResponse
from any_llm.types.rerank import RerankMeta, RerankResponse, RerankResult, RerankUsage
from any_llm.types.video import VideoJob
from openai.types.audio.transcription import UsageDuration, UsageTokens
from openai.types.images_response import Usage as ImageUsage
from openai.types.images_response import UsageInputTokensDetails

from gateway.services.inference import (
    IMAGES_BILLING,
    ReportedUsage,
    embedding_usage,
    image_billing,
    image_megapixels,
    image_usage,
    moderation_usage,
    rerank_usage,
    speech_billing,
    speech_usage,
    transcription_usage,
    video_usage,
)


def test_an_embedding_reports_its_input_tokens() -> None:
    result = CreateEmbeddingResponse(
        data=[Embedding(embedding=[0.1], index=0, object="embedding")],
        model="m",
        object="list",
        usage=Usage(prompt_tokens=12, total_tokens=12),
    )

    assert embedding_usage(result) == ReportedUsage(input_tokens=12, output_tokens=0, total_tokens=12)


def _rerank(billed_units: dict[str, float] | None, tokens: int | None = None) -> RerankResponse:
    return RerankResponse(
        id="r",
        model="rerank-v3.5",
        results=[RerankResult(index=0, relevance_score=0.9)],
        meta=RerankMeta(billed_units=billed_units) if billed_units is not None else None,
        usage=RerankUsage(total_tokens=tokens) if tokens is not None else None,
    )


@pytest.mark.parametrize(
    ("billed", "units"),
    [
        ({"search_units": 3}, 3),
        ({"search_units": 2.2}, 3),
        ({"search_units": 0}, 1),
        ({"search_units": -4}, 1),
        ({"search_units": float("nan")}, 1),
        ({"search_units": float("inf")}, 1),
        ({"search_units": 5e9}, 1_000_000),
        ({}, 1),
        (None, 1),
    ],
)
def test_a_rerank_reports_the_search_units_billed_at_least_one(billed: dict[str, float] | None, units: int) -> None:
    assert rerank_usage(_rerank(billed)).units == {"search_units": units}


def test_a_rerank_reports_its_tokens_where_the_provider_does() -> None:
    reported = rerank_usage(_rerank(None, tokens=40))

    assert (reported.input_tokens, reported.output_tokens, reported.total_tokens) == (40, 0, 40)


def test_an_image_reports_the_images_returned() -> None:
    result = ImagesResponse(created=0, data=[Image(url="https://a"), Image(url="https://b")])

    assert image_usage(result) == ReportedUsage(
        input_tokens=0, output_tokens=0, total_tokens=0, units={"images": 2}, quantities={"image": 2}
    )


@pytest.mark.parametrize("data", [None, []])
def test_no_images_returned_reports_none(data: list[Image] | None) -> None:
    reported = image_usage(ImagesResponse(created=0, data=data), size="1024x1024")

    assert reported.units == {"images": 0}
    assert reported.quantities == {"image": 0, "megapixel": Decimal(0)}


@pytest.mark.parametrize(
    ("size", "megapixels"),
    [
        ("1024x1024", Decimal("1.048576")),
        ("1536x1024", Decimal("1.572864")),
        ("256X256", Decimal("0.065536")),
        ("auto", None),
        ("", None),
        (None, None),
        ("0x1024", None),
        ("landscape", None),
    ],
)
def test_a_size_names_its_megapixels_or_none(size: str | None, megapixels: Decimal | None) -> None:
    assert image_megapixels(size) == megapixels


def test_an_image_reports_the_megapixels_of_the_size_asked_for_times_the_count() -> None:
    result = ImagesResponse(created=0, data=[Image(url="https://a"), Image(url="https://b")])

    assert image_usage(result, size="1024x1024").quantities == {"image": 2, "megapixel": Decimal("2.097152")}


def test_an_image_reports_the_size_the_provider_says_over_the_one_asked_for() -> None:
    result = ImagesResponse(created=0, data=[Image(url="https://a")], size="1536x1024")

    assert image_usage(result, size="1024x1024").quantities == {"image": 1, "megapixel": Decimal("1.572864")}


def test_image_billing_measures_megapixels_only_where_the_size_names_them() -> None:
    assert image_billing("1024x1024").measures == frozenset({"image", "megapixel"})
    assert image_billing("auto").measures == frozenset({"image"})
    assert IMAGES_BILLING.measures == frozenset({"image"})
    assert IMAGES_BILLING.slot == ("image", "output")


def test_a_token_billed_image_reports_its_tokens_beside_the_count() -> None:
    result = ImagesResponse(
        created=0,
        data=[Image(b64_json="aGk=")],
        usage=ImageUsage(
            input_tokens=50,
            output_tokens=4160,
            total_tokens=4210,
            input_tokens_details=UsageInputTokensDetails(image_tokens=0, text_tokens=50),
        ),
    )

    assert image_usage(result) == ReportedUsage(
        input_tokens=50, output_tokens=4160, total_tokens=4210, units={"images": 1}, quantities={"image": 1}
    )


def test_a_token_billed_transcription_reports_its_tokens_and_one_request() -> None:
    result = Transcription(
        text="hi", usage=UsageTokens(input_tokens=1000, output_tokens=100, total_tokens=1100, type="tokens")
    )

    assert transcription_usage(result) == ReportedUsage(
        input_tokens=1000, output_tokens=100, total_tokens=1100, units={"requests": 1}
    )


def test_a_transcription_without_usage_reports_one_request() -> None:
    assert transcription_usage(Transcription(text="hi", usage=None)) == ReportedUsage(
        input_tokens=0, output_tokens=0, total_tokens=0, units={"requests": 1}
    )


def test_a_transcription_reports_the_length_the_provider_says_in_seconds_and_minutes() -> None:
    result = Transcription(text="hi", usage=UsageDuration(seconds=90.5, type="duration"))

    assert transcription_usage(result) == ReportedUsage(
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        units={"requests": 1},
        quantities={"second": Decimal("90.5"), "minute": Decimal("90.5") / Decimal(60)},
    )


def test_speech_reports_the_characters_spoken_and_one_request() -> None:
    assert speech_usage(12)(b"audio") == ReportedUsage(
        input_tokens=0, output_tokens=0, total_tokens=0, units={"requests": 1}, quantities={"character": 12}
    )
    assert speech_billing(12).measures == frozenset({"character"})
    assert speech_billing(12).slot == ("text", "input")


def test_a_moderation_reports_one_request() -> None:
    assert moderation_usage(object()) == ReportedUsage(output_tokens=0, units={"requests": 1})


def test_a_video_job_reports_nothing_billable_when_it_starts() -> None:
    job = VideoJob(id="video_1", model="veo-3.0-generate-001", status="queued")

    assert video_usage(job) == ReportedUsage(input_tokens=0, output_tokens=0, total_tokens=0)
