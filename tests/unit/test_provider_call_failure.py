"""The answer to a failed provider call on a job's lifecycle: a classified status, a 424, else a 502."""

import httpx
import openai
from any_llm.utils.exception_handler import convert_exception

from gateway.api.routes._passthrough import provider_call_failure
from gateway.api.routes._pipeline import PROVIDER_CREDENTIALS_DETAIL
from gateway.core.error_codes import ERROR_CODE_HEADER, PROVIDER_CREDENTIAL_REJECTED

_REQUEST = httpx.Request("GET", "https://api.openai.com/v1/videos/video_abc")


def _upstream(error: type[openai.APIStatusError], status_code: int, message: str) -> Exception:
    return convert_exception(
        error(message, response=httpx.Response(status_code, request=_REQUEST), body=None), "openai"
    )


def test_a_rate_limit_answers_429_with_the_providers_sentence() -> None:
    answer = provider_call_failure(_upstream(openai.RateLimitError, 429, "Error code: 429 - slow down"))

    assert answer.status_code == 429
    assert "slow down" in str(answer.detail)


def test_a_rejected_request_answers_400() -> None:
    answer = provider_call_failure(
        _upstream(openai.BadRequestError, 400, "Error code: 400 - seconds must be 4, 8 or 12")
    )

    assert answer.status_code == 400
    assert "seconds must be" in str(answer.detail)


def test_a_rejected_credential_answers_424_without_the_key() -> None:
    answer = provider_call_failure(
        _upstream(openai.AuthenticationError, 401, "Error code: 401 - Incorrect API key provided: sk-proj-AAAA")
    )

    assert answer.status_code == 424
    assert answer.detail == PROVIDER_CREDENTIALS_DETAIL
    assert (answer.headers or {})[ERROR_CODE_HEADER] == PROVIDER_CREDENTIAL_REJECTED


def test_a_failure_the_classifier_cannot_name_answers_502() -> None:
    answer = provider_call_failure(RuntimeError("socket closed"))

    assert answer.status_code == 502
    assert answer.detail == "LLM provider error"
