from typing import Any

import pytest
from any_llm.types.batch import BatchResultError

from gateway.core.error_details import ProviderDetail
from gateway.services.inference import BatchLineError

_SECRET = "sk-proj-AAAAAAAAAAAAAAAAAAAAAAAA"


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("invalid_request_error", "Request body is invalid"),
        ("model_not_found", "The requested model was not found"),
        ("unprocessable_entity", "The request cannot be processed"),
        ("rate_limit_error", "You exceeded your per-minute rate limit"),
        ("400", "Request body is invalid"),
    ],
)
def test_caller_actionable_line_errors_pass_redacted_provider_text(code: str, message: str) -> None:
    raw = f"{message}. Request to https://provider.example failed with key {_SECRET}"

    result = BatchLineError.of(BatchResultError(code=code, message=raw))

    assert result.public["code"] == code
    assert message in result.public["message"]
    assert _SECRET not in result.public["message"]


@pytest.mark.parametrize(
    ("code", "message", "detail"),
    [
        ("authentication_error", f"invalid x-api-key {_SECRET}", ProviderDetail.CREDENTIALS),
        ("server_error", "provider internal details", ProviderDetail.ERROR),
        ("unknown", "provider internal details", ProviderDetail.ERROR),
        (None, None, ProviderDetail.ERROR),
    ],
)
def test_gateway_side_line_errors_use_fixed_public_text(code: str | None, message: str | None, detail: str) -> None:
    result = BatchLineError.of(BatchResultError(code=code, message=message))  # type: ignore[arg-type]

    assert result.public["message"] == detail


def test_an_exhausted_provider_account_reads_as_billing_not_a_bad_request() -> None:
    error = BatchResultError(
        code="invalid_request_error",
        message="Your credit balance is too low to access the Anthropic API.",
    )

    result = BatchLineError.of(error)

    assert result.public["message"] == ProviderDetail.BILLING


def test_a_numeric_error_code_is_classified_as_its_http_status() -> None:
    # Bedrock reports a line's errorCode as a number, which the adapter passes through unconverted.
    code: Any = 429

    result = BatchLineError.of(BatchResultError(code=code, message="Too many tokens, please wait"))

    assert result.public == {"code": "429", "message": "Too many tokens, please wait"}
    assert result.logged_status == 429
