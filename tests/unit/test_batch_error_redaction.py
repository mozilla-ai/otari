import pytest
from any_llm.types.batch import BatchResultError

from gateway.api.routes.batches import _public_batch_error
from gateway.services.upstream_redaction import PASSTHROUGH_PROVIDER_ERROR_DETAIL


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("invalid_request_error", "Request body is invalid"),
        ("model_not_found", "The requested model was not found"),
        ("unprocessable_entity", "The request cannot be processed"),
        ("rate_limit", "Quota exceeded; retry after 60 seconds"),
        ("rate_limit_error", "Quota exceeded; retry after 60 seconds"),
    ],
)
def test_actionable_batch_errors_return_redacted_provider_text(code: str, message: str) -> None:
    raw = f"{message}. Request to https://provider.example failed with key sk-proj-AAAAAAAAAAAAAAAA"

    result = _public_batch_error(BatchResultError(code=code, message=raw))

    assert result["code"] == code
    assert "https://provider.example" not in result["message"]
    assert "sk-proj-AAAAAAAAAAAAAAAA" not in result["message"]


@pytest.mark.parametrize(
    "code",
    [
        "400",
        "404",
        "422",
        "429",
        "invalid_api_key",
        "insufficient_quota",
        "billing_hard_limit_reached",
        "server_error",
        "unknown",
    ],
)
def test_non_actionable_batch_errors_use_fixed_public_text(code: str) -> None:
    result = _public_batch_error(BatchResultError(code=code, message="provider internal details"))

    assert result == {"code": code, "message": PASSTHROUGH_PROVIDER_ERROR_DETAIL}


@pytest.mark.parametrize("code", ["rate_limit", "rate_limit_error"])
def test_rate_limit_without_quota_or_retry_guidance_uses_fixed_public_text(code: str) -> None:
    result = _public_batch_error(BatchResultError(code=code, message="Provider account authentication failed"))

    assert result == {"code": code, "message": PASSTHROUGH_PROVIDER_ERROR_DETAIL}
