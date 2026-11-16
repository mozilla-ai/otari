"""What the caller and the usage log see of one failed batch line."""

from dataclasses import dataclass
from typing import Self

from any_llm.types.batch import BatchResultError

from gateway.core.error_details import ProviderDetail
from gateway.services.inference._provider_errors import UpstreamFailure
from gateway.services.upstream_redaction import redact_upstream_message

# A batch line reports its failure as a provider error code rather than an HTTP
# status. These are the codes whose status is known, so the line is classified
# by the same rules as a synchronous failure with that status.
_STATUS_BY_CODE = {
    "invalid_argument": 400,
    "invalid_parameter": 400,
    "invalid_prompt": 400,
    "invalid_request": 400,
    "invalid_request_error": 400,
    "invalid_value": 400,
    "unsupported_value": 400,
    "context_length_exceeded": 400,
    "billing_hard_limit_reached": 400,
    "insufficient_quota": 429,
    "authentication_error": 401,
    "permission_error": 403,
    "model_not_found": 404,
    "not_found": 404,
    "not_found_error": 404,
    "unprocessable_entity": 422,
    "validation_error": 422,
    "rate_limit": 429,
    "rate_limit_error": 429,
    "rate_limit_exceeded": 429,
    "rate_limit_reached": 429,
    "too_many_requests": 429,
}


class _BatchLineFailure(Exception):
    """A failed batch line, shaped like the upstream exception a synchronous call would raise."""

    def __init__(self, status_code: int | None, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _status_of(code: str) -> int | None:
    """The HTTP status a batch error code stands for, or ``None`` when it names none."""
    normalized = code.strip().casefold()
    if normalized.isdigit() and 400 <= int(normalized) <= 599:
        return int(normalized)
    return _STATUS_BY_CODE.get(normalized)


@dataclass(frozen=True)
class BatchLineError:
    """One failed batch line, classified with the synchronous routes' taxonomy."""

    public: dict[str, str]
    logged_status: int

    @classmethod
    def of(cls, error: BatchResultError) -> Self:
        """Classify ``error``.

        Provider adapters fill ``BatchResultError`` without validation, so a
        numeric code (Bedrock) or a missing message is coerced to text first.
        """
        raw_code = "" if error.code is None else str(error.code)
        message = "" if error.message is None else str(error.message)
        failure = UpstreamFailure(_BatchLineFailure(_status_of(raw_code), message))
        mapping = failure.mapping
        return cls(
            public={
                "code": redact_upstream_message(raw_code) or "provider_error",
                "message": mapping.detail if mapping is not None else ProviderDetail.ERROR,
            },
            logged_status=failure.logged_status,
        )
