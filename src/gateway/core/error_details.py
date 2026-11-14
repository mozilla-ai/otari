"""Fixed wordings Otari writes itself for a refusal, as against a message it relays."""

from enum import StrEnum


class ProviderDetail(StrEnum):
    """Fixed wordings Otari writes itself for a provider failure.

    These are client-visible API contract values, so a change to one needs a
    deprecation plan. Some are the whole answer, for a failure that is the
    deployment's own and carries no remedy the caller could apply. The rest stand
    in for a message the provider did not send on a refusal the caller could act
    on, which the provider otherwise words itself, redacted and length-capped.

    NOTE: This is not the whole inventory. Otari writes other fixed wordings at
    their point of use, and builds the ones that name a parameter or a status.
    """

    ACCOUNT_QUOTA = (
        "The upstream provider account is over one of its service quotas. "
        "Raise the quota, or route this model to another provider."
    )
    BAD_REQUEST = "The provider rejected the request as invalid (check the model name and parameters)"
    BILLING = (
        "The upstream provider account is out of credit or over its billing limit. "
        "Top up the provider account, or route this model to a provider that has credit."
    )
    CREDENTIALS = "The provider rejected the gateway's credentials"
    ERROR = "LLM provider error"
    MODEL_NOT_FOUND = "The requested model was not found on the provider"
    RATE_LIMITED = "The provider rate-limited this request"
    REQUEST_TOO_LARGE = "The provider refused the request as too large (reduce the request size)"
    STREAM_INTERRUPTED = "The provider's stream ended before it finished"
    TIMEOUT = "LLM provider timeout"
