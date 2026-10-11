"""Errors that the inference domain may raise, each carrying the status it renders as."""

from gateway.exceptions import TenancyValidationError


class EmbeddingOptionUnsupportedError(TenancyValidationError):
    """An embedding option was sent for a provider whose embedding call does not read it.

    Refused rather than dropped, because the embedding would otherwise come back
    as if the option had not been sent.
    """

    def __init__(self, option: str, selector: str) -> None:
        # The selector the caller sent, never the resolved provider, so an alias
        # does not reveal its target.
        super().__init__(f"'{option}' is not supported for model '{selector}'")
