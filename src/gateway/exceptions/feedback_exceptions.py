"""Safe feedback delivery failures, and the refusals of an email to share."""

from fastapi import status

from gateway.exceptions import TenancyValidationError

UNSHAREABLE_EMAIL = "Your email could not be attached. Untick it to send without it."
ANOTHER_ACCOUNT = "You are now signed in as another account. Reload the page to share that email."


class FeedbackDeliveryError(Exception):
    def __init__(self, status_code: int = 503, retry_after: int | None = None) -> None:
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__("Feedback delivery could not be confirmed")


class UnshareableEmailError(TenancyValidationError):
    """The caller has no email the receiver would accept as an address to reply to."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT

    def __init__(self) -> None:
        super().__init__(UNSHAREABLE_EMAIL)


class AnotherAccountError(TenancyValidationError):
    """The form showed another address than the session's, as a page loaded before another tab signed in does."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT

    def __init__(self) -> None:
        super().__init__(ANOTHER_ACCOUNT)
