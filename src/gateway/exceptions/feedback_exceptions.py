"""Safe feedback delivery failures."""


class FeedbackDeliveryError(Exception):
    def __init__(self, status_code: int = 503, retry_after: int | None = None) -> None:
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__("Feedback delivery could not be confirmed")
