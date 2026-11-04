"""Check the address a reply may go to, and limit how often one caller sends."""

from gateway.core.addresses import normalized_address
from gateway.exceptions.feedback_exceptions import AnotherAccountError, UnshareableEmailError
from gateway.rate_limit import RateLimiter

# Each send is a real POST upstream, and the receiver's own limit is shared by
# everyone behind this gateway's address, so one caller must not spend it.
SENDS_PER_WINDOW = 5
WINDOW_SEC = 10 * 60
# The receiver's own cap on `reply_to`; keep the two together.
MAX_REPLY_TO_LENGTH = 254


def new_feedback_rate_limiter() -> RateLimiter:
    """A fresh per-caller send limit, kept on ``app.state.feedback_rate_limiter``."""
    return RateLimiter(SENDS_PER_WINDOW, window_sec=WINDOW_SEC)


class FeedbackService:
    def reply_to(self, shown: str | None, session_email: str | None) -> str | None:
        """The session's own email to send, and only when it is the address the form showed.

        Raises ``UnshareableEmailError`` when the receiver would refuse the session's
        address, and ``AnotherAccountError`` when the form showed a different one.
        """
        if shown is None:
            return None
        email = (session_email or "").strip()
        if len(email) > MAX_REPLY_TO_LENGTH or not email.isprintable() or normalized_address(email) is None:
            raise UnshareableEmailError
        if normalized_address(shown) != normalized_address(email):
            raise AnotherAccountError
        return email
