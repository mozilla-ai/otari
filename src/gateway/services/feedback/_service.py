"""The per-caller limit on feedback sends."""

from gateway.rate_limit import RateLimiter

# Each send is a real delivery, and the receiver's own limit is shared by
# everyone behind this gateway's address, so one caller must not spend it.
SENDS_PER_WINDOW = 5
WINDOW_SEC = 10 * 60


def new_feedback_rate_limiter() -> RateLimiter:
    """A fresh per-caller send limit, kept on ``app.state.feedback_rate_limiter``."""
    return RateLimiter(SENDS_PER_WINDOW, window_sec=WINDOW_SEC)
