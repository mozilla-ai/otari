"""Product feedback forwarding."""

from gateway.services.feedback._service import FeedbackService, new_feedback_rate_limiter

__all__ = ["FeedbackService", "new_feedback_rate_limiter"]
