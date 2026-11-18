"""Contact: the address a reply may go to, and how often one caller sends."""

from gateway.services.feedback._service import FeedbackService, new_feedback_rate_limiter

__all__ = ["FeedbackService", "new_feedback_rate_limiter"]
