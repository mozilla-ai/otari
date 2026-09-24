"""Product feedback: the send limit. Delivery is ``FeedbackDeliveryPort``."""

from gateway.services.feedback._service import new_feedback_rate_limiter

__all__ = ["new_feedback_rate_limiter"]
