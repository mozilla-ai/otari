"""Data access for the inference domain's own rows."""

from gateway.repositories.inference.idempotency_repository import IdempotencyRepository
from gateway.repositories.inference.inference_repositories import InferenceRepositories

__all__ = ["IdempotencyRepository", "InferenceRepositories"]
