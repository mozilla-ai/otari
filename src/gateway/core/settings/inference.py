"""Inference request settings."""

from typing import Annotated

from pydantic import BaseModel, Field

from gateway.core.settings_view import OMITTED


class InferenceSettings(BaseModel):
    """How a completion request that carries an ``Idempotency-Key`` is deduplicated."""

    idempotency_retention_sec: Annotated[int, OMITTED] = Field(
        default=86400,
        ge=0,
        description=(
            "How long a non-streaming completion sent with an Idempotency-Key is kept, so that a "
            "retry with the same key returns the stored response instead of calling the provider "
            "and billing again. The stored response includes the generated content. 0 ignores the "
            "header. Standalone mode only."
        ),
    )
    idempotency_lease_sec: Annotated[int, OMITTED] = Field(
        default=60,
        ge=3,
        description=(
            "How long a claim on an Idempotency-Key stays valid without being renewed. The request "
            "holding it renews it every third of this while it runs, so this bounds how long a key "
            "stays blocked after the worker running its request dies, not how long a request may take."
        ),
    )
    idempotency_sweep_interval_sec: Annotated[int, OMITTED] = Field(
        default=3600,
        ge=1,
        description="How often expired idempotency records are deleted.",
    )
