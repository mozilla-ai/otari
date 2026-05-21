"""Moderation types.

Gateway-local shim that prefers ``any_llm.types.moderation`` when available
and falls back to locally defined Pydantic models when the installed
``any-llm-sdk`` does not yet ship moderation support.

This keeps the gateway importable against older SDK releases while
preserving the OpenAI-compatible wire contract.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _LocalModerationResult(BaseModel):
    """Single moderation result (one per input item).

    Mirrors the OpenAI moderation result shape. Extra provider-specific
    fields are preserved via ``model_config`` to avoid data loss.
    """

    model_config = ConfigDict(extra="allow")

    flagged: bool
    categories: dict[str, bool] = Field(default_factory=dict)
    category_scores: dict[str, float] = Field(default_factory=dict)
    category_applied_input_types: dict[str, list[str]] | None = None


class _LocalModerationResponse(BaseModel):
    """OpenAI-compatible moderation response envelope."""

    model_config = ConfigDict(extra="allow")

    id: str
    model: str
    results: list[_LocalModerationResult]
    raw: dict[str, Any] | None = None


# Prefer SDK types when available; fall back to local shims otherwise.
try:
    from any_llm.types.moderation import ModerationResponse, ModerationResult
except ImportError:
    ModerationResult = _LocalModerationResult  # type: ignore[misc,assignment]
    ModerationResponse = _LocalModerationResponse  # type: ignore[misc,assignment]


__all__ = ["ModerationResponse", "ModerationResult"]
