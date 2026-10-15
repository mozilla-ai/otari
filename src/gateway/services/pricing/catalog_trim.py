"""Reduce a models.dev ``api.json`` to the fields the price index reads.

Standard library only: ``scripts/update_models_dev_snapshot.py`` loads this file
by path to build the bundled snapshot with the same rules the gateway applies to
every snapshot it stores.
"""

from typing import Any

_COST_FIELDS = ("input", "output", "cache_read", "cache_write", "tiers", "context_over_200k")


def _trim_model(key: str, model: dict[str, Any]) -> dict[str, Any]:
    trimmed: dict[str, Any] = {}
    if isinstance(model.get("id"), str) and model["id"] != key:
        trimmed["id"] = model["id"]
    for field in ("name", "canonical_model_id", "status"):
        if isinstance(model.get(field), str) and model[field]:
            trimmed[field] = model[field]
    cost = model.get("cost")
    if isinstance(cost, dict):
        kept = {name: cost[name] for name in _COST_FIELDS if name in cost}
        if kept:
            trimmed["cost"] = kept
    limit = model.get("limit")
    if isinstance(limit, dict) and isinstance(limit.get("context"), int) and not isinstance(limit["context"], bool):
        trimmed["limit"] = {"context": limit["context"]}
    return trimmed


def trim_catalog(raw: dict[str, Any]) -> dict[str, Any]:
    """The catalog with only provider id, name and doc, and per-model price fields."""
    trimmed: dict[str, Any] = {}
    for key, provider in raw.items():
        if not isinstance(provider, dict) or not isinstance(provider.get("models"), dict):
            continue
        entry: dict[str, Any] = {"id": provider["id"] if isinstance(provider.get("id"), str) else key}
        for field in ("name", "doc"):
            if isinstance(provider.get(field), str) and provider[field]:
                entry[field] = provider[field]
        entry["models"] = {
            str(model_key): _trim_model(str(model_key), model)
            for model_key, model in provider["models"].items()
            if isinstance(model, dict)
        }
        trimmed[str(key)] = entry
    return trimmed
