"""Reduce a models.dev ``api.json`` to the fields the price index reads.

Standard library only: ``scripts/update_models_dev_snapshot.py`` loads this file
by path to build the bundled snapshot with the same rules the gateway applies to
every snapshot it stores.
"""

from typing import Any

_COST_FIELDS = ("input", "output", "cache_read", "cache_write", "tiers", "context_over_200k", "per_image")


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


def _flat_image_price(model: dict[str, Any]) -> object:
    """The v2 price of one generated image, or ``None`` when it is not one flat amount.

    A price that varies by request option (an ``overrides`` list, such as a 4K
    tier) is left out: the gateway cannot tell which option a request used, and
    the base amount would undercharge the dearer ones.
    """
    if model.get("type") != "image":
        return None
    cost = model.get("cost")
    image = cost.get("image") if isinstance(cost, dict) else None
    output = image.get("output") if isinstance(image, dict) else None
    if not isinstance(output, dict) or output.get("unit") != "image" or "overrides" in output:
        return None
    price = output.get("price")
    return price if isinstance(price, (int, float)) and not isinstance(price, bool) else None


def merge_image_prices(catalog: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    """``catalog`` (a v1 ``api.json``) with each flat per-image price of ``v2`` as ``cost.per_image``.

    v1 leaves image models out, so a model only v2 lists is added under its
    provider, and a provider only v2 lists is added too.
    """
    merged = dict(catalog)
    for key, provider in v2.items():
        if not isinstance(provider, dict) or not isinstance(provider.get("models"), dict):
            continue
        prices = {
            model_key: (model, price)
            for model_key, model in provider["models"].items()
            if isinstance(model, dict) and (price := _flat_image_price(model)) is not None
        }
        if not prices:
            continue
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(current.get("models"), dict):
            target = {**current, "models": dict(current["models"])}
        else:
            target = {field: provider[field] for field in ("id", "name", "doc") if field in provider}
            target["models"] = {}
        merged[key] = target
        for model_key, (model, price) in prices.items():
            existing = target["models"].get(model_key)
            entry = dict(existing) if isinstance(existing, dict) else {}
            for field, source in (("id", "id"), ("name", "name"), ("canonical_model_id", "canonical_id")):
                if field not in entry and isinstance(model.get(source), str):
                    entry[field] = model[source]
            cost = entry.get("cost")
            entry["cost"] = {**(cost if isinstance(cost, dict) else {}), "per_image": price}
            target["models"][model_key] = entry
    return merged


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
