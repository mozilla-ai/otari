"""Which of a usage breakdown's model ids are one model, and that model's name.

Providers spell one model differently (``accounts/fireworks/models/deepseek-v4p1-flash-1017``,
``deepseek-ai/DeepSeek-V4.1-Flash-1017``) and usage records the spelling a request
was sent as. A breakdown keeps a row per id, since a model filter matches the id;
this gives each row the model it belongs to, so a client can show one row per model.

Grouping errs toward a row too many: two models under one row would put one
model's spend on another, while a model split in two only costs a row.
"""

import re
import threading

from gateway.schemas.usage_and_telemetry import UsageGroupRow
from gateway.services.catalog import SelectorIndex, current_selector_index
from gateway.services.model_identity import identity_key, name_from_id, org_vendor

# A path that names where a public model is served rather than whose it is:
# Fireworks' own catalog, and nano-gpt's confidential-compute variants. Another
# Fireworks account's path is that customer's own model, a fine-tune say. Case
# folds here as it does in every lookup below.
_PUBLIC_PATH = re.compile(r"^(?:accounts/fireworks/(?:models|routers)/|TEE/)", re.IGNORECASE)

_lock = threading.Lock()
_catalog: tuple[SelectorIndex, "_CatalogMaps"] | None = None

_Model = tuple[str, str]
"""A catalog ID and its display name."""


class _CatalogMaps:
    """The deployment's catalog read two ways: by the id an offering serves, and by that id's own key."""

    def __init__(self, index: SelectorIndex) -> None:
        self.by_id: dict[str, _Model | None] = {}
        self.by_key: dict[str, set[_Model]] = {}
        for slug, offerings in index.offerings.items():
            model = (slug, index.names.get(slug) or slug)
            for selector in offerings:
                served = selector.partition(":")[2]
                found = self.by_id.get(served.lower(), model)
                self.by_id[served.lower()] = model if found == model else None
                if not _names_an_owner(served):
                    self.by_key.setdefault(_own_identity(served)[0], set()).add(model)


def _catalog_maps() -> _CatalogMaps:
    """The maps for the index in force, rebuilt when the index is.

    An id is looked up across every offering the deployment serves, since usage
    records it without the instance it went to, and case-insensitively, since
    providers and clients disagree on case for one id. An id two catalog models
    share answers neither. Organizations' own offerings are not read: an id only
    an organization's key serves is grouped from the id alone.
    """
    global _catalog
    index = current_selector_index()
    with _lock:
        if _catalog is None or _catalog[0] is not index:
            _catalog = (index, _CatalogMaps(index))
        return _catalog[1]


def _names_an_owner(model_id: str) -> bool:
    """Whether an id names whose deployment, profile or account it is, beyond the model it serves.

    ``…#accounts/acme/deployments/k3x9a1``,
    ``arn:aws:bedrock:…:application-inference-profile/k8x2`` and another
    account's ``accounts/acme/models/glm-5p3`` say whose they are, and the
    catalog's id rung would keep only their last segment.
    """
    plain = _PUBLIC_PATH.sub("", model_id)
    return "#" in plain or plain.startswith("arn:") or plain.count("/") > 1


def _own_identity(model_id: str) -> tuple[str, str]:
    """The group key and name an id carries on its own, for an id the catalog does not serve.

    The catalog's id rung reads ``vendor/model`` and a public path in front of it,
    keeping the model and dropping the rest. An id that names an owner
    (:func:`_names_an_owner`), or whose org segment is no vendor the tables know
    (``acme/Llama-3.1-8B-Instruct``), may be someone's own model, so it is its
    own group, named as written. Case still folds, as it does for every key.
    """
    plain = _PUBLIC_PATH.sub("", model_id)
    org, slash, _ = plain.rpartition("/")
    if _names_an_owner(model_id) or (slash and org_vendor(org) is None):
        return f"raw:{model_id.lower()}", model_id
    # Read without the public path, which this module matches in any case.
    return identity_key("", plain, None), name_from_id(plain)


def grouped_models(rows: list[UsageGroupRow]) -> list[UsageGroupRow]:
    """Each model row with ``model_group`` and ``label``, the same on every row of one model.

    An id the catalog serves belongs to its catalog model and takes its name,
    unless the id names an owner (:func:`_names_an_owner`). An
    id it does not serve joins the one catalog model whose ids carry the same key
    (:func:`_own_identity`), which is how a spelling the deployment no longer
    serves stays with the model; where two catalog models' ids carry that key, it
    joins neither. The rest group by that key alone, under the least of their
    names, so a model's name does not change with which spelling was busiest.
    The fold and a row with no id are left as they are.
    """
    catalog = _catalog_maps()
    ids = [row.key for row in rows if row.key is not None]
    own = {model_id: _own_identity(model_id) for model_id in ids}
    # An id that names an owner is kept out of the catalog too, which may file
    # another account's model under the public one it was tuned from.
    known = {
        model_id: model
        for model_id in ids
        if not _names_an_owner(model_id) and (model := catalog.by_id.get(model_id.lower())) is not None
    }

    groups: dict[str, _Model] = {}
    names_by_key: dict[str, list[str]] = {}
    for model_id in ids:
        key, name = own[model_id]
        # Whether a key is one model's is the whole catalog's answer, not this window's.
        candidates = catalog.by_key.get(key, set())
        model = known.get(model_id) or (next(iter(candidates)) if len(candidates) == 1 else None)
        if model is not None:
            groups[model_id] = model
        else:
            names_by_key.setdefault(key, []).append(name)
    for model_id in ids:
        key = own[model_id][0]
        if model_id not in groups:
            groups[model_id] = (f"id:{key}", min(names_by_key[key]))

    return [
        row
        if row.key is None
        else row.model_copy(update={"model_group": groups[row.key][0], "label": groups[row.key][1]})
        for row in rows
    ]
