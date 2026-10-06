"""Which priced models the hosted-providers surface is responsible for.

A stored price is what lists a hosted model: the catalog lists a model it did
not discover only when a ``model_pricing`` row names it. So the deployment
price list *is* the hosted catalog, and a key nothing here offers is a model no
operator chose. This module decides, for each priced key, whether the surface
keeps it, and says why when it keeps one it does not offer.

Pure, so the rules are testable without a database or a provider.
"""

from collections.abc import Sequence
from typing import Literal, NamedTuple

from gateway.schemas.providers import HostedCatalogKeptGroupPublic
from gateway.services.pricing_service import GATEWAY_TOOL_PRICING_PROVIDER
from gateway.services.provider_kwargs import split_selector

# Why a sweep keeps a priced model it cannot attribute to the surface. Sent to
# the operator verbatim: this is an operator surface with no translation layer,
# and the reason is the whole value of showing the group at all.
KEEP_GATEWAY_TOOL = "prices a gateway-run tool, not a model"
KEEP_SEARCH_TOOL = "prices a configured search tool, not a model"
KEEP_DEPLOYMENT_INSTANCE = "served by a provider instance configured on this deployment"
KEEP_UNATTRIBUTABLE = "no provider prefix to attribute it to"

# How many model names a sweep's response carries per group. The counts are
# exact; the lists are a sample.
CATALOG_SAMPLE = 50

Verdict = Literal["offered", "kept", "removed"]


class CatalogVerdict(NamedTuple):
    """What becomes of one key in the deployment price list, and why.

    ``offered`` and ``kept`` both survive, and they are separate because the
    surface reports them differently: an offered model is the catalog working,
    while a kept one looks like drift and needs its ``reason`` shown.
    ``reason`` is set only for ``kept``.
    """

    model_key: str
    versions: int
    verdict: Verdict
    reason: str | None = None
    canonical_key: str = ""
    """``model_key`` with the legacy ``provider/model`` spelling folded onto ``provider:model``.

    Carried because the two pricing tables are keyed differently: ``model_pricing``
    holds whatever spelling was written, while an override goes through
    ``normalize_pricing_key`` before it is stored, so deleting a slash-spelled
    price by its own key would leave its colon-spelled override orphaned.
    """


def _keep_reason(
    prefix: str | None,
    *,
    deployment_instances: frozenset[str],
    search_providers: frozenset[str],
) -> str | None:
    """What keeps an unoffered key on the catalog, or None when nothing does."""
    if prefix is None:
        return KEEP_UNATTRIBUTABLE
    if prefix == GATEWAY_TOOL_PRICING_PROVIDER:
        return KEEP_GATEWAY_TOOL
    if prefix in search_providers:
        return KEEP_SEARCH_TOOL
    if prefix in deployment_instances:
        return KEEP_DEPLOYMENT_INSTANCE
    return None


def classify_priced_keys(
    priced: Sequence[tuple[str, int]],
    *,
    offered: frozenset[str],
    deployment_instances: frozenset[str],
    search_providers: frozenset[str] = frozenset(),
) -> list[CatalogVerdict]:
    """Decide, for each priced key, whether the hosted-providers surface is what lists it.

    Order is the safety rule rather than a style choice. A key the surface
    offers is kept whatever else also serves it. Only what is left over goes: a
    model under a provider the surface does not offer it on, or under a
    provider nothing on this deployment serves at all. An organization holding
    its own key for the provider keeps nothing here: its models are listed from
    that key's own roster and priced from its own override first. The override
    itself is spared by the repository, not by this classification.

    **Not every rate on the list prices a model.** A gateway-run tool is keyed
    under the reserved ``otari`` provider, and a configured search tool is keyed
    ``<provider>:<tool>`` off ``config.search_tools`` rather than
    ``config.providers``, so ``search_providers`` has to be passed or a
    deployment with ``search_tools:`` configured loses the rate ``POST
    /v1/search`` reserves against.

    **The one gap this cannot close.** A provider credentialed from the
    environment alone, with no ``providers:`` entry and no hosted provider, is
    invisible to every set below. An operator-set rate for one of those
    classifies as removable. The preview covers it: every key is named before
    anything is deleted, so a rate somebody set by hand can be seen and the
    sweep cancelled.
    """
    verdicts: list[CatalogVerdict] = []
    for model_key, versions in priced:
        # The legacy ``provider/model`` spelling names the same model as the
        # roster's ``provider:model``, so it is folded before the roster is
        # consulted.
        split = split_selector(model_key)
        prefix = None if split is None else split[0]
        canonical = model_key if split is None else f"{split[0]}:{split[1]}"
        if canonical in offered:
            verdicts.append(CatalogVerdict(model_key, versions, "offered", None, canonical))
            continue
        reason = _keep_reason(prefix, deployment_instances=deployment_instances, search_providers=search_providers)
        verdict: Verdict = "removed" if reason is None else "kept"
        verdicts.append(CatalogVerdict(model_key, versions, verdict, reason, canonical))
    return verdicts


def kept_groups(verdicts: Sequence[CatalogVerdict]) -> list[HostedCatalogKeptGroupPublic]:
    """Fold the kept verdicts into one group per reason, largest first.

    Only the kept ones: an offered model is the catalog working as configured,
    and listing every one of those back would bury the few that need reading.
    """
    by_reason: dict[str, list[str]] = {}
    for verdict in verdicts:
        if verdict.verdict == "kept" and verdict.reason is not None:
            by_reason.setdefault(verdict.reason, []).append(verdict.model_key)
    return [
        HostedCatalogKeptGroupPublic(reason=reason, count=len(keys), models=sorted(keys)[:CATALOG_SAMPLE])
        for reason, keys in sorted(by_reason.items(), key=lambda item: (-len(item[1]), item[0]))
    ]


__all__ = [
    "CATALOG_SAMPLE",
    "KEEP_DEPLOYMENT_INSTANCE",
    "KEEP_GATEWAY_TOOL",
    "KEEP_SEARCH_TOOL",
    "KEEP_UNATTRIBUTABLE",
    "CatalogVerdict",
    "classify_priced_keys",
    "kept_groups",
]
