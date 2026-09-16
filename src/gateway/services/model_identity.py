"""Which provider offerings are the same model.

The catalog keys everything on a selector, ``instance:model``, and nothing in it
knows that ``fireworks:accounts/fireworks/models/glm-5p3`` and
``nebius:zai-org/GLM-5.3`` are one model served twice. This module is that
knowledge: a pure function from what a provider calls a model to a grouping key,
and a vote over the group for the name and vendor to show.

Two rungs, and the order matters. models.dev's display ``name`` is the primary
key because its maintainers have already normalized the provider spellings
("GLM 5.3", "GLM-5.3" and "GLM5.3" all name one thing); the provider's own id is
the fallback, cleaned of the prefixes and suffixes that are about the serving
rather than the model. Measured over the whole models.dev catalog the id rung
agrees with the name rung on three entries in four, and the disagreements are
resellers whose names drop a size or date the id keeps, which is why it is the
fallback and not the rule.

The normalizer keeps digits and words and folds only punctuation and case, so a
dated build (``deepseek-v4-pro-0813``), a size or tier (``glm-5.3-flash``) and a
mode a reseller exposes as its own id stay separate models. Over-grouping would
put two prices under one header; under-grouping only costs a singleton, and the
raw selector is on every offering row, so a wrong group is never a wrong call.

No I/O and no configuration: everything here is decidable from the strings.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

# Provider-side path prefixes that say where a model is hosted, not what it is.
_PATH_PREFIXES = (
    re.compile(r"^accounts/[^/]+/(?:models|routers)/"),  # Fireworks
    re.compile(r"^(?:us|eu|apac|global|jp)\."),  # Bedrock region routing
    re.compile(r"^TEE/"),  # nano-gpt's confidential-compute variants
)

# Bedrock and its imitators write the vendor into the id with a dot:
# ``anthropic.claude-sonnet-5-v1:0``. The vendor is kept as a hint and the
# segment is dropped from the key.
_DOTTED_VENDOR = re.compile(r"^(anthropic|openai|meta|qwen|deepseek|mistral|cohere|amazon|google|ai21|writer)\.")

# Suffixes that name a version pin or a tier of service on one provider.
_SUFFIXES = (
    re.compile(r"-v\d+:\d+$"),  # Bedrock ``-v1:0``
    re.compile(r":(?:latest|free)$"),
    re.compile(r"@\d{8}$"),  # a dated pin
)

# A trailing quantization describes the serving, so it leaves the key and lands
# on the offering. Case-insensitive because providers disagree (``FP8``, ``fp8``).
_QUANTIZATION = re.compile(r"[-_](fp4|fp8|fp16|bf16|int4|int8|awq|gptq|nvfp4|mxfp4)$", re.IGNORECASE)

# Fireworks writes a decimal point as ``p``: ``glm-5p3``, ``kimi-k2p6``,
# ``qwen3p7-plus``. Only between digits, so a ``p`` in a word is left alone.
_P_FOR_POINT = re.compile(r"(?<=\d)p(?=\d)")

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# The vendor an org segment in a model id names. Lowercased on lookup.
_ORG_VENDORS: dict[str, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "meta": "Meta",
    "meta-llama": "Meta",
    "qwen": "Alibaba",
    "deepseek": "DeepSeek",
    "deepseek-ai": "DeepSeek",
    "moonshotai": "Moonshot AI",
    "mistral": "Mistral AI",
    "mistralai": "Mistral AI",
    "z-ai": "Z.ai",
    "zai": "Z.ai",
    "zai-org": "Z.ai",
    "nvidia": "NVIDIA",
    "microsoft": "Microsoft",
    "minimaxai": "MiniMax",
    "minimax": "MiniMax",
    "bytedance": "ByteDance",
    "nousresearch": "Nous Research",
    "openbmb": "OpenBMB",
    "xai": "xAI",
    "x-ai": "xAI",
    "cohere": "Cohere",
    "amazon": "Amazon",
    "ai21": "AI21",
    "perplexity": "Perplexity",
}

# The vendor a model family's leading token implies, for an id with no org
# segment. Matched as a prefix of the normalized key, longest first, so
# ``gptoss`` wins over ``gpt`` and ``codestral`` is not read as ``code``.
_KEY_VENDORS: tuple[tuple[str, str], ...] = tuple(
    sorted(
        {
            "gptoss": "OpenAI",
            "gpt": "OpenAI",
            "o1": "OpenAI",
            "o3": "OpenAI",
            "o4": "OpenAI",
            "chatgpt": "OpenAI",
            "claude": "Anthropic",
            "llama": "Meta",
            "qwen": "Alibaba",
            "qwq": "Alibaba",
            "deepseek": "DeepSeek",
            "kimi": "Moonshot AI",
            "glm": "Z.ai",
            "gemini": "Google",
            "gemma": "Google",
            "mistral": "Mistral AI",
            "mixtral": "Mistral AI",
            "codestral": "Mistral AI",
            "devstral": "Mistral AI",
            "pixtral": "Mistral AI",
            "magistral": "Mistral AI",
            "ministral": "Mistral AI",
            "voxtral": "Mistral AI",
            "grok": "xAI",
            "nemotron": "NVIDIA",
            "phi": "Microsoft",
            "minimax": "MiniMax",
            "command": "Cohere",
            "nova": "Amazon",
            "seed": "ByteDance",
            "hermes": "Nous Research",
            "jamba": "AI21",
            "sonar": "Perplexity",
        }.items(),
        key=lambda item: -len(item[0]),
    )
)

# The provider ids a vendor publishes its own models under, so the vendor's own
# entry can win the display-name vote over a reseller's spelling.
_VENDOR_PROVIDERS: dict[str, frozenset[str]] = {
    "OpenAI": frozenset({"openai"}),
    "Anthropic": frozenset({"anthropic"}),
    "Google": frozenset({"google", "gemini", "vertex", "vertexai"}),
    "Meta": frozenset({"meta"}),
    "Alibaba": frozenset({"alibaba", "dashscope", "qwen"}),
    "DeepSeek": frozenset({"deepseek"}),
    "Moonshot AI": frozenset({"moonshot", "moonshotai"}),
    "Mistral AI": frozenset({"mistral"}),
    "Z.ai": frozenset({"zai", "z-ai", "zhipu"}),
    "xAI": frozenset({"xai"}),
    "Cohere": frozenset({"cohere"}),
    "Amazon": frozenset({"bedrock", "aws"}),
    "Perplexity": frozenset({"perplexity"}),
    "MiniMax": frozenset({"minimax"}),
}

# Hand rulings for what the two rungs get wrong, keyed ``provider_type/model_id``
# and valued with the grouping key to use. Empty until a report says otherwise;
# the point of having it is that a fix is one line here rather than a rule.
_OVERRIDES: dict[str, str] = {}


def normalize(text: str) -> str:
    """Letters and digits only, lowercase, with Fireworks' ``5p3`` read as ``5.3``.

    The grouping key. Punctuation and case are the two things providers disagree
    about for one model; digits and words are what tell two models apart.
    """
    return _NON_ALNUM.sub("", _P_FOR_POINT.sub(".", text.lower()))


_NON_SLUG = re.compile(r"[^a-z0-9.]+")
_VENDOR_SPACES = re.compile(r"\s+")


def slugify(text: str) -> str:
    """A URL-safe id from a display name: ``GLM-5.3`` becomes ``glm-5.3``.

    Dots stay, since a version reads as one; everything else that is not a
    letter or digit becomes a dash. Removing the dashes and dots gives
    :func:`normalize` of the same text, so two groups with different keys
    cannot share a slug.
    """
    return _NON_SLUG.sub("-", _P_FOR_POINT.sub(".", text.lower())).strip("-.")


def vendor_slug(vendor: str) -> str:
    """The vendor's segment of a model id: ``Z.ai`` becomes ``z-ai``, ``Moonshot AI`` ``moonshotai``.

    Spaces vanish and other punctuation becomes a dash, which is the spelling
    the wider ecosystem already uses for these vendors.
    """
    return _NON_ALNUM.sub("-", _VENDOR_SPACES.sub("", vendor.lower())).strip("-")


@dataclass(frozen=True)
class CleanedId:
    """A provider's model id with the serving details separated out."""

    model: str
    """The id with path, org, region and version pins removed."""

    vendor_hint: str | None
    """The vendor an org segment or dotted prefix named, when one did."""

    quantization: str | None
    """A trailing ``-fp8`` and the like, lowercased, when the id carried one."""


def clean_model_id(model_id: str) -> CleanedId:
    """Strip what a provider adds around a model id."""
    text = model_id
    for prefix in _PATH_PREFIXES:
        text = prefix.sub("", text)

    vendor_hint: str | None = None
    if "/" in text:
        org, _, text = text.rpartition("/")
        vendor_hint = _ORG_VENDORS.get(org.rsplit("/", 1)[-1].lower())

    dotted = _DOTTED_VENDOR.match(text)
    if dotted:
        vendor_hint = vendor_hint or _ORG_VENDORS.get(dotted.group(1))
        text = text[dotted.end() :]

    for suffix in _SUFFIXES:
        text = suffix.sub("", text)

    quantization: str | None = None
    quant = _QUANTIZATION.search(text)
    if quant:
        quantization = quant.group(1).lower()
        text = text[: quant.start()]

    return CleanedId(model=text, vendor_hint=vendor_hint, quantization=quantization)


def _name_key(name: str) -> str:
    # A reseller sometimes puts the org into the display name too
    # (``deepseek-ai/DeepSeek-V4-Pro``); the org is not part of the model.
    return normalize(name.rsplit("/", 1)[-1])


def identity_key(provider_type: str, model_id: str, name: str | None) -> str:
    """The grouping key for one offering.

    An override wins outright; then the models.dev display name when there is
    one; then the cleaned id. Never empty: an id that normalizes to nothing
    (punctuation only) falls back to the raw id so it still groups with itself.
    """
    override = _OVERRIDES.get(f"{provider_type}/{model_id}")
    if override is not None:
        return override
    if name:
        key = _name_key(name)
        if key:
            return key
    key = normalize(clean_model_id(model_id).model)
    return key or normalize(model_id) or model_id


def infer_vendor(model_id: str, key: str) -> str | None:
    """The vendor that built a model, from its id's org segment or its family."""
    hint = clean_model_id(model_id).vendor_hint
    if hint is not None:
        return hint
    for prefix, vendor in _KEY_VENDORS:
        if key.startswith(prefix):
            return vendor
    return None


@dataclass(frozen=True)
class OfferingSeed:
    """What one selector contributes to the vote for its group's identity."""

    selector: str
    provider_type: str
    model_id: str
    name: str | None = None


@dataclass(frozen=True)
class ModelIdentity:
    """One model, as the catalog names it."""

    key: str
    slug: str
    """The model's own segment of its id: ``glm-5.3``."""
    name: str
    vendor: str | None
    selectors: tuple[str, ...]

    @property
    def id(self) -> str:
        """The catalog id, vendor-qualified where the vendor is known: ``z-ai/glm-5.3``.

        What a request may send as ``model`` to reach the model, and the path
        of its page. A model whose vendor nobody could name is its bare slug.
        """
        return f"{vendor_slug(self.vendor)}/{self.slug}" if self.vendor else self.slug


def _fallback_name(model_id: str) -> str:
    return _P_FOR_POINT.sub(".", clean_model_id(model_id).model)


def _vote_name(seeds: list[OfferingSeed], vendor: str | None) -> str:
    """The display name for a group: the vendor's own, else the most common.

    A tie goes to the spelling that appears first, which is the first selector in
    the caller's order (sorted, at the route), so the answer is stable and the
    description shown beside it comes from the same offering. A group with no
    models.dev name at all is named after the first cleaned id, which is what a
    self-hosted model with no public entry gets.
    """
    if vendor is not None:
        own = _VENDOR_PROVIDERS.get(vendor, frozenset())
        for seed in seeds:
            if seed.name and seed.provider_type in own:
                return seed.name.rsplit("/", 1)[-1]
    spellings = [seed.name.rsplit("/", 1)[-1] for seed in seeds if seed.name]
    if spellings:
        counts = Counter(spellings)
        return max(spellings, key=lambda name: (counts[name], -spellings.index(name)))
    return _fallback_name(seeds[0].model_id)


def _vote_vendor(seeds: list[OfferingSeed], key: str) -> str | None:
    votes = Counter(vendor for seed in seeds if (vendor := infer_vendor(seed.model_id, key)) is not None)
    if not votes:
        return None
    return sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0][0]


def group_offerings(seeds: Iterable[OfferingSeed]) -> dict[str, ModelIdentity]:
    """Fold offerings into models, keyed by :func:`identity_key`.

    Selector order within a group is the input order, so a caller that sorts its
    seeds gets sorted offerings back.
    """
    by_key: dict[str, list[OfferingSeed]] = {}
    for seed in seeds:
        by_key.setdefault(identity_key(seed.provider_type, seed.model_id, seed.name), []).append(seed)

    identities: dict[str, ModelIdentity] = {}
    for key, members in by_key.items():
        vendor = _vote_vendor(members, key)
        name = _vote_name(members, vendor)
        # The slug is the key with its separators kept, so it is unique across
        # groups only while it still normalizes to the key. An override can hand
        # a group a name that does not, and the key itself is the safe slug then.
        slug = slugify(name)
        identities[key] = ModelIdentity(
            key=key,
            slug=slug if normalize(slug) == key else key,
            name=name,
            vendor=vendor,
            selectors=tuple(seed.selector for seed in members),
        )
    return identities


__all__ = [
    "CleanedId",
    "vendor_slug",
    "ModelIdentity",
    "OfferingSeed",
    "clean_model_id",
    "group_offerings",
    "identity_key",
    "infer_vendor",
    "normalize",
    "slugify",
]
