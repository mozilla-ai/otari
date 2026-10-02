"""The identity rules, pinned on the ids that motivated them.

Every example here is a real spelling from models.dev or genai-prices. The two
Fireworks cases are the ones the rules were designed against: a path prefix
around the id, and ``p`` where the version has a point.
"""

import pytest

from gateway.services.model_identity import (
    OfferingSeed,
    canonical_vendor,
    clean_model_id,
    group_offerings,
    identity_key,
    infer_vendor,
    normalize,
    slugify,
    vendor_slug,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("GLM-5.3", "glm53"),
        ("GLM 5.3", "glm53"),
        ("GLM5.3", "glm53"),
        ("glm-5p3", "glm53"),
        ("Kimi K2.6", "kimik26"),
        ("kimi-k2p6", "kimik26"),
        ("GPT OSS 120B", "gptoss120b"),
        ("gpt-oss:120b", "gptoss120b"),
        # ``p`` between letters is a letter: nothing in "gpt" or "pro" is a point.
        ("deepseek-v4-pro", "deepseekv4pro"),
    ],
)
def test_normalize_folds_case_and_punctuation_only(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("model_id", "model", "vendor", "quantization"),
    [
        ("accounts/fireworks/models/glm-5p3", "glm-5p3", None, None),
        ("accounts/fireworks/routers/glm-5p3-fast", "glm-5p3-fast", None, None),
        ("openai/gpt-oss-120b", "gpt-oss-120b", "OpenAI", None),
        ("zai-org/GLM-5.2-FP8", "GLM-5.2", "Z.ai", "fp8"),
        ("us.anthropic.claude-sonnet-5-v1:0", "claude-sonnet-5", "Anthropic", None),
        ("openai.gpt-oss-120b-1:0", "gpt-oss-120b-1:0", "OpenAI", None),
        ("TEE/glm-5.3", "glm-5.3", None, None),
        ("deepseek-ai/DeepSeek-V4-Pro", "DeepSeek-V4-Pro", "DeepSeek", None),
        ("gpt-oss:120b", "gpt-oss:120b", None, None),
        ("nemotron-3-ultra-550b-a55b:free", "nemotron-3-ultra-550b-a55b", None, None),
        ("nvidia.nemotron-nano-9b-v2", "nemotron-nano-9b-v2", "NVIDIA", None),
        # A vendor's name in front of its own family is the vendor, not the model.
        ("nvidia-nemotron-3-super-120b-a12b", "nemotron-3-super-120b-a12b", "NVIDIA", None),
        ("openai-gpt-oss-120b", "gpt-oss-120b", "OpenAI", None),
        # ...but a model named after its vendor keeps the name.
        ("deepseek-v4-pro", "deepseek-v4-pro", None, None),
        ("mistral-medium-3", "mistral-medium-3", None, None),
    ],
)
def test_clean_model_id_separates_serving_from_model(
    model_id: str, model: str, vendor: str | None, quantization: str | None
) -> None:
    cleaned = clean_model_id(model_id)
    assert cleaned.model == model
    assert cleaned.vendor_hint == vendor
    assert cleaned.quantization == quantization


def test_the_name_rung_wins_over_the_id() -> None:
    # Fireworks' id says ``glm-5p3``; models.dev names it "GLM 5.3". Either rung
    # lands on the same key here, and the name is what is asked first.
    assert identity_key("fireworks", "accounts/fireworks/models/glm-5p3", "GLM 5.3") == "glm53"
    # A reseller that shortens the name groups by the name, not the longer id.
    assert identity_key("greenpt", "devstral-2-123b-instruct-2512", "Devstral 2") == "devstral2"


def test_the_id_rung_is_the_fallback() -> None:
    assert identity_key("fireworks", "accounts/fireworks/models/deepseek-v4-pro", None) == "deepseekv4pro"
    assert identity_key("ollama", "gpt-oss:120b", None) == "gptoss120b"


def test_a_vendor_name_in_front_of_the_family_groups_with_the_bare_name() -> None:
    bare = identity_key("fireworks", "accounts/fireworks/models/nemotron-3-ultra", "Nemotron 3 Ultra")
    assert identity_key("venice", "nvidia-nemotron-3-ultra-550b-a55b", "NVIDIA Nemotron 3 Ultra") == bare
    assert identity_key("requesty", "nvidia-nemotron-3-ultra", None) == bare
    assert identity_key("venice", "google-gemma-4-31b-it", "Google Gemma 4 31B") == "gemma431b"
    # Mistral's own names start with its name; nothing is stripped there.
    assert identity_key("mistral", "mistral-medium-3", "Mistral Medium 3") == "mistralmedium3"


def test_a_dated_build_and_a_tier_stay_apart() -> None:
    base = identity_key("nebius", "deepseek/deepseek-v4-pro", "DeepSeek V4 Pro")
    dated = identity_key("fireworks", "accounts/fireworks/models/deepseek-v4-pro-0813", "DeepSeek V4 Pro 0813")
    flash = identity_key("nebius", "zai-org/GLM-5.3-Flash", "GLM-5.3-Flash")
    assert len({base, dated, flash, identity_key("nebius", "zai-org/GLM-5.3", "GLM-5.3")}) == 4


def test_a_name_that_normalizes_to_nothing_falls_through() -> None:
    assert identity_key("home_lab", "qwen3-32b", "///") == "qwen332b"
    assert identity_key("home_lab", "---", None) == "---"


@pytest.mark.parametrize(
    ("model_id", "vendor"),
    [
        ("openai/gpt-oss-120b", "OpenAI"),
        ("gpt-oss-120b", "OpenAI"),
        ("claude-sonnet-4-6", "Anthropic"),
        ("meta-llama/Llama-3.3-70B-Instruct", "Meta"),
        ("Qwen/Qwen3-32B", "Alibaba"),
        ("kimi-k2.6", "Moonshot AI"),
        ("glm-5.3", "Z.ai"),
        ("codestral-2501", "Mistral AI"),
        ("some-house-model", None),
    ],
)
def test_infer_vendor(model_id: str, vendor: str | None) -> None:
    key = identity_key("x", model_id, None)
    assert infer_vendor(model_id, key) == vendor


@pytest.mark.parametrize(
    ("canonical_id", "vendor"),
    [
        ("zhipuai/glm-5.3", "Z.ai"),
        ("nvidia/nemotron-3-ultra", "NVIDIA"),
        ("thinkingmachines/inkling", "Thinking Machines"),
        # A lab the table does not know is still named, as models.dev spells it.
        ("quiverai/arrow-2", "quiverai"),
        ("no-org", None),
        (None, None),
    ],
)
def test_canonical_vendor(canonical_id: str | None, vendor: str | None) -> None:
    assert canonical_vendor(canonical_id) == vendor


def test_the_canonical_id_wins_over_a_router_s_org_segment() -> None:
    # Eden AI routes through Amazon and says so in the id; Google made the model.
    model_id = "amazon/google.gemma-3-4b-it@us"
    key = identity_key("edenai", model_id, None)
    assert infer_vendor(model_id, key) == "Amazon"
    assert infer_vendor(model_id, key, "google/gemma-3-4b-it") == "Google"


def test_fireworks_own_model_is_credited_to_fireworks() -> None:
    model_id = "accounts/fireworks/models/ember-1"
    groups = group_offerings([OfferingSeed(f"fireworks:{model_id}", "fireworks-ai", model_id, "Ember-1")])
    assert groups["ember1"].vendor == "Fireworks"
    assert groups["ember1"].id == "fireworks/ember-1"


def test_nvidia_models_are_credited_to_nvidia_however_the_provider_spells_them() -> None:
    seeds = [
        OfferingSeed("venice:a", "venice", "nvidia-nemotron-3-ultra", "NVIDIA Nemotron 3 Ultra"),
        OfferingSeed("requesty:b", "requesty", "nvidia-nemotron-3-ultra", None),
        OfferingSeed("do:c", "digitalocean", "nemotron-3-ultra", "Nemotron 3 Ultra"),
    ]
    groups = group_offerings(seeds)
    assert set(groups) == {"nemotron3ultra"}
    identity = groups["nemotron3ultra"]
    assert identity.vendor == "NVIDIA"
    # The name drops the vendor too, since the page already shows it beside the name.
    assert identity.name == "Nemotron 3 Ultra"
    assert identity.id == "nvidia/nemotron-3-ultra"


def test_group_offerings_folds_the_fireworks_spellings_with_everyone_else() -> None:
    seeds = [
        OfferingSeed("nebius:zai-org/GLM-5.3", "nebius", "zai-org/GLM-5.3", "GLM-5.3"),
        OfferingSeed(
            "fireworks:accounts/fireworks/models/glm-5p3", "fireworks", "accounts/fireworks/models/glm-5p3", "GLM 5.3"
        ),
        OfferingSeed("digitalocean:glm-5.3", "digitalocean", "glm-5.3", "GLM5.3"),
        OfferingSeed("zai:glm-5.3", "zai", "glm-5.3", "GLM-5.3"),
        OfferingSeed("home_lab:glm-5.3", "openai", "glm-5.3", None),
    ]

    groups = group_offerings(seeds)

    assert set(groups) == {"glm53"}
    identity = groups["glm53"]
    assert identity.name == "GLM-5.3"
    assert identity.slug == "glm-5.3"
    assert identity.vendor == "Z.ai"
    assert identity.id == "z-ai/glm-5.3"
    assert identity.selectors == tuple(seed.selector for seed in seeds)


def test_the_vendor_s_own_spelling_wins_the_name_vote() -> None:
    # Four resellers say "GLM 5.3"; Z.ai itself says "GLM-5.3". The vendor wins.
    seeds = [OfferingSeed(f"r{i}:glm-5.3", f"reseller{i}", "glm-5.3", "GLM 5.3") for i in range(4)]
    seeds.append(OfferingSeed("zai:glm-5.3", "zai", "glm-5.3", "GLM-5.3"))
    assert group_offerings(seeds)["glm53"].name == "GLM-5.3"


def test_without_the_vendor_the_most_common_spelling_wins_and_ties_are_stable() -> None:
    seeds = [
        OfferingSeed("a:x", "a", "gpt-oss-120b", "gpt-oss-120b"),
        OfferingSeed("b:x", "b", "gpt-oss-120b", "GPT OSS 120B"),
        OfferingSeed("c:x", "c", "gpt-oss-120b", "GPT OSS 120B"),
    ]
    assert group_offerings(seeds)["gptoss120b"].name == "GPT OSS 120B"
    # A tie goes to the spelling that came first, so a caller that sorts its
    # seeds gets the same answer every time.
    tied = [seeds[0], seeds[1]]
    assert group_offerings(tied)["gptoss120b"].name == "gpt-oss-120b"
    assert group_offerings(list(reversed(tied)))["gptoss120b"].name == "GPT OSS 120B"


def test_a_model_nobody_names_is_named_after_its_cleaned_id() -> None:
    groups = group_offerings([OfferingSeed("home_lab:qwen3-32b", "openai", "qwen3-32b", None)])
    identity = groups["qwen332b"]
    assert identity.name == "qwen3-32b"
    assert identity.slug == "qwen3-32b"
    assert identity.vendor == "Alibaba"
    assert identity.id == "alibaba/qwen3-32b"


def test_slug_removes_to_the_key() -> None:
    for name in ("GLM-5.3", "DeepSeek V4 Pro", "Kimi K2.6", "gpt-oss-120b", "Claude Sonnet 4.6"):
        assert normalize(slugify(name)) == normalize(name)


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("GLM-5.3", "glm-5.3"),
        ("DeepSeek V4 Pro", "deepseek-v4-pro"),
        ("Kimi K2.6", "kimi-k2.6"),
        ("glm-5p3", "glm-5.3"),
    ],
)
def test_slug_keeps_a_version_s_dot(name: str, slug: str) -> None:
    assert slugify(name) == slug


@pytest.mark.parametrize(
    ("vendor", "slug"),
    [
        ("Z.ai", "z-ai"),
        ("Moonshot AI", "moonshotai"),
        ("OpenAI", "openai"),
        ("Mistral AI", "mistralai"),
        ("xAI", "xai"),
    ],
)
def test_vendor_slug_spells_the_vendor_the_way_the_ecosystem_does(vendor: str, slug: str) -> None:
    assert vendor_slug(vendor) == slug
