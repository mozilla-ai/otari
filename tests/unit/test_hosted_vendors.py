"""Who makes each model a hosted provider offers, as the vendor switch reads it."""

from gateway.services.providers._hosted_vendors import OTHER_VENDOR, vendor_of


def test_a_known_org_prefix_names_its_vendor() -> None:
    assert vendor_of("together", "Qwen/Qwen3.5-9B") == "Alibaba"
    assert vendor_of("together", "deepseek-ai/DeepSeek-V3.2") == "DeepSeek"


def test_an_org_prefix_nothing_knows_is_the_vendor_as_spelled() -> None:
    assert vendor_of("together", "black-forest-labs/FLUX.1-schnell") == "black-forest-labs"


def test_a_vendors_own_provider_names_it_for_a_bare_model() -> None:
    assert vendor_of("openai", "dall-e-3") == "OpenAI"


def test_a_bare_model_on_a_reseller_nothing_names_is_other() -> None:
    assert vendor_of("together", "m2-bert-80M-retrieval") == OTHER_VENDOR


def test_another_otari_is_read_under_the_vendor_segment() -> None:
    assert vendor_of("otari", "z-ai/glm-5.3") == "Z.ai"
