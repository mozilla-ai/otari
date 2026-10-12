"""What a provider's error may say about the model behind an alias."""

from any_llm import LLMProvider

from gateway.services.catalog import hide_alias_target
from gateway.services.provider_kwargs import ResolvedProvider


def _hide(text: str, alias: str | None) -> str:
    resolved = ResolvedProvider(instance="home_lab", provider=LLMProvider.OPENAI, model="qwen3", kwargs={}, alias=alias)
    return hide_alias_target(
        text,
        alias=resolved.alias,
        instance=resolved.instance,
        model=resolved.model,
        dispatch_model=resolved.dispatch_model,
    )


def test_text_is_unchanged_without_an_alias() -> None:
    assert _hide("The model `qwen3` does not exist", None) == "The model `qwen3` does not exist"


def test_every_spelling_of_the_target_becomes_the_alias() -> None:
    text = "home_lab:qwen3 failed; openai:qwen3 is unknown; `qwen3` does not exist"
    assert _hide(text, "housemodel") == ("housemodel failed; housemodel is unknown; `housemodel` does not exist")


def test_an_alias_containing_its_target_is_not_rewritten_twice() -> None:
    assert _hide("`qwen3` does not exist", "qwen3-fast") == "`qwen3-fast` does not exist"


def _hide_gpt4(text: str) -> str:
    return hide_alias_target(text, alias="fast", instance="openai", model="gpt-4", dispatch_model="openai:gpt-4")


def test_a_longer_model_name_is_not_garbled() -> None:
    text = "model 'gpt-4o' and gpt-4-turbo and gpt-4.1 and openai:gpt-4o-mini do not exist"
    assert _hide_gpt4(text) == text


def test_the_whole_name_is_replaced_at_any_boundary() -> None:
    assert _hide_gpt4("model 'gpt-4' not found. Try gpt-4.") == "model 'fast' not found. Try fast."


def test_a_qualified_name_is_replaced_whole() -> None:
    assert _hide_gpt4("openai:gpt-4 is unavailable (gpt-4)") == "fast is unavailable (fast)"


def test_another_instance_of_the_name_is_left_alone() -> None:
    assert _hide_gpt4("azure:gpt-4 and gpt-4/v2") == "azure:gpt-4 and gpt-4/v2"


def test_a_path_prefixed_name_is_still_hidden() -> None:
    assert _hide_gpt4("models/gpt-4 is not found") == "models/fast is not found"


def test_an_alias_containing_its_qualified_target_is_not_rewritten_twice() -> None:
    text = "openai:gpt-4 failed"
    result = hide_alias_target(
        text, alias="openai:gpt-4-fast", instance="openai", model="gpt-4", dispatch_model="openai:gpt-4"
    )
    assert result == "openai:gpt-4-fast failed"
