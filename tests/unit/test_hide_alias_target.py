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
