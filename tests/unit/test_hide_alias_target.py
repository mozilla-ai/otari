"""What a provider's error may say about the model behind an alias."""

from any_llm import LLMProvider

from gateway.services.catalog import hide_alias_target
from gateway.services.provider_kwargs import ResolvedProvider


def _resolved(alias: str | None, *, instance: str = "home_lab", model: str = "qwen3") -> ResolvedProvider:
    return ResolvedProvider(instance=instance, provider=LLMProvider.OPENAI, model=model, kwargs={}, alias=alias)


def test_text_is_unchanged_without_an_alias() -> None:
    assert hide_alias_target("The model `qwen3` does not exist", _resolved(None)) == "The model `qwen3` does not exist"


def test_every_spelling_of_the_target_becomes_the_alias() -> None:
    text = "home_lab:qwen3 failed; openai:qwen3 is unknown; `qwen3` does not exist"
    assert hide_alias_target(text, _resolved("housemodel")) == (
        "housemodel failed; housemodel is unknown; `housemodel` does not exist"
    )


def test_an_alias_containing_its_target_is_not_rewritten_twice() -> None:
    assert hide_alias_target("`qwen3` does not exist", _resolved("qwen3-fast")) == "`qwen3-fast` does not exist"
