"""How a hybrid gateway splits the caller's selector for the control plane's resolve call."""

import pytest

from gateway.api.routes._platform import _split_model_selector


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("openai:gpt-4o", ("openai", "gpt-4o")),
        # A pinned spelling keeps the catalog id whole, so the peer can read it.
        ("nebius:deepseek/deepseek-v4.1-flash", ("nebius", "deepseek/deepseek-v4.1-flash")),
        # A catalog id travels as vendor and model; the peer's index tells the two readings apart.
        ("deepseek/deepseek-v4.1-flash", ("deepseek", "deepseek-v4.1-flash")),
        # The first delimiter decides, as every local resolver splits.
        ("openai/gpt-4o:latest", ("openai", "gpt-4o:latest")),
        ("gpt-4o", (None, "gpt-4o")),
        (":gpt-4o", (None, ":gpt-4o")),
    ],
)
def test_the_selector_is_split_on_its_first_delimiter(selector: str, expected: tuple[str | None, str]) -> None:
    assert _split_model_selector(selector) == expected
