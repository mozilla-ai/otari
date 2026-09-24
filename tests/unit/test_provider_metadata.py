"""Unit tests for provider metadata and model context-window lookups.

These read only the bundled any-llm and genai-prices datasets, so they need no
database or network.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from any_llm import AnyLLM

from gateway.core.config import GatewayConfig
from gateway.services import provider_metadata_service
from gateway.services.pricing_service import model_context_window
from gateway.services.provider_metadata_service import (
    env_only_providers,
    known_provider_detail,
    list_known_provider_summaries,
    list_provider_info,
    provider_info,
    run_env_only_provider_notice,
)


def _config(providers: dict[str, dict[str, object]]) -> GatewayConfig:
    return GatewayConfig(master_key="test", providers=providers)


def test_context_window_known_model() -> None:
    """A model genai-prices knows reports its context window."""
    assert model_context_window("openai", "gpt-4o") == 128000


def test_context_window_bare_name_resolves() -> None:
    """A provider-agnostic name still resolves when it is unambiguous."""
    assert model_context_window(None, "gpt-4o") == 128000


def test_context_window_unknown_model_is_none() -> None:
    """An unknown model yields None rather than raising."""
    assert model_context_window("openai", "totally-made-up-model-xyz") is None


def test_provider_info_openai_metadata() -> None:
    """OpenAI resolves a display name, doc URL, pricing links, and capabilities."""
    info = provider_info(_config({"openai": {"api_key": "sk-x"}}), "openai")

    assert info.instance == "openai"
    assert info.provider_type == "openai"
    assert info.name == "OpenAI"
    assert info.doc_url is not None and info.doc_url.startswith("http")
    assert info.pricing_urls  # genai-prices lists OpenAI pricing pages
    assert info.env_key == "OPENAI_API_KEY"
    # any-llm reports OpenAI as vision-capable and able to list models.
    assert info.capabilities.vision is True
    assert info.capabilities.list_models is True


def test_provider_info_uses_instance_type_for_named_instance() -> None:
    """A named instance backed by openai gets openai's metadata, keeps its name."""
    config = _config({"my-openai": {"provider_type": "openai", "api_key": "sk-x", "api_base": "http://x/v1"}})
    info = provider_info(config, "my-openai")

    assert info.instance == "my-openai"
    assert info.provider_type == "openai"
    assert info.name == "OpenAI"
    assert info.capabilities.list_models is True


def test_provider_info_unknown_type_is_graceful() -> None:
    """An unknown provider type falls back to the instance name and empty caps."""
    info = provider_info(_config({"mystery": {"api_key": "x"}}), "mystery")

    assert info.instance == "mystery"
    assert info.name == "mystery"
    assert info.capabilities.vision is False
    assert info.capabilities.list_models is False


def test_list_provider_info_sorted_by_instance() -> None:
    """Providers come back sorted by their configured instance name."""
    config = _config(
        {
            "openai": {"api_key": "sk-x"},
            "anthropic": {"api_key": "sk-y"},
        }
    )
    names = [info.instance for info in list_provider_info(config)]
    assert names == ["anthropic", "openai"]


def test_provider_summaries_import_no_provider_sdks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Listing the picker's providers must not import any provider SDK module.

    This is the whole point of the lazy split (issue #365): the picker list is
    built from the any-llm registry plus the bundled genai-prices names, so it
    stays cheap. get_provider_class is what pulls in a provider SDK, so it must
    never be called while building the summary list.
    """

    def fail_get_provider_class(provider_type: str) -> object:
        raise AssertionError(f"summary listing imported provider SDK for {provider_type!r}")

    monkeypatch.setattr(AnyLLM, "get_provider_class", staticmethod(fail_get_provider_class))

    summaries = list_known_provider_summaries()

    ids = {s.id for s in summaries}
    assert "openai" in ids and "anthropic" in ids
    # Sorted by display name, case-insensitively.
    names = [s.name for s in summaries]
    assert names == sorted(names, key=str.lower)


def test_provider_summaries_do_not_import_provider_submodules() -> None:
    """The summary path leaves any-llm's per-provider submodules unimported.

    Measured in a fresh interpreter rather than in this one. Imports are cached
    process-wide, so an in-process check really asks "did some earlier test in this
    pytest worker already trigger the registry import", and under ``-n auto`` that
    is scheduling luck: the same assertion flaked on main at 4600ae4b while passing
    on the commit before it. Warming the import first and re-measuring would be
    worse than useless, since the second call cannot import anything and the
    assertion would then hold no matter what the picker path started doing.

    A subprocess also tests the case that actually matters. Picker latency is paid
    on a cold gateway process, which is the only place these imports are timed.
    """
    program = textwrap.dedent(
        """
        import json
        import sys

        from gateway.services.provider_metadata_service import list_known_provider_summaries

        offered = len(list_known_provider_summaries())
        submodules = sorted(m for m in sys.modules if m.startswith("any_llm.providers.") and m.count(".") == 2)
        print(json.dumps({"offered": offered, "submodules": submodules}))
        """
    )
    src = Path(__file__).resolve().parents[2] / "src"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(src)},
        timeout=120,
    )
    measured = json.loads(completed.stdout)

    # any-llm's registry module imports its openai base class, which every
    # openai-compatible provider subclasses, so those two are an unavoidable floor.
    # Anything past them is a provider SDK the picker dragged in, which is the lag
    # the lazy split removed.
    assert set(measured["submodules"]) <= {"any_llm.providers.registry", "any_llm.providers.openai"}
    # Stops the assertion above from passing trivially: the picker offers far more
    # providers than it imports, which is the property that keeps it instant.
    assert measured["offered"] > 20


def test_provider_detail_openai() -> None:
    """Detail for a single provider carries its autofill hints."""
    openai = known_provider_detail("openai")

    assert openai is not None
    assert openai.id == "openai"
    assert openai.env_key == "OPENAI_API_KEY"
    assert openai.requires_api_key is True


def test_provider_detail_env_key_present_false_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key-based provider whose env var is unset reports env_key_present False."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    openai = known_provider_detail("openai")

    assert openai is not None
    assert openai.env_key_present is False


def test_provider_detail_env_key_present_true_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key-based provider whose env var is set reports env_key_present True."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    openai = known_provider_detail("openai")

    assert openai is not None
    assert openai.env_key_present is True


def test_provider_detail_env_key_present_false_for_blank_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A whitespace-only env var counts as absent, not a usable key."""
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    openai = known_provider_detail("openai")

    assert openai is not None
    assert openai.env_key_present is False


def test_provider_detail_keyless_backend_never_present() -> None:
    """A keyless backend (ollama) has no env var, so env_key_present stays False."""
    ollama = known_provider_detail("ollama")

    assert ollama is not None
    assert ollama.env_key is None
    assert ollama.requires_api_key is False
    assert ollama.env_key_present is False


def test_provider_detail_unknown_id_is_none() -> None:
    """An unknown provider id yields None (the route maps this to a 404)."""
    assert known_provider_detail("definitely-not-a-provider") is None


# A fixed registry, so the host's own provider env vars cannot leak in and the
# tests do not pay for importing every provider SDK.
_REGISTRY = (
    ("anthropic", ("ANTHROPIC_API_KEY",)),
    ("gemini", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    ("openai", ("OPENAI_API_KEY",)),
)


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider_metadata_service, "_listing_providers_with_env_credential", lambda: _REGISTRY)
    for _pid, names in _REGISTRY:
        for name in names:
            monkeypatch.delenv(name, raising=False)


@pytest.mark.asyncio
@pytest.mark.usefixtures("registry")
async def test_env_only_providers_names_an_unconfigured_env_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider with its credential env var set and no instance is named; a configured one is not."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test")

    names = await env_only_providers(_config({"openai": {"api_key": "sk-test"}}))

    assert names == ["anthropic", "gemini"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("registry")
async def test_env_only_providers_empty_without_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """No credential env var, or a blank one, names nothing."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  ")

    assert await env_only_providers(_config({})) == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("registry")
async def test_env_only_providers_skips_an_implementation_a_named_instance_backs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``my-anthropic`` (provider_type: anthropic) already lists anthropic's models."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    config = _config({"my-anthropic": {"provider_type": "anthropic", "api_key": "sk-ant-test"}})

    assert await env_only_providers(config) == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("registry")
async def test_env_only_providers_empty_on_hosted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hosted deployment serves no inference, so no env credential makes a provider callable there."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    config = GatewayConfig(master_key="test", mode="hosted", providers={})

    assert await env_only_providers(config) == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("registry")
async def test_run_env_only_provider_notice_names_providers_never_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """The startup notice names the provider and where to configure it, never the credential."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    log = MagicMock()
    monkeypatch.setattr(provider_metadata_service, "logger", log)

    await run_env_only_provider_notice(_config({}))

    log.info.assert_called_once()
    template, *args = log.info.call_args.args
    message = template % tuple(args)
    assert "anthropic" in message
    assert "providers:" in message and "Providers page" in message
    assert "sk-ant-secret-value" not in message


@pytest.mark.asyncio
@pytest.mark.usefixtures("registry")
async def test_run_env_only_provider_notice_silent_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    log = MagicMock()
    monkeypatch.setattr(provider_metadata_service, "logger", log)

    await run_env_only_provider_notice(_config({}))

    log.info.assert_not_called()


def test_listing_registry_holds_only_keyed_listing_providers() -> None:
    """The real registry keeps keyed, model-listing providers and drops keyless ones."""
    registry = dict(provider_metadata_service._listing_providers_with_env_credential())

    assert registry["anthropic"] == ("ANTHROPIC_API_KEY",)
    assert registry["gemini"] == ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    assert "ollama" not in registry
