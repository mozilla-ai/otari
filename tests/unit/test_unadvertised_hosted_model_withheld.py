"""What the hosted port's advertised models withhold from the deployment price list, and what they leave alone."""

import pytest

from gateway.core.config import GatewayConfig
from gateway.services.merged_catalog_service import CatalogScope, withheld_as_unadvertised

_CONFIG = GatewayConfig(master_key="k", providers={"home_lab": {"api_key": "x", "provider_type": "openai"}})
_ADVERTISED = {"openai": frozenset({"gpt-4o-mini"}), "mistral": None}


def _scope(byo_providers: frozenset[str]) -> CatalogScope:
    return CatalogScope(
        allowlist=None,
        reads_workspace_layer=True,
        deployment_supplied_providers=frozenset(),
        offered_keys=frozenset(),
        hosted_models=_ADVERTISED,
        byo_providers=byo_providers,
    )


@pytest.mark.parametrize(
    ("model_key", "byo_providers", "withheld"),
    [
        pytest.param("openai:gpt-4o", frozenset(), True, id="not-advertised"),
        pytest.param("openai/gpt-4o", frozenset(), True, id="not-advertised-legacy-spelling"),
        pytest.param("openai:gpt-4o-mini", frozenset(), False, id="advertised"),
        pytest.param("mistral:mistral-small-latest", frozenset(), False, id="hosted-without-a-per-model-answer"),
        pytest.param("anthropic:claude-3-5-haiku-latest", frozenset(), False, id="not-hosted"),
        pytest.param("home_lab:gpt-4o", frozenset(), False, id="configured-instance"),
        pytest.param("openai:gpt-4o", frozenset({"openai"}), False, id="reached-on-the-callers-own-key"),
        pytest.param("__manual__", frozenset(), False, id="no-provider-to-attribute"),
    ],
)
def test_only_an_unadvertised_hosted_model_is_withheld(
    model_key: str, byo_providers: frozenset[str], withheld: bool
) -> None:
    assert withheld_as_unadvertised(_CONFIG, _scope(byo_providers), model_key) is withheld


@pytest.mark.parametrize(
    ("model_key", "byo_providers", "withheld"),
    [
        pytest.param("bedrock:claude-x", frozenset(), True, id="restricted-to-another-organization"),
        pytest.param("bedrock:claude-x", frozenset({"bedrock"}), False, id="reached-on-the-callers-own-key"),
        pytest.param("openai:gpt-4o-mini", frozenset(), False, id="another-provider-unaffected"),
    ],
)
def test_a_provider_withheld_from_the_callers_organization_lists_none_of_its_priced_models(
    model_key: str, byo_providers: frozenset[str], withheld: bool
) -> None:
    scope = CatalogScope(
        allowlist=None,
        reads_workspace_layer=True,
        deployment_supplied_providers=frozenset(),
        offered_keys=frozenset(),
        hosted_models=_ADVERTISED,
        byo_providers=byo_providers,
        withheld_providers=frozenset({"bedrock"}),
    )

    assert withheld_as_unadvertised(_CONFIG, scope, model_key) is withheld


@pytest.mark.parametrize(
    ("model_key", "env", "withheld"),
    [
        pytest.param("anthropic:claude-x", None, True, id="nothing-serves-it"),
        # A variable serves nothing on its own, and a keyless backend serves only when declared.
        pytest.param("anthropic:claude-x", "ANTHROPIC_API_KEY", True, id="its-own-environment-variable"),
        pytest.param("ollama:llama3", None, True, id="undeclared-needing-no-key"),
        pytest.param("not-a-provider:model", None, True, id="no-such-provider"),
        pytest.param("home_lab:gpt-4o", None, False, id="configured-instance"),
        pytest.param("openai:gpt-4o", None, False, id="reached-on-the-callers-own-key"),
        pytest.param("mistral:mistral-small-latest", None, False, id="hosted"),
    ],
)
def test_a_key_that_withholds_the_unserved_lists_only_what_some_credential_serves(
    monkeypatch: pytest.MonkeyPatch, model_key: str, env: str | None, withheld: bool
) -> None:
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    if env is not None:
        monkeypatch.setenv(env, "sk-ambient")
    scope = CatalogScope(
        allowlist=None,
        reads_workspace_layer=True,
        deployment_supplied_providers=frozenset(),
        offered_keys=frozenset(),
        hosted_models={"mistral": None},
        byo_providers=frozenset({"openai"}),
        withhold_unserved=True,
    )

    assert withheld_as_unadvertised(_CONFIG, scope, model_key) is withheld
