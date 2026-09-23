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
