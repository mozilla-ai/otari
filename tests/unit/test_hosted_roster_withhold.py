"""What a hosted roster withholds from the deployment price list, and what it leaves alone."""

import pytest

from gateway.core.config import GatewayConfig
from gateway.services.merged_catalog_service import withheld_by_hosted_roster

_CONFIG = GatewayConfig(master_key="k", providers={"home_lab": {"api_key": "x", "provider_type": "openai"}})
_ROSTER = {"openai": frozenset({"gpt-4o-mini"}), "mistral": None}


@pytest.mark.parametrize(
    ("model_key", "byo_providers", "withheld"),
    [
        pytest.param("openai:gpt-4o", frozenset(), True, id="off-the-roster"),
        pytest.param("openai/gpt-4o", frozenset(), True, id="off-the-roster-legacy-spelling"),
        pytest.param("openai:gpt-4o-mini", frozenset(), False, id="on-the-roster"),
        pytest.param("mistral:mistral-small-latest", frozenset(), False, id="hosted-with-no-roster"),
        pytest.param("anthropic:claude-3-5-haiku-latest", frozenset(), False, id="not-hosted"),
        pytest.param("home_lab:gpt-4o", frozenset(), False, id="configured-instance"),
        pytest.param("openai:gpt-4o", frozenset({"openai"}), False, id="reached-on-the-callers-own-key"),
        pytest.param("__manual__", frozenset(), False, id="no-provider-to-attribute"),
    ],
)
def test_the_roster_withholds_only_a_hosted_model_the_deployment_no_longer_advertises(
    model_key: str, byo_providers: frozenset[str], withheld: bool
) -> None:
    assert withheld_by_hosted_roster(_CONFIG, _ROSTER, byo_providers, model_key) is withheld
