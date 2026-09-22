"""The gateway installs what the guardrails in its built-in catalog need.

The catalog lists a guardrail because any-guardrail's metadata says it runs over a
hosted API, not because its vendor SDK is installed, so the two can disagree. A
guardrail the catalog offers and the gateway cannot import is one an organization
can define, save and mandate, and then every request in its scope is refused.
"""

from __future__ import annotations

import pytest
from any_guardrail import AnyGuardrail, GuardrailName

from gateway.services.guardrail_catalog import build_builtin_guardrail_catalog


@pytest.mark.parametrize(
    "guardrail_name",
    [spec.guardrail_name for spec in build_builtin_guardrail_catalog().guardrails],
)
def test_every_listed_guardrail_imports_with_its_vendor_sdk(guardrail_name: str) -> None:
    """Loading the class imports the module, and with it the SDK the module imports at load.

    An SDK a guardrail imports only inside its constructor (Bedrock's boto3,
    watsonx's ibm-watsonx-ai) is not reached here: building one would dial the
    vendor.
    """
    AnyGuardrail.get_supported_model(GuardrailName(guardrail_name))
