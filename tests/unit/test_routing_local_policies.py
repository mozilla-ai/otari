"""A gateway whose control plane resolves every request applies none of its own routing policies."""

import pytest

from gateway.exceptions.routing_exceptions import PolicyOnHybridGatewayError
from gateway.services.routing import refuse_local_policy


def test_a_local_policy_name_is_refused_with_what_to_name_instead() -> None:
    with pytest.raises(PolicyOnHybridGatewayError, match="Routing policy 'smart' cannot be used in hybrid mode"):
        refuse_local_policy("smart", {"smart", "cheap"})


@pytest.mark.parametrize("model", ["openai:gpt-4o-mini", "gpt-4o-mini", "smarter"])
def test_a_model_that_names_no_local_policy_passes(model: str) -> None:
    refuse_local_policy(model, {"smart"})


def test_a_gateway_with_no_policies_refuses_nothing() -> None:
    refuse_local_policy("smart", ())
