"""Which routing policies a gateway whose control plane resolves every request can apply: none of its own."""

from collections.abc import Collection

from gateway.exceptions.routing_exceptions import PolicyOnHybridGatewayError


def refuse_local_policy(model: str, policy_names: Collection[str]) -> None:
    """Refuse ``model`` when it names one of this gateway's own routing policies.

    The control plane resolves the model for every request on a hybrid gateway,
    and a local policy name is not a model it knows, so it would come back as a
    confusing unknown-model refusal. The platform still routes; this gateway's
    own policies are what does not apply.

    Raises:
        PolicyOnHybridGatewayError: ``model`` names a local policy.
    """
    if model in policy_names:
        raise PolicyOnHybridGatewayError(
            f"Routing policy {model!r} cannot be used in hybrid mode. The connected platform resolves "
            "the model for every request, so a local policy name is not a model it knows. Name a concrete "
            "model, or run this gateway in standalone mode where routing policies apply."
        )
