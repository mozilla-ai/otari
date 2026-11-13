"""Which providers the country a request comes from, and the country its account is in, may reach."""

from collections.abc import Collection

from gateway.core.config import PROVIDER_TYPE_ALIASES, GatewayConfig
from gateway.services.providers._hosted_name_index import hosted_implementation
from gateway.types.country_gate import CountryGate


def provider_vendor(config: GatewayConfig, instance: str) -> str | None:
    """The provider type serving ``instance``, or None for a server that only speaks a vendor's protocol.

    An instance declared ``openai-compatible`` is a self-hosted vLLM or Ollama
    server, not OpenAI, so no vendor's list applies to it. An instance with no
    ``provider_type`` is named for its provider, as a bare provider name is.
    """
    entry = config.providers.get(instance)
    if isinstance(entry, dict):
        declared = entry.get("provider_type")
        if not isinstance(declared, str) or not declared:
            return instance
        return None if declared in PROVIDER_TYPE_ALIASES else declared
    return hosted_implementation(instance) or instance


def provider_countries(config: GatewayConfig, instance: str) -> Collection[str] | None:
    """The countries ``provider_supported_countries`` lets reach ``instance``, or None for every country."""
    vendor = provider_vendor(config, instance)
    return config.provider_supported_countries.get(vendor) if vendor is not None else None


def closed_provider(config: GatewayConfig, gate: CountryGate, instance: str) -> str | None:
    """What ``instance``'s provider type's list leaves out of this request, for an operator, or None when served."""
    return gate.closed_to(provider_countries(config, instance))


class CountryNotServedError(Exception):
    """The provider that would serve a request does not serve its country, as ``gate`` reads it.

    ``closed`` names the half that decided, for an operator, as ``CountryGate.closed_to`` words it; a caller is
    answered with ``gate.refusal_detail``.
    """

    def __init__(self, gate: CountryGate, closed: str, *, under: str) -> None:
        super().__init__(f"{under} closes to {closed}")
        self.gate = gate
        self.closed = closed


def ensure_hosted_country(gate: CountryGate | None, countries: Collection[str] | None) -> None:
    """Refuse a request a hosted provider's own ``countries`` leave out.

    Raises:
        CountryNotServedError: ``countries`` does not hold the request's country or its account's.
    """
    if gate is None:
        return
    closed = gate.closed_to(countries)
    if closed is not None:
        raise CountryNotServedError(gate, closed, under="the provider's supported_countries")


def ensure_provider_country(config: GatewayConfig, gate: CountryGate | None, instance: str) -> None:
    """Refuse a request ``instance``'s provider type does not serve under ``provider_supported_countries``.

    Raises:
        CountryNotServedError: the type lists countries and the request's or its account's is not one of them.
    """
    if gate is None:
        return
    closed = closed_provider(config, gate, instance)
    if closed is not None:
        raise CountryNotServedError(gate, closed, under=f"provider_supported_countries for {instance}")
