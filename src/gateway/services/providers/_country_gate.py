"""Which providers the country a request comes from may reach (``provider_supported_countries``)."""

from collections.abc import Collection
from dataclasses import dataclass

from gateway.core.config import PROVIDER_TYPE_ALIASES, GatewayConfig
from gateway.services.providers._hosted_name_index import hosted_implementation


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


@dataclass(frozen=True)
class CountryGate:
    """The country a request comes from, None when no source knows it, as the gate reads it."""

    country: str | None

    def refuses(self, config: GatewayConfig, instance: str) -> bool:
        """Whether ``instance``'s provider type lists countries and this request's is not one of them."""
        vendor = provider_vendor(config, instance)
        allowed = config.provider_supported_countries.get(vendor) if vendor is not None else None
        return allowed is not None and self.country not in allowed

    def excludes(self, countries: Collection[str] | None) -> bool:
        """Whether a hosted provider's own ``countries`` leave this request's out; None serves every country."""
        return countries is not None and self.country not in countries

    def refusal_detail(self, selector: str) -> str:
        """The caller-facing reason ``selector`` is refused."""
        origin = f"from country '{self.country}'" if self.country is not None else "from an unknown country"
        return f"Model '{selector}' is not available to requests {origin}."


class CountryNotServedError(Exception):
    """The provider that would serve a request does not serve its country, as ``gate`` reads it."""

    def __init__(self, gate: CountryGate) -> None:
        super().__init__(gate.country)
        self.gate = gate


def ensure_hosted_country(gate: CountryGate | None, countries: Collection[str] | None) -> None:
    """Refuse a request a hosted provider's own ``countries`` leave out.

    Raises:
        CountryNotServedError: ``countries`` does not hold the request's country.
    """
    if gate is not None and gate.excludes(countries):
        raise CountryNotServedError(gate)


def ensure_provider_country(config: GatewayConfig, gate: CountryGate | None, instance: str) -> None:
    """Refuse a request ``instance``'s provider type does not serve under ``provider_supported_countries``.

    Raises:
        CountryNotServedError: the type lists countries and the request's is not one of them.
    """
    if gate is not None and gate.refuses(config, instance):
        raise CountryNotServedError(gate)
