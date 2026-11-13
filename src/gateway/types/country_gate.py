"""The countries a request is judged by: where it comes from, and where its account is."""

from collections.abc import Collection
from dataclasses import dataclass


@dataclass(frozen=True)
class CountryGate:
    """The countries a request is judged by, as the provider country gate reads them.

    ``country`` is where the request comes from, None when no source knows it,
    and an unknown one is refused by any list. ``account_country`` is where the
    account lives, as ``AccountCountryPort`` answers it, and an unknown one is
    no opinion: a list refuses it only when it is known and left out.
    ``account_unresolved`` says the port could not answer, which is not the
    same as knowing no country: a list then refuses rather than going by the
    request's country alone, so an outage cannot open the gate.
    """

    country: str | None
    account_country: str | None = None
    account_unresolved: bool = False

    def closed_to(self, countries: Collection[str] | None) -> str | None:
        """What ``countries`` leaves out, as an operator reads it, or None when the request is served.

        None ``countries`` serves every country. The phrase names the half that
        decided, for logs and operator details only: ``refusal_detail`` is what
        a caller sees.
        """
        if countries is None:
            return None
        if self.country not in countries:
            return f"country '{self.country}'" if self.country is not None else "an unknown country"
        if self.account_unresolved:
            return "an account country that could not be read"
        if self.account_country is not None and self.account_country not in countries:
            return f"account country '{self.account_country}'"
        return None

    def excludes(self, countries: Collection[str] | None) -> bool:
        """Whether a hosted provider's own ``countries`` leave this request out; None serves every country."""
        return self.closed_to(countries) is not None

    def refusal_detail(self, selector: str, *, noun: str = "Model") -> str:
        """The caller-facing reason ``selector`` is refused; ``noun`` says what it names.

        Names the request's country, which is the caller's own, and never the
        account's, nor which of the two a list left out: the wording is the
        same either way, so the detail gives nothing away about what the
        deployment holds on the account.
        """
        origin = f"from country '{self.country}'" if self.country is not None else "from an unknown country"
        return f"{noun} '{selector}' is not available to requests {origin}."
