"""Errors that the rate-limits domain may raise.

Each carries the HTTP status it renders as.
"""

from gateway.exceptions import TenancyConflictError, TenancyNotFoundError, TenancyValidationError


class RateLimitRuleNotFoundError(TenancyNotFoundError):
    def __init__(self, name: str):
        super().__init__(f"Rate limit rule '{name}' not found")


class RateLimitRuleAlreadyExistsError(TenancyConflictError):
    def __init__(self, name: str):
        super().__init__(f"A rate limit rule named '{name}' already exists")


class RateLimitRuleInConfigError(TenancyConflictError):
    """The rule is declared in config.yml, which the dashboard does not edit."""

    def __init__(self, name: str):
        super().__init__(f"Rate limit rule '{name}' is defined in config.yml; change it there")


class RateLimitRuleInvalidError(TenancyValidationError):
    """An update would leave the rule invalid, such as with no limit at all."""
