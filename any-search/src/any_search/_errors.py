"""The errors any_search raises.

None of them carries a response body, a URL, the query or a key: a host logs
and returns these, and the query is the user's content.
"""


class AnySearchError(Exception):
    """Base of every error any_search raises."""


class ProviderError(AnySearchError):
    """The provider failed the call, or could not be reached."""

    def __init__(self, provider: str, status: int | None, tag: str) -> None:
        self.provider = provider
        self.status = status
        self.tag = tag
        detail = tag if status is None else f"{tag}, HTTP {status}"
        super().__init__(f"{provider} search failed ({detail})")


class MissingCredentialError(AnySearchError):
    """The provider needs an API key or base URL that was neither passed nor set in the environment."""

    def __init__(self, provider: str, setting: str, env_var: str | None) -> None:
        self.provider = provider
        self.setting = setting
        self.env_var = env_var
        where = f"pass {setting}= or set {env_var}" if env_var else f"pass {setting}="
        super().__init__(f"{provider} needs {setting}: {where}")


class UnsupportedParameterError(AnySearchError):
    """A native option the provider does not take. Refused rather than dropped, so no option is lost silently."""

    def __init__(self, provider: str, parameter: str) -> None:
        self.provider = provider
        self.parameter = parameter
        super().__init__(f"{provider} does not take the option {parameter!r}")


class UnsupportedProviderError(AnySearchError):
    """A provider name this package does not serve."""

    def __init__(self, provider: str, supported: list[str]) -> None:
        self.provider = provider
        self.supported = supported
        super().__init__(f"unknown search provider {provider!r}; supported: {', '.join(supported)}")
