"""Errors the provider-key surfaces raise, and the HTTP status each carries."""

from gateway.exceptions import TenancyConflictError, TenancyNotFoundError, TenancyValidationError


class OrgProviderKeyNotFoundError(TenancyNotFoundError):
    def __init__(self, key_id: object):
        super().__init__(f"Provider key {key_id} not found")


class OrgProviderKeyNameRequiredError(TenancyValidationError):
    """A key name that is absent, null, or blank once trimmed.

    ``OrgProviderKey.name`` is NOT NULL, and ``OrgProviderKeyUpdateRequest``
    types it as nullable so a client can send an explicit ``null``, the same
    shape ``WorkspaceNameRequiredError`` guards against for a workspace. Left
    unguarded, an explicit ``null`` reaches the database as a NOT NULL
    violation, which the surrounding duplicate-name handling then reports as a
    409 naming a key called "None" rather than the 400 this is.
    """

    def __init__(self) -> None:
        super().__init__("A provider key name is required")


class OrgProviderKeyUnknownProviderError(TenancyValidationError):
    """A ``provider`` that is blank, or does not resolve to a known any-llm implementation.

    ``OrgProviderKey.provider`` is stored verbatim and is exactly the string
    ``cached_org_provider_kwargs`` keys its cache on, matched against a
    resolved selector's ``LLMProvider.value`` at dispatch (see
    ``org_provider_key_service.refresh_org_provider_cache``). Left unguarded,
    a typo, unexpected casing, or an unaliased value (``"OpenAI"``,
    ``"azure-openai"``, trailing whitespace) is accepted with a 201 and then
    never resolves at dispatch, with no error at either point. Mirrors
    ``/v1/provider-credentials``'s ``_validate_instance`` provider_type guard,
    including ``PROVIDER_TYPE_ALIASES`` so an aliased name still resolves.
    """

    def __init__(self, provider: str) -> None:
        if not provider:
            super().__init__("A provider is required")
        else:
            super().__init__(f"'{provider}' is not a known provider implementation")


class OrgProviderKeyUnsafeApiBaseError(TenancyValidationError):
    """An ``api_base`` that resolves to an internal address, gated off.

    Wraps ``services.url_safety.UnsafeURLError`` as a tenancy error so the
    route stays thin; the message is that function's own, which already
    carries no more than the host it refused.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class OrgProviderKeyAlreadyExistsError(TenancyConflictError):
    def __init__(self, provider: str, name: str):
        super().__init__(f"A '{provider}' key named '{name}' already exists in this organization")


class OrgProviderKeyArchivedError(TenancyValidationError):
    """The key is archived, which refuses every mutation except restore."""

    def __init__(self, key_id: object):
        super().__init__(f"Provider key {key_id} is archived; restore it before changing it")


class OrgProviderKeyNotArchivedError(TenancyValidationError):
    """Deletion requires archiving first, the same two-step every irreversible action here takes."""

    def __init__(self, key_id: object):
        super().__init__(f"Provider key {key_id} must be archived before it can be deleted")


class OrgDefaultProviderKeyConflictError(TenancyConflictError):
    """Two concurrent 'set default' calls raced for the same (organization, provider).

    The partial unique index (``uq_org_provider_keys_org_default``) is the
    actual arbiter; this is what the loser's ``IntegrityError`` is mapped to.
    """

    def __init__(self, provider: str) -> None:
        super().__init__(f"Another request just changed the default '{provider}' key; retry")


class OrgProviderKeyDisabledForWorkspaceError(TenancyValidationError):
    """A model restriction was requested for a key this workspace has disabled.

    Refused rather than stored: a restriction on a key the workspace cannot
    use anyway would resurface with a stale list if the key were re-enabled
    later, which `set_workspace_override_for_user` already deletes for the
    opposite transition (see the repository docstring on the cascade).
    """

    def __init__(self) -> None:
        super().__init__("This provider key is disabled for the workspace; enable it before restricting its models")


class WorkspaceProviderKeyOverrideConflictError(TenancyValidationError):
    """A caller asked to pin and disable the same key in the same request.

    Sending one flag lets the other auto-resolve (pinning re-enables a
    disabled key, disabling un-pins a pinned one); sending both explicitly
    true is a contradiction with no safe default to pick.
    """

    def __init__(self) -> None:
        super().__init__("A provider key override cannot be both pinned as default and disabled")


class OrgProviderModelNotFoundError(TenancyNotFoundError):
    def __init__(self, model_id: object):
        super().__init__(f"Offered model {model_id} not found")

class OrgProviderModelNameRequiredError(TenancyValidationError):
    """A model name that is blank once trimmed.

    The name is half of the pricing key, so a blank one would store a row
    nothing can ever price or dispatch.
    """

    def __init__(self) -> None:
        super().__init__("A model name is required")

class OrgProviderLastModelError(TenancyValidationError):
    """Removing the last offered model would widen the key rather than narrow it.

    No offered rows means a key is unnarrowed, which is what a key nobody has
    refreshed looks like. Deleting the last row therefore returns the key to
    reaching every model its provider serves, which is the opposite of what
    "stop offering" reads as. The switch is how a model stops being served.
    """

    def __init__(self) -> None:
        super().__init__(
            "This is the last model offered on the key, and removing it would return the key to serving every "
            "model its provider does. Switch the model off instead."
        )

class OrgProviderModelUnpricedError(TenancyValidationError):
    """Serving was asked for a model nothing prices.

    The other half of the disabled-until-priced rule. The offer path records such
    a model unserved so it cannot be billed at nothing; without this the switch
    would be a way straight past that, putting the model in the catalog and
    through the dispatch gate with no rate behind it.
    """

    def __init__(self, model: str) -> None:
        super().__init__(
            f"Nothing prices '{model}' yet, so it cannot be served. Set a rate for it, or refresh pricing "
            "if the community data has since caught up."
        )

class OrgProviderModelAlreadyOfferedError(TenancyConflictError):
    """The model is already offered on this key.

    The unique index (``uq_org_provider_key_models_key_model``) is the actual
    arbiter; this is what a racing insert's ``IntegrityError`` is mapped to, the
    same shape ``OrgProviderKeyAlreadyExistsError`` has.
    """

    def __init__(self, provider: str, model: str) -> None:
        super().__init__(f"'{model}' is already offered on this '{provider}' key")


__all__ = [
    "OrgDefaultProviderKeyConflictError",
    "OrgProviderKeyAlreadyExistsError",
    "OrgProviderKeyArchivedError",
    "OrgProviderKeyDisabledForWorkspaceError",
    "OrgProviderKeyNameRequiredError",
    "OrgProviderKeyNotArchivedError",
    "OrgProviderKeyNotFoundError",
    "OrgProviderKeyUnknownProviderError",
    "OrgProviderKeyUnsafeApiBaseError",
    "OrgProviderLastModelError",
    "OrgProviderModelAlreadyOfferedError",
    "OrgProviderModelNameRequiredError",
    "OrgProviderModelNotFoundError",
    "OrgProviderModelUnpricedError",
    "WorkspaceProviderKeyOverrideConflictError",
]
