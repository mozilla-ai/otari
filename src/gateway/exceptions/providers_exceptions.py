"""Errors the provider-key surfaces raise, and the HTTP status each carries."""

from gateway.exceptions import TenancyConflictError, TenancyNotFoundError, TenancyValidationError


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
