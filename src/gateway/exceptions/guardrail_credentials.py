"""The ways a stored guardrail definition can be wrong.

Every member names a rule the guardrail catalog states, not a rule written here,
so the store can only ever refuse what the picker would not have offered. The
one exception is the pair at the bottom, which are about the row rather than its
arguments.

No ``status_code`` on the class. That convention belongs to
``services/tenancy/errors.py``, where one registered handler renders a family
too large to wrap in a try block per route; this family has one caller, and
``api/routes/guardrail_credentials.py`` maps it the way
``api/routes/search_tools.py`` maps its own. A message here is shown to a
deployment operator verbatim, so it names the parameter and the alternative and
never the value.
"""


class GuardrailCredentialError(Exception):
    """A stored guardrail definition could not be written as asked."""


class UnknownGuardrailError(GuardrailCredentialError):
    """The named guardrail is not one this gateway can build and call itself.

    Either the name is not an ``any_guardrail`` class at all, or it is one that
    works by holding model weights in the process running it, which belongs in
    the guardrails service rather than here.
    """

    def __init__(self, guardrail_name: str) -> None:
        super().__init__(
            f"'{guardrail_name}' is not a guardrail this gateway can run. "
            f"Choose one the guardrail catalog lists."
        )


class UnknownGuardrailParameterError(GuardrailCredentialError):
    """An argument name the guardrail does not take.

    No guardrail accepts ``**kwargs``, so this would fail when the guardrail was
    built. It is refused at write time because a secret under an unexpected name
    would otherwise be stored in the plain column, the catalog having no ``secret``
    flag to classify it by.
    """

    def __init__(self, guardrail_name: str, parameter: str, stage: str) -> None:
        super().__init__(f"'{guardrail_name}' takes no {stage} argument called '{parameter}'.")


class UnstorableGuardrailParameterError(GuardrailCredentialError):
    """An argument that is a live object rather than a value.

    Upstream types two secrets as JSON because they are already-built Python
    objects holding an open connection and refreshed tokens. No row can hold
    one, encrypted or not, so the write is refused rather than silently dropped,
    which would leave an operator believing a session was saved.
    """

    def __init__(self, guardrail_name: str, parameter: str, alternatives: str) -> None:
        super().__init__(
            f"'{parameter}' is a live object that cannot be stored. "
            f"Configure '{guardrail_name}' with {alternatives} instead."
        )


class MissingGuardrailParameterError(GuardrailCredentialError):
    """A required argument was not supplied and nothing else can supply it.

    ``requirement`` is a whole sentence rather than a name, because a one-of
    constraint has no single name to give: it comes from the catalog's
    requirement group, whose wording is upstream's own.
    """

    def __init__(self, guardrail_name: str, requirement: str) -> None:
        super().__init__(f"'{guardrail_name}' cannot be stored: {requirement}")


class GuardrailCredentialNotFoundError(GuardrailCredentialError):
    """No stored guardrail goes by that name."""

    def __init__(self, name: str) -> None:
        super().__init__(f"No stored guardrail '{name}'.")


class GuardrailCredentialExistsError(GuardrailCredentialError):
    """A stored guardrail already goes by that name."""

    def __init__(self, name: str) -> None:
        super().__init__(f"A stored guardrail '{name}' already exists; use PATCH to update it.")
