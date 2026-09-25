"""Errors an organization's guardrail configuration raises, and the HTTP status each carries."""

from fastapi import status

from gateway.exceptions import TenancyConflictError, TenancyError, TenancyNotFoundError, TenancyValidationError


class OrganizationGuardrailNotFoundError(TenancyNotFoundError):
    def __init__(self, guardrail_id: object):
        super().__init__(f"Organization guardrail {guardrail_id} not found")


class OrganizationGuardrailAlreadyExistsError(TenancyConflictError):
    """The organization already mandates this guardrail profile.

    One row per profile, not per nickname: the effective guardrail set on the
    request path is keyed by profile, so a second row of the same profile could
    never run alongside the first. Refused at the write rather than silently
    losing at admission.
    """

    def __init__(self, profile: object):
        super().__init__(f"This organization already configures the guardrail profile '{profile}'")


class OrganizationGuardrailScopeConflictError(TenancyValidationError):
    """A workspace list was sent for a guardrail that applies to every workspace.

    The two say different things about the same guardrail and the flag wins at
    resolve time, so accepting both would store a list that never decides
    anything while reading as though it does. The create path refuses the same
    pair in its request model; an update can reach it by setting only one half,
    which is why the rule also lives here.
    """

    def __init__(self) -> None:
        super().__init__("workspace_ids must be empty when applies_to_all_workspaces is true")


class OrganizationGuardrailCredentialNeedsUrlError(TenancyValidationError):
    """A credential was stored on an entry that names no endpoint of its own.

    Without a ``url`` the credential rides to whatever the deployment's
    ``guardrails_url`` points at, which the shipped compose file makes a
    same-host ``http://`` sidecar, so the bearer would cross the wire in clear.
    Naming the endpoint is what puts it through ``validate_mcp_url``, which
    refuses ``http`` once a credential is in play, so requiring one here is what
    makes "a credential is sent over https" true rather than aspirational.
    """

    def __init__(self) -> None:
        super().__init__(
            "A guardrail credential requires the entry to name its own https url; "
            "the deployment's guardrails_url is not necessarily encrypted"
        )


class OrganizationGuardrailUnsafeUrlError(TenancyValidationError):
    """The endpoint failed the same SSRF and TLS checks a request-body guardrail faces.

    Carries the reason from `services.url_safety.UnsafeURLError` verbatim, for
    the reason `WorkspaceMcpServerUnsafeUrlError` does: it names the host and the
    range it resolved into, and this surface is management-gated rather than
    caller-supplied.
    """

    def __init__(self, reason: str):
        super().__init__(reason)


class OrganizationGuardrailLimitReachedError(TenancyValidationError):
    """The organization already mandates as many guardrails as it may.

    Every mandated guardrail in scope for a workspace is one more sequential
    call the guardrails service makes before the provider is reached, on every
    request that workspace sends, so the list is latency an organization spends
    rather than only rows it stores.
    """

    def __init__(self, limit: int):
        super().__init__(f"This organization already configures the maximum of {limit} guardrails")


class OrganizationGuardrailSingleBackendError(TenancyValidationError):
    """A mandate named both an endpoint of its own and a definition to build.

    Two ways to run one check, and nothing in the row decides between them, so
    the pair is refused rather than resolved. ``credential`` counts as naming an
    endpoint: it is only ever sent to one, so a credential beside a definition is
    the same contradiction one step back.

    ``ck_organization_guardrails_single_backend`` says the url half in the
    database too, which is what holds when a write path forgets. It says it as
    an ``IntegrityError``, and the write path reports one of those as a profile
    collision, so the answer a caller can act on has to come from here.
    """

    def __init__(self) -> None:
        super().__init__(
            "A guardrail mandate names either its own endpoint or a definition for Otari to "
            "build, not both; clear one of them"
        )


class OrganizationGuardrailTestsItsDefinitionError(TenancyConflictError):
    """A test asked for a mandate that runs one of the organization's definitions.

    That check is the definition's, and its own test runs it; testing it here
    as well would be a second way to reach one runner.
    """

    def __init__(self) -> None:
        super().__init__("This mandate runs a guardrail you configured; test that guardrail instead")


class OrganizationGuardrailNoEndpointError(TenancyConflictError):
    """A test asked for a mandate with no endpoint, on a deployment that sets none either."""

    def __init__(self) -> None:
        super().__init__(
            "This guardrail names no endpoint and the deployment has no guardrails URL, so there is "
            "nothing to test against"
        )


class OrganizationGuardrailCheckFailedError(TenancyError):
    """The guardrails service was called and the check did not come back.

    A 502 naming nothing the service said, for the reason
    `OrganizationGuardrailDefinitionCheckFailedError` gives: the reason is logged.
    """

    status_code = status.HTTP_502_BAD_GATEWAY

    def __init__(self) -> None:
        super().__init__("The guardrail could not be evaluated. The reason is in the gateway's log.")


class OrganizationGuardrailDefinitionNotFoundError(TenancyNotFoundError):
    def __init__(self, definition_id: object):
        super().__init__(f"Organization guardrail definition {definition_id} not found")


class OrganizationGuardrailDefinitionAlreadyExistsError(TenancyConflictError):
    """The organization already has a definition under this name.

    One definition per name, which is what a mandate points at and what an
    organization recognizes it by. Unlike the mandate's ``profile``, the name is
    the organization's own label, so the same guardrail may be defined twice
    under two names with different arguments.
    """

    def __init__(self, name: object):
        super().__init__(f"This organization already defines a guardrail named '{name}'")


class OrganizationGuardrailDefinitionLimitReachedError(TenancyValidationError):
    """The organization already defines as many guardrails as it may.

    A separate ceiling from the mandates', and bounding something else: each
    enabled definition becomes a built vendor client held in memory by every
    worker, where a mandate is one more sequential call on a request. Neither
    number constrains the other, so neither limit stands in for the other.
    """

    def __init__(self, limit: int):
        super().__init__(f"This organization already defines the maximum of {limit} guardrails")


class OrganizationGuardrailNotBuildableError(TenancyValidationError):
    """The named guardrail is not one this gateway can construct and call itself.

    One answer for a name the installed any-guardrail has never heard of and for
    one whose guardrail holds local model weights, because a caller can do
    nothing different with either. What this gateway builds is a property of the
    library it ships with, so the catalog read is where the answer comes from.
    """

    def __init__(self, guardrail_name: object):
        super().__init__(
            f"'{guardrail_name}' is not a guardrail this deployment can run itself; "
            "the built-in guardrail catalog lists the ones it can"
        )


class OrganizationGuardrailNotDefinableError(TenancyValidationError):
    """The guardrail exists and an organization still may not define it.

    A different refusal from :class:`OrganizationGuardrailNotBuildableError`,
    which is about what is possible. This one is about who pays: a guardrail that
    takes no credential of its own bills whatever the deployment's environment
    holds, outside the budget the request reserved.
    """

    def __init__(self, guardrail_name: object):
        super().__init__(
            f"'{guardrail_name}' cannot be defined by an organization, because it would run on the "
            "deployment's own credentials with nothing metering it"
        )


class OrganizationGuardrailDefinitionArgumentsError(TenancyValidationError):
    """The submitted build arguments do not make a guardrail this gateway can construct.

    Carries the reason verbatim, the way `OrganizationGuardrailUnsafeUrlError`
    does, because every reason names a parameter and this surface is
    management-gated rather than caller-supplied. Every rule behind it is read
    off the guardrail catalog, so the refusals and the form's fields are one
    derivation; a list written here could only drift from the picker it exists to
    accept.
    """

    def __init__(self, reason: str):
        super().__init__(reason)


class OrganizationGuardrailDefinitionUnsafeUrlError(TenancyValidationError):
    """A build argument names an address this gateway must not dial.

    Named after the argument it came from, because a guardrail can take several
    and the caller has to know which field to fix. The reason is carried
    verbatim, as `OrganizationGuardrailDefinitionArgumentsError` explains, and it
    is a reason that names the host and the range it resolved into rather than
    the URL: an endpoint can carry a credential in its userinfo, and this answer
    goes back over the API.
    """

    def __init__(self, argument: str, reason: str):
        super().__init__(f"'{argument}' is not an address this gateway may dial: {reason}")


class OrganizationGuardrailDefinitionNotRunningError(TenancyConflictError):
    """A test asked for a guardrail this worker does not hold built.

    Disabled, failed to build, or not caught up with a write yet: the
    definition's `build_state` says which, and that is where the fix is.
    """

    def __init__(self) -> None:
        super().__init__(
            "This guardrail is not running here, so it cannot be tested. Its status says why; fix that and try again."
        )


class OrganizationGuardrailDefinitionCheckFailedError(TenancyError):
    """The vendor was called and the check did not come back.

    A 502, because the fault is upstream of this gateway. The message names
    neither the vendor's error nor its type: a vendor library may put the
    credentials it was handed into its own message, so the reason is logged and
    only the log carries it.
    """

    status_code = status.HTTP_502_BAD_GATEWAY

    def __init__(self) -> None:
        super().__init__("The guardrail could not be evaluated. The reason is in the gateway's log.")


class OrganizationGuardrailDefinitionInUseError(TenancyConflictError):
    """A mandate still names the definition the caller asked to drop.

    The link is ``RESTRICT`` so that dropping a definition cannot silently stop
    a guardrail running, and without this the database's refusal would reach the
    caller as a 500. The profiles are named because clearing them is the
    caller's next move and both surfaces belong to the same organization; the
    list is read after the delete was refused, so a mandate removed in between
    leaves it empty rather than making this a different answer.
    """

    def __init__(self, profiles: list[str]):
        mandated = f" by {', '.join(profiles)}" if profiles else ""
        super().__init__(
            f"This guardrail definition is still mandated{mandated}; stop mandating it first, "
            "or set enabled to false to switch it off everywhere at once"
        )


__all__ = [
    "OrganizationGuardrailAlreadyExistsError",
    "OrganizationGuardrailCheckFailedError",
    "OrganizationGuardrailCredentialNeedsUrlError",
    "OrganizationGuardrailDefinitionAlreadyExistsError",
    "OrganizationGuardrailDefinitionArgumentsError",
    "OrganizationGuardrailDefinitionCheckFailedError",
    "OrganizationGuardrailDefinitionInUseError",
    "OrganizationGuardrailDefinitionLimitReachedError",
    "OrganizationGuardrailDefinitionNotFoundError",
    "OrganizationGuardrailDefinitionNotRunningError",
    "OrganizationGuardrailDefinitionUnsafeUrlError",
    "OrganizationGuardrailLimitReachedError",
    "OrganizationGuardrailNoEndpointError",
    "OrganizationGuardrailNotBuildableError",
    "OrganizationGuardrailNotDefinableError",
    "OrganizationGuardrailNotFoundError",
    "OrganizationGuardrailScopeConflictError",
    "OrganizationGuardrailSingleBackendError",
    "OrganizationGuardrailTestsItsDefinitionError",
    "OrganizationGuardrailUnsafeUrlError",
]
