"""The plugin's domain errors, on Otari's own tenancy error family.

Subclassed from ``gateway.services.tenancy.errors`` rather than declared
fresh, and that is the whole registration: Otari registers one handler for
``TenancyError`` in ``gateway.main`` and each error carries its own
``status_code``, so a subclass declared out here renders as FastAPI's
``{"detail": ...}`` body with the right status and the plugin's routes stay as
thin as Otari's own. There is no per-plugin exception-handler seam and none is
needed.
"""

from gateway.services.tenancy.errors import (
    TenancyConflictError,
    TenancyNotFoundError,
    TenancyValidationError,
)


class AlertRuleNotFoundError(TenancyNotFoundError):
    """No such rule, or not one this caller's organization owns.

    One status for both, so a rule id is not an existence oracle across
    tenants.
    """

    def __init__(self, rule_id: object):
        super().__init__(f"Alert rule {rule_id} not found")


class AlertRuleAlreadyExistsError(TenancyConflictError):
    """The organization already has an alert rule under this name.

    Unique per organization so the name is usable as the thing an operator
    recognizes a rule by, in the rule list and in the alert body itself, where
    it is the only field identifying which rule produced the message.
    """

    def __init__(self, name: object):
        super().__init__(f"This organization already has an alert rule named '{name}'")


class AlertRuleUnsupportedDestinationError(TenancyValidationError):
    """Apprise cannot deliver to this destination, or Otari has not classified it.

    Carries the dispatcher's own message, which names some accepted schemas
    and deliberately does not echo the rejected URL back: an unparseable
    destination is still a string the operator may have pasted a real token
    into.
    """

    def __init__(self, reason: str):
        super().__init__(reason)


class AlertRuleUnsafeDestinationError(TenancyValidationError):
    """The destination resolves somewhere this deployment must not post to.

    Only the schemas whose netloc is an operator-chosen host reach this check;
    see ``otari_alerts.destinations`` for which those are and why the rest
    cannot be checked this way. Carries the reason from ``UnsafeURLError``
    verbatim: it names the host and the range it resolved into, and this
    surface is management-gated rather than caller-supplied.
    """

    def __init__(self, reason: str):
        super().__init__(reason)


class AlertRuleInertError(TenancyValidationError):
    """The rule has no warning threshold and no exceeded alert, so it can never fire.

    Storable and meaningless: it would sit in the list looking like configured
    alerting while being incapable of producing a message. ``enabled`` is the
    field that means "stop sending", and it keeps saying so.

    Checked on the merged row as well as in the create body, because a PATCH
    can reach the same dead state by sending only one of the two halves.
    """

    def __init__(self) -> None:
        super().__init__(
            "A rule must do something: set warn_at_percent, or notify_on_exceeded, or both. "
            "Use enabled=false to stop a rule without discarding it"
        )


class AlertRuleLimitReachedError(TenancyValidationError):
    """The organization already has as many alert rules as it may configure.

    A bound on fan-out rather than on storage: every rule that matches a
    crossing is one more outbound send the evaluator's tick waits on, so the
    limit is what keeps one organization's rule list from setting the pace of
    the whole deployment's alerting.
    """

    def __init__(self, limit: int):
        super().__init__(f"An organization may configure at most {limit} alert rules")


__all__ = [
    "AlertRuleAlreadyExistsError",
    "AlertRuleInertError",
    "AlertRuleLimitReachedError",
    "AlertRuleNotFoundError",
    "AlertRuleUnsafeDestinationError",
    "AlertRuleUnsupportedDestinationError",
]
