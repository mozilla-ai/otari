"""Who can enable a disabled API key."""

from gateway.exceptions.api_keys_exceptions import KeyEnableRefusedError
from gateway.models.api_keys import APIKey, DisabledReason

# The disabled reasons an operator can clear. Both write paths read this: the
# single-key switch below, and the bulk disable the membership listener runs.
OPERATOR_CLEARABLE: frozenset[DisabledReason] = frozenset(
    reason for reason in DisabledReason if reason.operator_may_clear
)


def operator_set_active(api_key: APIKey, is_active: bool) -> None:
    """Enable or disable ``api_key`` as an operator.

    A disable records a reason the owner cannot clear.

    Raises:
        KeyEnableRefusedError: the key carries a reason an operator cannot clear.
            This refuses a disable too, which would replace that reason with one
            an operator can clear.
    """
    reason = api_key.disabled_reason
    if reason is not None and reason not in OPERATOR_CLEARABLE:
        raise KeyEnableRefusedError(reason)
    api_key.is_active = is_active
    api_key.disabled_reason = None if is_active else DisabledReason.OPERATOR
