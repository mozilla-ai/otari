"""Who can enable a disabled API key."""

from gateway.exceptions.api_keys_exceptions import KeyEnableRefusedError
from gateway.models.api_keys import APIKey, DisabledReason


def operator_enable_refusal(api_key: APIKey) -> DisabledReason | None:
    """Why an operator cannot enable ``api_key``, or None if they can.

    The only place the reasons are read for an operator's authority. The key API
    publishes the answer, so a client does not repeat the reasons itself.
    """
    reason = api_key.disabled_reason
    return None if reason is None or reason.operator_may_clear else reason


def operator_set_active(api_key: APIKey, is_active: bool) -> None:
    """Enable or disable ``api_key`` as an operator.

    A disable records a reason the owner cannot clear.

    Raises:
        KeyEnableRefusedError: the key carries a reason an operator cannot clear.
            This refuses a disable too, which would replace that reason with one
            an operator can clear.
    """
    refusal = operator_enable_refusal(api_key)
    if refusal is not None:
        raise KeyEnableRefusedError(refusal)
    api_key.is_active = is_active
    api_key.disabled_reason = None if is_active else DisabledReason.OPERATOR
