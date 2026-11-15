"""How many invitation requests an organization and an inviting account may make in a day.

How many of them may name one address in a day.
"""

import uuid
from datetime import timedelta

import pytest

from gateway.adapters.rate_limit_store_adapter import InMemoryRateLimitStore
from gateway.exceptions.organizations_exceptions import InvitationLimitCeiling
from gateway.ports.rate_limit_store_port import RateLimitStorePort, RateLimitWindow
from gateway.services.organizations._invitation_limits import INVITATION_LIMIT_WINDOW, Admission, InvitationLimits

pytestmark = pytest.mark.asyncio


class _RecordingStore(InMemoryRateLimitStore):
    """An in-memory store that remembers every key and window it was asked to count or settle."""

    def __init__(self) -> None:
        super().__init__()
        self.keys: list[str] = []
        self.windows: list[float] = []

    async def hit(self, key: str, limit: int, window_sec: float, cost: int = 1) -> RateLimitWindow:
        self.keys.append(key)
        self.windows.append(window_sec)
        return await super().hit(key, limit, window_sec, cost)

    async def settle(self, key: str, handle: str, cost: int) -> None:
        self.keys.append(key)
        await super().settle(key, handle, cost)


class _RaisingStore(InMemoryRateLimitStore):
    """An in-memory store whose ``hit`` raises once it has been called ``fail_at`` times."""

    def __init__(self, *, fail_at: int) -> None:
        super().__init__()
        self._fail_at = fail_at
        self._calls = 0

    async def hit(self, key: str, limit: int, window_sec: float, cost: int = 1) -> RateLimitWindow:
        self._calls += 1
        if self._calls == self._fail_at:
            msg = "the store is unreachable"
            raise RuntimeError(msg)
        return await super().hit(key, limit, window_sec, cost)


def _limits(store: RateLimitStorePort, *, per_day: int, per_address_per_day: int) -> InvitationLimits:
    return InvitationLimits(store, per_day=per_day, per_address_per_day=per_address_per_day)


async def test_admit_counts_one_ask_on_each_of_the_four_keys() -> None:
    organization = uuid.uuid4()
    limits = _limits(InMemoryRateLimitStore(), per_day=2, per_address_per_day=10)

    first = await limits.admit(account_id=uuid.uuid4(), email="a@example.com", organization_id=organization)
    second = await limits.admit(account_id=uuid.uuid4(), email="b@example.com", organization_id=organization)
    assert isinstance(first, Admission)
    assert isinstance(second, Admission)

    # A new account asking with a new address is still refused: the organization is full.
    third = await limits.admit(account_id=uuid.uuid4(), email="c@example.com", organization_id=organization)
    assert third is InvitationLimitCeiling.DAILY


async def test_a_refused_ask_counts_nothing() -> None:
    account = uuid.uuid4()
    full_organization = uuid.uuid4()
    limits = _limits(InMemoryRateLimitStore(), per_day=2, per_address_per_day=2)

    # Two asks for the same address fill both the account's own count and its address count.
    assert isinstance(
        await limits.admit(account_id=account, email="a@example.com", organization_id=full_organization), Admission
    )
    assert isinstance(
        await limits.admit(account_id=account, email="a@example.com", organization_id=full_organization), Admission
    )

    fresh_organization = uuid.uuid4()
    refused = await limits.admit(account_id=account, email="b@example.com", organization_id=fresh_organization)
    assert refused is InvitationLimitCeiling.DAILY

    # The refused ask counted nothing against the fresh organization: it still admits a full day's asks.
    other_account = uuid.uuid4()
    assert isinstance(
        await limits.admit(account_id=other_account, email="c@example.com", organization_id=fresh_organization),
        Admission,
    )
    assert isinstance(
        await limits.admit(account_id=other_account, email="d@example.com", organization_id=fresh_organization),
        Admission,
    )


async def test_an_error_mid_count_leaves_no_count_behind() -> None:
    account = uuid.uuid4()
    organization = uuid.uuid4()
    store = _RaisingStore(fail_at=3)
    limits = _limits(store, per_day=1, per_address_per_day=10)

    with pytest.raises(RuntimeError):
        await limits.admit(account_id=account, email="a@example.com", organization_id=organization)

    again = await limits.admit(account_id=account, email="a@example.com", organization_id=organization)
    assert isinstance(again, Admission)


async def test_leaving_the_block_without_keep_gives_back_all_four_counts() -> None:
    account = uuid.uuid4()
    organization = uuid.uuid4()
    limits = _limits(InMemoryRateLimitStore(), per_day=1, per_address_per_day=1)

    admission = await limits.admit(account_id=account, email="a@example.com", organization_id=organization)
    assert isinstance(admission, Admission)
    async with admission:
        pass

    again = await limits.admit(account_id=account, email="a@example.com", organization_id=organization)
    assert isinstance(again, Admission)


async def test_keep_keeps_all_four_counts() -> None:
    account = uuid.uuid4()
    organization = uuid.uuid4()
    limits = _limits(InMemoryRateLimitStore(), per_day=1, per_address_per_day=1)

    admission = await limits.admit(account_id=account, email="a@example.com", organization_id=organization)
    assert isinstance(admission, Admission)
    async with admission:
        admission.keep()

    again = await limits.admit(account_id=account, email="a@example.com", organization_id=organization)
    assert again is InvitationLimitCeiling.DAILY


async def test_leaving_the_block_twice_gives_back_only_once() -> None:
    account = uuid.uuid4()
    organization = uuid.uuid4()
    limits = _limits(InMemoryRateLimitStore(), per_day=1, per_address_per_day=1)

    admission = await limits.admit(account_id=account, email="a@example.com", organization_id=organization)
    assert isinstance(admission, Admission)
    async with admission:
        pass
    async with admission:
        pass

    again = await limits.admit(account_id=account, email="a@example.com", organization_id=organization)
    assert isinstance(again, Admission)


async def test_one_organization_does_not_use_another_organizations_count() -> None:
    full_organization = uuid.uuid4()
    other_organization = uuid.uuid4()
    limits = _limits(InMemoryRateLimitStore(), per_day=1, per_address_per_day=10)

    assert isinstance(
        await limits.admit(account_id=uuid.uuid4(), email="a@example.com", organization_id=full_organization),
        Admission,
    )
    refused = await limits.admit(account_id=uuid.uuid4(), email="b@example.com", organization_id=full_organization)
    assert refused is InvitationLimitCeiling.DAILY

    assert isinstance(
        await limits.admit(account_id=uuid.uuid4(), email="c@example.com", organization_id=other_organization),
        Admission,
    )


async def test_the_account_is_counted_across_organizations() -> None:
    account = uuid.uuid4()
    first_organization = uuid.uuid4()
    second_organization = uuid.uuid4()
    limits = _limits(InMemoryRateLimitStore(), per_day=2, per_address_per_day=10)

    assert isinstance(
        await limits.admit(account_id=account, email="a@example.com", organization_id=first_organization), Admission
    )
    assert isinstance(
        await limits.admit(account_id=account, email="b@example.com", organization_id=second_organization), Admission
    )

    # The account has already made two asks today, so a third is refused,
    # even though the first organization alone has only counted one of them.
    third = await limits.admit(account_id=account, email="c@example.com", organization_id=first_organization)
    assert third is InvitationLimitCeiling.DAILY


async def test_the_same_address_is_counted_per_organization_and_per_account() -> None:
    account = uuid.uuid4()
    organization = uuid.uuid4()
    limits = _limits(InMemoryRateLimitStore(), per_day=10, per_address_per_day=1)

    assert isinstance(
        await limits.admit(account_id=account, email="Grace@X.test", organization_id=organization), Admission
    )
    refused = await limits.admit(account_id=account, email="grace@x.test", organization_id=organization)
    assert refused is InvitationLimitCeiling.ADDRESS

    other_account = uuid.uuid4()
    other_organization = uuid.uuid4()
    passed = await limits.admit(account_id=other_account, email="grace@x.test", organization_id=other_organization)
    assert isinstance(passed, Admission)


async def test_a_second_account_inviting_the_same_address_in_one_organization_is_refused() -> None:
    organization = uuid.uuid4()
    limits = _limits(InMemoryRateLimitStore(), per_day=10, per_address_per_day=1)

    assert isinstance(
        await limits.admit(account_id=uuid.uuid4(), email="a@example.com", organization_id=organization), Admission
    )

    refused = await limits.admit(account_id=uuid.uuid4(), email="a@example.com", organization_id=organization)
    assert refused is InvitationLimitCeiling.ADDRESS


async def test_the_same_account_inviting_an_address_in_a_second_organization_is_refused() -> None:
    account = uuid.uuid4()
    limits = _limits(InMemoryRateLimitStore(), per_day=10, per_address_per_day=1)

    assert isinstance(
        await limits.admit(account_id=account, email="a@example.com", organization_id=uuid.uuid4()), Admission
    )

    refused = await limits.admit(account_id=account, email="a@example.com", organization_id=uuid.uuid4())
    assert refused is InvitationLimitCeiling.ADDRESS


async def test_an_address_that_cannot_be_stored_raises() -> None:
    limits = _limits(InMemoryRateLimitStore(), per_day=10, per_address_per_day=10)

    with pytest.raises(ValueError, match="stored"):
        await limits.admit(account_id=uuid.uuid4(), email="not an address", organization_id=uuid.uuid4())


async def test_no_key_holds_the_address() -> None:
    store = _RecordingStore()
    limits = _limits(store, per_day=10, per_address_per_day=10)

    admission = await limits.admit(account_id=uuid.uuid4(), email="grace@example.com", organization_id=uuid.uuid4())
    assert isinstance(admission, Admission)

    assert store.keys
    assert all(key.startswith("invitation:") for key in store.keys)
    assert not any("grace" in key or "example.com" in key for key in store.keys)


async def test_the_two_per_address_keys_do_not_share_a_digest() -> None:
    store = _RecordingStore()
    limits = _limits(store, per_day=10, per_address_per_day=10)

    admission = await limits.admit(account_id=uuid.uuid4(), email="grace@example.com", organization_id=uuid.uuid4())
    assert isinstance(admission, Admission)

    address_keys = [key for key in store.keys if ":address:" in key]
    assert len(address_keys) == 2
    digests = {key.rsplit(":address:", 1)[1] for key in address_keys}
    assert len(digests) == 2


async def test_two_organizations_inviting_the_same_address_do_not_share_a_digest() -> None:
    store = _RecordingStore()
    limits = _limits(store, per_day=10, per_address_per_day=10)

    first = await limits.admit(account_id=uuid.uuid4(), email="grace@example.com", organization_id=uuid.uuid4())
    second = await limits.admit(account_id=uuid.uuid4(), email="grace@example.com", organization_id=uuid.uuid4())
    assert isinstance(first, Admission)
    assert isinstance(second, Admission)

    organization_address_keys = [key for key in store.keys if ":organization:" in key and ":address:" in key]
    assert len(organization_address_keys) == 2
    digests = {key.rsplit(":address:", 1)[1] for key in organization_address_keys}
    assert len(digests) == 2


async def test_the_window_is_24_hours() -> None:
    assert INVITATION_LIMIT_WINDOW == timedelta(hours=24)

    store = _RecordingStore()
    limits = _limits(store, per_day=10, per_address_per_day=10)
    admission = await limits.admit(account_id=uuid.uuid4(), email="grace@example.com", organization_id=uuid.uuid4())
    assert isinstance(admission, Admission)

    assert store.windows and all(window == 86400.0 for window in store.windows)
