"""Who an idempotency key belongs to, and when two requests count as the same one."""

import pytest

from gateway.services.inference import IdempotentRequest, InvalidKey


def _request(
    key: str = "k", *, body: bytes = b'{"a":1,"b":2}', options: list[str | None] | None = None, api_key_id: str | None
) -> IdempotentRequest | InvalidKey:
    return IdempotentRequest.of(
        key, endpoint="/v1/chat/completions", body=body, options=options or [None], user_id="u", api_key_id=api_key_id
    )


def test_a_key_belongs_to_the_api_key_that_sent_it() -> None:
    request = _request(api_key_id="key-1")

    assert isinstance(request, IdempotentRequest)
    assert request.scope == "key:key-1"


def test_a_master_key_request_belongs_to_the_billed_user() -> None:
    request = _request(api_key_id=None)

    assert isinstance(request, IdempotentRequest)
    assert request.scope == "master:u"


@pytest.mark.parametrize("key", ["", "x" * 256, " padded", "tab\there", "café"])
def test_an_unusable_key_is_refused(key: str) -> None:
    assert isinstance(_request(key, api_key_id=None), InvalidKey)


def test_key_order_and_spacing_do_not_change_the_request() -> None:
    first = _request(body=b'{"a":1,"b":2}', api_key_id=None)
    second = _request(body=b'{ "b": 2, "a": 1 }', api_key_id=None)

    assert isinstance(first, IdempotentRequest)
    assert isinstance(second, IdempotentRequest)
    assert first.request_hash == second.request_hash


def test_a_different_option_changes_the_request() -> None:
    plain = _request(options=[None], api_key_id=None)
    searched = _request(options=["otari"], api_key_id=None)

    assert isinstance(plain, IdempotentRequest)
    assert isinstance(searched, IdempotentRequest)
    assert plain.request_hash != searched.request_hash
