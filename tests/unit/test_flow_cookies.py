"""The flow cookies: one per sign-in in flight, each set without rewriting the others."""

from http.cookies import SimpleCookie

import pytest
from fastapi import Request, Response

from gateway.api.routes._flow_cookies import MAX_LIVE_FLOWS, FlowCookies
from gateway.models.tenancy import OAUTH_STATE_TTL_SECONDS

FLOWS = FlowCookies(prefix="otari_test_flow", path="/api/v1/auth/test")


class Browser:
    """A cookie jar that sends what it holds and keeps what a response sets."""

    def __init__(self, **held: str) -> None:
        self.jar: dict[str, str] = dict(held)

    def request(self, *, scheme: str = "https") -> Request:
        header = "; ".join(f"{name}={value}" for name, value in self.jar.items())
        return Request(
            {
                "type": "http",
                "method": "GET",
                "scheme": scheme,
                "server": ("testserver", 443),
                "path": FLOWS.path,
                "query_string": b"",
                "headers": [(b"cookie", header.encode())],
            }
        )

    def land(self, response: Response) -> None:
        for morsel in set_cookies(response).values():
            if morsel["max-age"] == "0":
                self.jar.pop(morsel.key, None)
            else:
                self.jar[morsel.key] = morsel.value

    def start(self, state: str) -> None:
        response = Response()
        FLOWS.start(self.request(), response, state=state, browser_token=f"token-for-{state}")
        self.land(response)

    def token(self, state: str) -> str | None:
        return FLOWS.browser_token(self.request(), state)


def set_cookies(response: Response) -> SimpleCookie:
    jar: SimpleCookie = SimpleCookie()
    for name, value in response.raw_headers:
        if name == b"set-cookie":
            jar.load(value.decode())
    return jar


def test_every_live_flow_keeps_its_own_token() -> None:
    browser = Browser()
    browser.start("first")
    browser.start("second")

    assert browser.token("first") == "token-for-first"
    assert browser.token("second") == "token-for-second"


def test_a_state_the_browser_started_no_flow_for_has_no_token() -> None:
    browser = Browser()
    browser.start("first")

    assert browser.token("second") is None


def test_two_flows_started_from_the_same_cookies_both_keep_their_tokens() -> None:
    browser = Browser()
    browser.start("earlier")
    first, second = Response(), Response()
    FLOWS.start(browser.request(), first, state="first", browser_token="token-for-first")
    FLOWS.start(browser.request(), second, state="second", browser_token="token-for-second")

    browser.land(first)
    browser.land(second)

    assert browser.token("earlier") == "token-for-earlier"
    assert browser.token("first") == "token-for-first"
    assert browser.token("second") == "token-for-second"


def test_a_finished_flow_takes_its_token_out_of_the_browser_and_leaves_the_others() -> None:
    browser = Browser()
    browser.start("first")
    browser.start("second")
    response = Response()

    FLOWS.finish(browser.request(), response, "first")
    browser.land(response)

    assert browser.token("first") is None
    assert browser.token("second") == "token-for-second"


def test_a_flow_that_finishes_while_another_starts_does_not_drop_it() -> None:
    browser = Browser()
    browser.start("finishing")
    started, finished = Response(), Response()
    FLOWS.start(browser.request(), started, state="starting", browser_token="token-for-starting")
    FLOWS.finish(browser.request(), finished, "finishing")

    browser.land(started)
    browser.land(finished)

    assert browser.token("finishing") is None
    assert browser.token("starting") == "token-for-starting"


def test_the_oldest_flows_go_once_a_browser_holds_too_many() -> None:
    browser = Browser()
    states = [f"flow-{index}" for index in range(MAX_LIVE_FLOWS + 2)]
    for state in states:
        browser.start(state)

    assert [state for state in states if browser.token(state) is None] == states[:2]
    assert len(browser.jar) == MAX_LIVE_FLOWS


def test_a_browser_that_holds_more_than_the_bound_is_brought_back_to_it() -> None:
    # Sign-ins that start together each see the cookies as they were, so together they can overshoot.
    browser = Browser()
    overshoot = [Response() for _ in range(MAX_LIVE_FLOWS + 3)]
    for index, response in enumerate(overshoot):
        FLOWS.start(browser.request(), response, state=f"together-{index}", browser_token="token")
    for response in overshoot:
        browser.land(response)
    assert len(browser.jar) == MAX_LIVE_FLOWS + 3

    browser.start("next")

    assert len(browser.jar) == MAX_LIVE_FLOWS
    assert browser.token("next") == "token-for-next"


def test_a_browser_far_past_the_bound_gets_a_bounded_response_and_still_comes_back_to_it() -> None:
    planted = 90
    browser = Browser(**{f"{FLOWS.prefix}_{index:064x}": "token" for index in range(planted)})
    response = Response()

    FLOWS.start(browser.request(), response, state="first", browser_token="token")
    browser.land(response)

    assert len(set_cookies(response)) == MAX_LIVE_FLOWS + 1
    for index in range(planted):
        browser.start(f"later-{index}")
    assert len(browser.jar) == MAX_LIVE_FLOWS


def test_an_expired_flow_is_expired_at_the_path_it_was_set_on() -> None:
    browser = Browser()
    browser.start("first")
    response = Response()

    FLOWS.finish(browser.request(), response, "first")

    (cookie,) = set_cookies(response).values()
    assert cookie["max-age"] == "0"
    assert cookie["path"] == FLOWS.path


def test_one_prefix_leaves_the_flows_of_another_alone() -> None:
    others = FlowCookies(prefix="otari_other_flow", path=FLOWS.path)
    browser = Browser(**{f"{others.prefix}_{index:064x}": "token" for index in range(MAX_LIVE_FLOWS + 1)})
    response = Response()

    FLOWS.start(browser.request(), response, state="first", browser_token="token")

    assert len(set_cookies(response)) == 1


def test_a_state_that_carries_a_header_reaches_no_response_header() -> None:
    state = "state\r\nSet-Cookie: session=stolen"
    browser = Browser()
    started, finished = Response(), Response()

    FLOWS.start(browser.request(), started, state=state, browser_token="token")
    browser.land(started)
    held = browser.token(state)
    FLOWS.finish(browser.request(), finished, state)
    browser.land(finished)

    assert held == "token"
    assert browser.jar == {}
    for response in (started, finished):
        assert [name for name, _ in response.raw_headers].count(b"set-cookie") == 1
        assert "stolen" not in str(response.raw_headers)


@pytest.mark.parametrize(
    "planted",
    [
        "otari_test_flow_",
        "otari_test_flow_not-a-digest",
        f"otari_test_flow_{'A' * 64}",
        f"otari_test_flow_{'a' * 65}",
        f"otari_test_flow_{'a' * 63},",
        f"other_flow_{'a' * 64}",
    ],
)
def test_a_cookie_this_deployment_did_not_name_is_never_named_back(planted: str) -> None:
    browser = Browser(**{planted: "planted"})
    for index in range(MAX_LIVE_FLOWS + 1):
        browser.start(f"flow-{index}")

    assert browser.jar[planted] == "planted"


def test_the_cookie_is_scoped_hidden_from_script_and_short_lived() -> None:
    response = Response()

    FLOWS.start(Browser().request(), response, state="first", browser_token="token")

    (cookie,) = set_cookies(response).values()
    assert cookie["path"] == FLOWS.path
    assert cookie["httponly"]
    assert cookie["secure"]
    assert cookie["samesite"] == "lax"
    assert cookie["max-age"] == str(OAUTH_STATE_TTL_SECONDS)


def test_a_plain_http_browser_gets_a_cookie_it_will_send_back() -> None:
    response = Response()

    FLOWS.start(Browser().request(scheme="http"), response, state="first", browser_token="token")

    (cookie,) = set_cookies(response).values()
    assert not cookie["secure"]
