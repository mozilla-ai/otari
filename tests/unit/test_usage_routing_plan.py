"""Unit tests for reading the routing plan a usage row recorded.

``usage_logs.routing_plan`` is written by this gateway alone, so its stored shape
normally parses. It is read back on every usage page, though, and one unreadable
value there would turn the whole read into a 500, which is the rule the other
stored shapes on the row follow: an unrecognised value renders as absent rather
than failing the page.
"""

from gateway.schemas.usage_and_telemetry import RoutingPlanRecord, read_routing_plan

PLAN = {
    "declined": "caller sent Otari-Router: off",
    "dropped": [{"selector": "openai:gpt-5", "reason": "not_allowed", "detail": "is not in allowed_models"}],
}


def test_a_stored_plan_comes_back_as_its_own_shape() -> None:
    plan = read_routing_plan(PLAN)
    assert plan is not None
    assert plan.declined == "caller sent Otari-Router: off"
    assert [(item.selector, item.reason) for item in plan.dropped] == [("openai:gpt-5", "not_allowed")]


def test_no_stored_plan_is_no_plan() -> None:
    # The common case: an unrouted row, or a routed one that dropped nothing.
    assert read_routing_plan(None) is None
    assert read_routing_plan({}) is None


def test_an_unreadable_plan_renders_as_absent() -> None:
    # A hand-edited row, or one written by a shape this build does not know. The
    # row's own attribution fields still render, which is what the page needs.
    assert read_routing_plan({"dropped": "not a list"}) is None
    assert read_routing_plan({"dropped": [{"reason": "not_allowed"}]}) is None


def test_the_round_trip_the_writer_does_parses() -> None:
    # What `_routing_plan_record` stores: a dump with nulls omitted.
    stored = RoutingPlanRecord.model_validate(PLAN).model_dump(exclude_none=True)
    assert read_routing_plan(stored) == read_routing_plan(PLAN)
