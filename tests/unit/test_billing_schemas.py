"""The billing detail on a usage row keeps its wire format, and keeps old rows.

``UsageEntry.billing_meters`` and ``pricing_breakdown`` describe JSON columns.
Naming their real shapes is what lets a generated client type them, but these
models validate rows read back out of a database a self-hosted gateway has been
accumulating across upgrades. Two things therefore have to hold, and neither is
visible from the type annotations alone:

* what goes in comes back out byte-identical, including keys these models do not
  declare, so tightening the contract cannot silently rewrite or drop billing
  detail on the way to a client; and
* a row whose stored shape matches nothing still renders, because the
  alternative is one unrecognised row turning the whole usage page into a 500.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from gateway.api.routes._billing_schemas import TokenChargeLine, UnitChargeLine
from gateway.api.routes.usage import UsageEntry

# Every charge line the gateway writes today, one per producer. Sourced from
# rerank.py, embeddings.py, images.py, metered_pricing.py and pricing_service.py
# rather than invented, so this fails if a producer's shape drifts from the
# contract the models now publish.
TOKEN_LINE = {"meter": "input", "units": 1200, "rate_per_million": 3.0, "cost": 0.0036}
IMAGE_LINE = {"meter": "images", "units": 2, "unit_rate": 0.04, "cost": 0.08}
REQUEST_LINE = {"meter": "request", "units": 1, "unit_rate": 0.002, "cost": 0.002}
TOOL_LINE = {"meter": "web_search_calls", "units": 3, "unit_rate": 0.01, "cost": 0.03}


def entry(**overrides: object) -> UsageEntry:
    """A minimal row, with only the billing fields worth varying."""
    fields: dict[str, object] = {
        "id": "req-1",
        "user_id": None,
        "api_key_id": None,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "model": "gpt-5.6",
        "provider": "openai",
        "endpoint": "chat",
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "cache_read_tokens": None,
        "cache_write_tokens": None,
        "cache_write_1h_tokens": None,
        "billing_meters": None,
        "pricing_breakdown": None,
        "cost": None,
        "status": "success",
        "error_message": None,
        "status_code": None,
        "latency_ms": None,
        "source": "gateway",
        "source_label": None,
        "counts_toward_budget": True,
    }
    fields.update(overrides)
    return UsageEntry(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize("line", [TOKEN_LINE, IMAGE_LINE, REQUEST_LINE, TOOL_LINE])
def test_charge_line_round_trips_unchanged(line: dict[str, object]) -> None:
    """A stored line serializes back exactly as written, keys and values."""
    dumped = entry(pricing_breakdown=[line]).model_dump(mode="json")["pricing_breakdown"]
    assert dumped == [line]


def test_which_rate_a_line_carries_is_what_types_it() -> None:
    """The rate key discriminates, which is the whole point of naming the shapes.

    A renderer picks the branch on the key it finds; if both shapes collapsed to
    one model, a token line could be rendered with an undefined per-call rate.
    """
    breakdown = entry(pricing_breakdown=[TOKEN_LINE, TOOL_LINE]).pricing_breakdown
    assert breakdown is not None
    assert isinstance(breakdown[0], TokenChargeLine)
    assert isinstance(breakdown[1], UnitChargeLine)


def test_an_unrecognised_line_survives_instead_of_failing_the_row() -> None:
    """A shape from an older gateway falls to the permissive arm, and is kept.

    This is the case the fallback exists for: rejecting it would take out the
    whole usage page, not just this line.
    """
    legacy = {"meter": "mystery", "units": 5, "cost": 0.5}
    dumped = entry(pricing_breakdown=[legacy]).model_dump(mode="json")["pricing_breakdown"]
    assert dumped == [legacy]


def test_a_line_keeps_keys_the_models_do_not_declare() -> None:
    """Extra keys ride along rather than being dropped on the way out.

    Without extra="allow", validating into a model would delete billing detail
    silently, which is worse than never having typed the field.
    """
    enriched = {**TOKEN_LINE, "tier": "over_200k", "note": "kept"}
    dumped = entry(pricing_breakdown=[enriched]).model_dump(mode="json")["pricing_breakdown"]
    assert dumped == [enriched]


def test_meters_keep_flat_token_names_and_typed_tool_counts() -> None:
    """Token meters stay flat and open; the reserved `tools` key gets a shape."""
    meters = {
        "input": 1200,
        "completion": 340,
        "some_provider_specific_meter": 7,
        "tools": {"web_search": {"billed": 3, "errors": 1, "unit_rate": 0.01}},
    }
    row = entry(billing_meters=meters)
    assert isinstance(row.billing_meters, dict)
    assert row.billing_meters["tools"]["web_search"]["billed"] == 3
    assert row.model_dump(mode="json")["billing_meters"] == meters


def test_meters_survive_a_tools_value_of_the_wrong_shape() -> None:
    """A `tools` key that is not the tool map still renders the row.

    Reserved or not, the column is JSON and nobody can inspect what older
    versions wrote into it.
    """
    meters = {"input": 10, "tools": "unexpected"}
    dumped = entry(billing_meters=meters).model_dump(mode="json")["billing_meters"]
    assert dumped == meters


def test_a_partial_tool_entry_still_validates() -> None:
    """Counts default, so an entry missing one is not a hard failure."""
    row = entry(billing_meters={"tools": {"code_execution": {"billed": 2}}})
    assert isinstance(row.billing_meters, dict)
    # Absent counts stay absent rather than defaulting to 0 on the wire, so the
    # assertion is about what was stored, not what a model would have filled in.
    assert row.billing_meters["tools"]["code_execution"] == {"billed": 2}


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="null"),
        pytest.param({"nested": 1}, id="nested-object"),
        pytest.param(["a", "b"], id="list"),
    ],
)
def test_a_legacy_line_may_hold_any_json_value(value: object) -> None:
    """The fallback arm accepts whatever a line was written with.

    Constraining its value type defeats the point: a stored line carrying a null,
    a list or a nested object matched no arm and failed the whole row, turning one
    unreadable line into a 500 for the usage page.
    """
    legacy = {"meter": "mystery", "units": 5, "cost": 0.5, "extra": value}
    dumped = entry(pricing_breakdown=[legacy]).model_dump(mode="json")["pricing_breakdown"]
    assert dumped == [legacy]


def test_a_line_that_is_not_an_object_is_still_rejected() -> None:
    """The fallback is permissive about shape, not about type.

    A scalar where a charge line belongs is corruption rather than an older
    shape, and quietly rendering it would hide that.
    """
    with pytest.raises(ValidationError):
        entry(pricing_breakdown=["not-a-line"])


def test_a_row_without_tools_does_not_grow_a_tools_key() -> None:
    """Declaring `tools` must not add it to the rows that never ran one.

    The regression this pins: every non-tool request (nearly all of them) gained
    `"tools": null` in its meters purely because the field became declared, which
    is a wire-format change for the majority of rows.
    """
    meters = {"total_input_tokens": 100}
    assert entry(billing_meters=meters).model_dump(mode="json")["billing_meters"] == meters


def test_an_unpriced_tool_entry_does_not_grow_a_rate() -> None:
    """The same rule one level down: a tool with no rate stored keeps none.

    `unit_rate` is written only once a tool has been priced, so emitting an
    explicit null would claim a rate was recorded when none was.
    """
    meters = {"tools": {"web_search": {"billed": 2, "errors": 0}}}
    assert entry(billing_meters=meters).model_dump(mode="json")["billing_meters"] == meters


def test_the_shapes_reach_the_published_spec() -> None:
    """The properties have to survive into docs/public/openapi.json.

    The spec is the deliverable here: it is what every generated client and SDK
    core is built from, so a shape that validates correctly at runtime but
    publishes as a bare object has bought nothing.

    This exists because that is precisely what happened. Suppressing the models'
    defaults with a ``model_serializer`` replaced the *serialization* JSON schema,
    which is the mode FastAPI publishes for a response, and both meter schemas
    reached the spec with no properties at all while every runtime test above
    still passed.
    """
    spec = json.loads((Path(__file__).resolve().parents[2] / "docs/public/openapi.json").read_text())
    schemas = spec["components"]["schemas"]

    assert "tools" in schemas["BillingMeters"]["properties"]
    assert set(schemas["ToolMeter"]["properties"]) == {"billed", "errors", "unit_rate"}
    # The charge lines carry the discriminator that says how to read a line, and
    # unlike the meters above they require every field.
    assert "rate_per_million" in schemas["TokenChargeLine"]["properties"]
    assert "unit_rate" in schemas["UnitChargeLine"]["properties"]
    assert set(schemas["TokenChargeLine"]["required"]) == {"meter", "units", "rate_per_million", "cost"}
    assert set(schemas["UnitChargeLine"]["required"]) == {"meter", "units", "unit_rate", "cost"}

    # And the union is what a client narrows on, so a row's breakdown has to
    # offer both named shapes rather than only the permissive arm.
    breakdown = schemas["UsageEntry"]["properties"]["pricing_breakdown"]
    refs = json.dumps(breakdown)
    assert "TokenChargeLine" in refs and "UnitChargeLine" in refs
