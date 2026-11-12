"""How an organization's mandated guardrails fold in with the caller's and a policy's.

The composition half of otari#654. `merge_guardrail_layers` is the one place
three layers meet, so the rules that matter are asserted directly on it rather
than through a request: an organization may add a check or tighten one and can
never weaken what is already there, the operator's policy is the outermost layer
and owns the endpoint where both name a profile, and a credential never travels
to an endpoint other than the one it was stored for.

The zero-rows case (`test_no_layer_asked_for_anything...`) is the requirement
#655 puts on every one of the four planes: a deployment that configures nothing
behaves as it did.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal, cast

import pytest
from any_llm import LLMProvider

from gateway.api.routes._pipeline import RequestContext, merge_guardrail_layers
from gateway.core.config import GatewayConfig
from gateway.models.guardrails import GuardrailConfig
from gateway.services.guardrails import GuardrailVerdict, deployment_guardrails_url
from gateway.services.routing import CompiledPlan
from gateway.services.tenancy import organization_guardrail_service
from gateway.services.tenancy.organization_guardrail_service import (
    ResolvedOrganizationGuardrail,
    run_organization_guardrails,
)
from gateway.types.attempt import Attempt


def _guardrail(
    profile: str,
    *,
    mode: Literal["block", "monitor"] = "block",
    on_unavailable: Literal["block", "monitor"] = "block",
    url: str | None = None,
    credential: str | None = None,
) -> GuardrailConfig:
    return GuardrailConfig.model_validate(
        {"profile": profile, "mode": mode, "on_unavailable": on_unavailable, "url": url, "credential": credential}
    )


def _organization(
    guardrail: GuardrailConfig,
    *,
    credential: str | None = None,
    definition_id: uuid.UUID | None = None,
) -> ResolvedOrganizationGuardrail:
    return ResolvedOrganizationGuardrail(config=guardrail, credential=credential, definition_id=definition_id)


def _ctx(*policy_guardrails: GuardrailConfig) -> RequestContext:
    plan: Any = None
    if policy_guardrails:
        plan = CompiledPlan(
            policy_name="p",
            attempts=[
                Attempt(
                    position=1,
                    instance="openai",
                    provider=LLMProvider.OPENAI,
                    model="m",
                    kwargs={"api_key": "sk-test"},
                )
            ],
            guardrails=list(policy_guardrails),
        )
    return RequestContext(
        config=GatewayConfig(),
        db=cast(Any, object()),
        uow=cast(Any, object()),
        log_writer=cast(Any, None),
        api_key_id="key-1",
        user_id="user-1",
        rate_limit_info=None,
        reservation=None,
        started_at=time.monotonic(),
        workspace_id=uuid.uuid4(),
        plan=plan,
    )


def test_no_layer_asked_for_anything_leaves_the_request_exactly_as_it_was() -> None:
    """The zero-rows requirement: no organization entries, no policy, no change."""
    caller = [_guardrail("pii", mode="monitor")]

    unrouted = merge_guardrail_layers(_ctx(), caller, [])
    assert unrouted.configs is caller
    assert unrouted.credentials == {} and unrouted.mandated == frozenset() and unrouted.in_process == {}

    empty = merge_guardrail_layers(_ctx(), None, [])
    assert empty.configs is None
    assert empty.credentials == {} and empty.mandated == frozenset() and empty.in_process == {}


def test_an_organization_guardrail_runs_when_the_caller_asked_for_nothing() -> None:
    merged = merge_guardrail_layers(_ctx(), None, [_organization(_guardrail("prompt-injection"))])

    assert merged.configs is not None
    assert [g.profile for g in merged.configs] == ["prompt-injection"]
    assert merged.configs[0].mode == "block"
    assert merged.credentials == {}


def test_a_caller_cannot_weaken_what_the_organization_mandated() -> None:
    merged = merge_guardrail_layers(
        _ctx(),
        [_guardrail("prompt-injection", mode="monitor", on_unavailable="monitor")],
        [_organization(_guardrail("prompt-injection", mode="block", on_unavailable="block"))],
    )

    assert merged.configs is not None and len(merged.configs) == 1
    assert merged.configs[0].mode == "block"
    assert merged.configs[0].on_unavailable == "block"


def test_a_caller_may_tighten_what_the_organization_mandated() -> None:
    merged = merge_guardrail_layers(
        _ctx(),
        [_guardrail("prompt-injection", mode="block", on_unavailable="block")],
        [_organization(_guardrail("prompt-injection", mode="monitor", on_unavailable="monitor"))],
    )

    assert merged.configs is not None
    assert merged.configs[0].mode == "block"
    assert merged.configs[0].on_unavailable == "block"


def test_a_caller_may_add_their_own_guardrails_alongside_an_organization_mandate() -> None:
    merged = merge_guardrail_layers(
        _ctx(),
        [_guardrail("pii", mode="monitor")],
        [_organization(_guardrail("prompt-injection"))],
    )

    assert merged.configs is not None
    assert sorted(g.profile for g in merged.configs) == ["pii", "prompt-injection"]


def test_the_organization_owns_the_endpoint_for_a_profile_the_caller_also_named() -> None:
    """A caller cannot point a mandated check at a service of their choosing."""
    merged = merge_guardrail_layers(
        _ctx(),
        [_guardrail("prompt-injection", url="https://caller.example/guardrails")],
        [_organization(_guardrail("prompt-injection", url="https://org.example/guardrails"), credential="s3cret")],
    )

    assert merged.configs is not None
    assert merged.configs[0].url == "https://org.example/guardrails"
    assert merged.credentials == {"prompt-injection": "s3cret"}


def test_the_policy_layer_is_outermost_and_takes_the_endpoint_from_the_organization() -> None:
    """Where both mandate one profile, the operator's entry wins, so the credential is dropped.

    The credential was stored for the endpoint the organization named. Once the
    policy's URL has replaced it, carrying the secret along would be sending it
    somewhere it was never meant for.
    """
    merged = merge_guardrail_layers(
        _ctx(_guardrail("prompt-injection", url="https://operator.example/guardrails")),
        None,
        [_organization(_guardrail("prompt-injection", url="https://org.example/guardrails"), credential="s3cret")],
    )

    assert merged.configs is not None and len(merged.configs) == 1
    assert merged.configs[0].url == "https://operator.example/guardrails"
    assert merged.credentials == {}


def test_a_credential_survives_a_policy_that_mandates_a_different_profile() -> None:
    merged = merge_guardrail_layers(
        _ctx(_guardrail("pii")),
        None,
        [_organization(_guardrail("prompt-injection"), credential="s3cret")],
    )

    assert merged.configs is not None
    assert sorted(g.profile for g in merged.configs) == ["pii", "prompt-injection"]
    assert merged.credentials == {"prompt-injection": "s3cret"}


def test_a_mandate_that_names_a_definition_says_which_one_after_the_merge() -> None:
    """The merge is the last place the link exists, so it has to survive it.

    A profile alone is not enough: the runner holds its guardrails under the
    definition's id, because one definition mandated under three profiles is one
    built vendor client.
    """
    definition_id = uuid.uuid4()

    merged = merge_guardrail_layers(
        _ctx(),
        None,
        [_organization(_guardrail("prompt-injection"), definition_id=definition_id)],
    )

    assert merged.in_process == {"prompt-injection": definition_id}
    assert merged.configs is not None and merged.configs[0].url is None
    assert merged.credentials == {}


def test_a_remote_mandate_names_no_definition() -> None:
    merged = merge_guardrail_layers(
        _ctx(),
        None,
        [_organization(_guardrail("prompt-injection", url="https://org.example/guardrails"), credential="s3cret")],
    )

    assert merged.in_process == {}


def test_the_policy_layer_takes_a_profile_off_the_in_process_path() -> None:
    """The exact mirror of the credential drop beside it.

    An operator who named a URL for a profile meant the check to go there, so
    the organization's own definition stops serving it. Leaving the link in
    place would run the check here and ignore the endpoint the outermost layer
    just set.
    """
    merged = merge_guardrail_layers(
        _ctx(_guardrail("prompt-injection", url="https://operator.example/guardrails")),
        None,
        [_organization(_guardrail("prompt-injection"), definition_id=uuid.uuid4())],
    )

    assert merged.in_process == {}
    assert merged.configs is not None and merged.configs[0].url == "https://operator.example/guardrails"


def test_a_definition_survives_a_policy_that_mandates_a_different_profile() -> None:
    definition_id = uuid.uuid4()

    merged = merge_guardrail_layers(
        _ctx(_guardrail("pii")),
        None,
        [_organization(_guardrail("prompt-injection"), definition_id=definition_id)],
    )

    assert merged.in_process == {"prompt-injection": definition_id}


def test_the_strictest_of_all_three_layers_wins() -> None:
    merged = merge_guardrail_layers(
        _ctx(_guardrail("prompt-injection", mode="monitor", on_unavailable="monitor")),
        [_guardrail("prompt-injection", mode="monitor", on_unavailable="block")],
        [_organization(_guardrail("prompt-injection", mode="block", on_unavailable="monitor"))],
    )

    assert merged.configs is not None and len(merged.configs) == 1
    assert merged.configs[0].mode == "block", "the organization's block survives two monitor layers"
    assert merged.configs[0].on_unavailable == "block", "and so does the caller's own"


_CALLER_URL = "https://caller.example/guardrails"


def test_a_callers_credential_reaches_their_own_endpoint_when_nothing_is_mandated() -> None:
    caller = [_guardrail("probe", url=_CALLER_URL, credential="tok")]

    merged = merge_guardrail_layers(_ctx(), caller, [])

    assert merged.configs is caller
    assert merged.credentials == {"probe": "tok"}


def test_a_callers_credential_survives_a_mandate_of_another_profile() -> None:
    merged = merge_guardrail_layers(
        _ctx(_guardrail("pii")),
        [_guardrail("probe", url=_CALLER_URL, credential="tok")],
        [_organization(_guardrail("prompt-injection"), credential="s3cret")],
    )

    assert merged.credentials == {"probe": "tok", "prompt-injection": "s3cret"}


def test_a_callers_credential_never_follows_an_organization_mandate_of_their_profile() -> None:
    """The mandate's URL replaces the caller's, so the caller's secret has nowhere it was meant for."""
    merged = merge_guardrail_layers(
        _ctx(),
        [_guardrail("probe", url=_CALLER_URL, credential="tok")],
        [_organization(_guardrail("probe", url="https://org.example/guardrails"))],
    )

    assert merged.configs is not None
    assert merged.configs[0].url == "https://org.example/guardrails"
    assert merged.credentials == {}


def test_a_callers_credential_never_replaces_the_organizations() -> None:
    merged = merge_guardrail_layers(
        _ctx(),
        [_guardrail("probe", url=_CALLER_URL, credential="tok")],
        [_organization(_guardrail("probe", url="https://org.example/guardrails"), credential="s3cret")],
    )

    assert merged.credentials == {"probe": "s3cret"}


def test_a_callers_credential_never_follows_a_policy_mandate_of_their_profile() -> None:
    merged = merge_guardrail_layers(
        _ctx(_guardrail("probe")),
        [_guardrail("probe", url=_CALLER_URL, credential="tok")],
        [],
    )

    assert merged.credentials == {}


# -- The organization's layer alone, as memory's consolidator runs it --------------------------------------------


async def _run_alone(
    monkeypatch: pytest.MonkeyPatch, mandates: list[ResolvedOrganizationGuardrail], organization_id: uuid.UUID
) -> tuple[dict[str, Any], list[tuple[uuid.UUID, uuid.UUID]]]:
    """What ``run_organization_guardrails`` hands the runner, and which held guardrails it looked up."""
    handed: dict[str, Any] = {}
    looked_up: list[tuple[uuid.UUID, uuid.UUID]] = []

    async def runner(guardrails: list[GuardrailConfig], text: str, **kwargs: Any) -> GuardrailVerdict:
        handed.update(kwargs, guardrails=guardrails, text=text)
        return GuardrailVerdict()

    def held(organization: uuid.UUID, definition: uuid.UUID) -> Any:
        looked_up.append((organization, definition))
        return f"held-{definition}"

    monkeypatch.setattr(organization_guardrail_service, "run_input_guardrails", runner)
    await run_organization_guardrails(
        mandates, "some text", organization_id=organization_id, default_url="http://guardrails:8000", held=held
    )
    return handed, looked_up


@pytest.mark.asyncio
async def test_the_organization_layer_alone_runs_as_the_merge_leaves_it_for_a_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What memory runs a proposal through is what a request from the workspace would run, and nothing else."""
    organization_id, definition_id = uuid.uuid4(), uuid.uuid4()
    mandates = [
        _organization(_guardrail("pii", mode="monitor", url="https://org.example/guardrails"), credential="s3cret"),
        _organization(_guardrail("prompt-injection"), definition_id=definition_id),
        _organization(_guardrail("toxicity", on_unavailable="monitor")),
    ]

    handed, looked_up = await _run_alone(monkeypatch, mandates, organization_id)

    merged = merge_guardrail_layers(_ctx(), None, mandates)
    assert handed["guardrails"] == merged.configs
    assert handed["credentials"] == merged.credentials == {"pii": "s3cret"}
    assert handed["mandated"] == merged.mandated
    assert handed["in_process"] == {"prompt-injection": f"held-{definition_id}"}
    assert set(merged.in_process) == {"prompt-injection"}
    assert looked_up == [(organization_id, definition_id)]
    assert handed["default_url"] == "http://guardrails:8000" and handed["text"] == "some text"


def test_the_deployment_guardrails_url_prefers_the_config_and_falls_back_to_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_GUARDRAILS_URL", "http://from-env:8000")

    assert deployment_guardrails_url(GatewayConfig(guardrails_url="http://from-config:8000")) == (
        "http://from-config:8000"
    )
    assert deployment_guardrails_url(None) == "http://from-env:8000"
    monkeypatch.delenv("OTARI_GUARDRAILS_URL")
    assert deployment_guardrails_url(None) is None
