"""Which provider refusals switch a hosted model off, and which never do."""

from unittest.mock import patch

import pytest

from gateway.api.routes import _unavailable_models
from gateway.core.config import GatewayConfig
from gateway.models.providers import HostedProviderModel
from gateway.services.health_checks._probe import capped_error
from gateway.services.inference import is_model_unavailable_error
from gateway.services.providers._hosted_offers import _hold_for_a_price, _release_price_hold
from gateway.services.upstream_redaction import redact_upstream_message


class _UpstreamError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


NOT_SERVERLESS = (
    "Unable to access non-serverless model meta-llama/Llama-3.3-70B-Instruct-Turbo. "
    "Please visit https://api.together.ai/models/meta-llama/Llama-3.3-70B-Instruct-Turbo "
    "to create and start a new dedicated endpoint for the model."
)


@pytest.mark.parametrize(
    "exc",
    [
        _UpstreamError(400, NOT_SERVERLESS),
        _UpstreamError(404, "The model `llama3-70b-8192` has been decommissioned and is no longer supported."),
    ],
)
def test_a_provider_refusing_the_model_itself_is_recognized(exc: Exception) -> None:
    assert is_model_unavailable_error(exc)


@pytest.mark.parametrize(
    "exc",
    [
        # Some providers answer a chat request to an embedding model this way, so it says nothing about the model.
        _UpstreamError(404, "The model `Qwen/Qwen3-Embedding-4B` does not exist"),
        _UpstreamError(400, "Invalid model: mistral-embed"),
        # Wording alone is not enough: a 5xx or a rate limit is not the provider refusing the model.
        _UpstreamError(500, NOT_SERVERLESS),
        _UpstreamError(429, NOT_SERVERLESS),
        Exception(NOT_SERVERLESS),
    ],
)
def test_a_refusal_that_may_be_the_requests_or_transient_is_not(exc: Exception) -> None:
    assert not is_model_unavailable_error(exc)


class _BodiedError(_UpstreamError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(status_code, f"Error code: {status_code} - {{'error': {{'message': {message!r}}}}}")
        self.body = {"message": message}


@pytest.mark.parametrize(
    "message",
    [
        f"Invalid value for 'reasoning_effort': '{NOT_SERVERLESS}'",
        "Invalid value: 'The model `gpt-4o` has been decommissioned'. Supported values are: 'low', 'high'.",
    ],
)
def test_a_refusal_echoed_back_from_the_request_switches_nothing(message: str) -> None:
    assert not is_model_unavailable_error(_BodiedError(400, message))


def test_a_refusal_in_an_sdk_error_body_is_recognized_past_its_framing() -> None:
    assert is_model_unavailable_error(_BodiedError(400, NOT_SERVERLESS))


@pytest.mark.parametrize(
    ("exc", "deployment_paid"),
    [
        # An organization's own key says nothing about what the deployment's key reaches.
        (_UpstreamError(400, NOT_SERVERLESS), False),
        (_UpstreamError(404, "The model `x` does not exist"), True),
    ],
)
def test_nothing_is_switched_off_unless_the_deployments_key_met_the_refusal(
    exc: Exception, deployment_paid: bool
) -> None:
    with patch("gateway.api.routes._unavailable_models.asyncio.create_task") as create_task:
        _unavailable_models.note_unavailable_model(
            GatewayConfig(), "together", "m", exc, deployment_paid=deployment_paid
        )

    create_task.assert_not_called()


def test_the_refusal_mailed_to_operators_is_redacted_and_capped() -> None:
    refusal = capped_error(redact_upstream_message(NOT_SERVERLESS + " key sk-live-abcdefghijklmnop " * 20))

    assert "sk-live-abcdefghijklmnop" not in refusal
    assert len(refusal) <= 300


@pytest.mark.parametrize("reason", ["unavailable", "operator"])
def test_losing_a_price_does_not_relabel_a_model_off_for_another_reason(reason: str) -> None:
    """A rate cures ``unpriced`` and nothing else, so a later price would otherwise switch it back on."""
    row = HostedProviderModel(provider="together", model="m", enabled=False, disabled_reason=reason)

    _hold_for_a_price(row)
    _release_price_hold(row)

    assert row.enabled is False
    assert row.disabled_reason == reason
