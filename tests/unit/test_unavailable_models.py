"""Which provider refusals switch a hosted model off, and which never do."""

import asyncio
from unittest.mock import patch

import pytest

from _unavailable_model_support import RecordingListener, background_tasks_done
from gateway.api.deps import build_unavailable_model_reporter
from gateway.core.config import GatewayConfig
from gateway.models.providers import HostedProviderModel
from gateway.services.health_checks import UnavailableModelListener
from gateway.services.health_checks._probe import capped_error
from gateway.services.inference import UnavailableModel, UnavailableModelReporter
from gateway.services.inference._upstream_errors import is_model_unavailable_error
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


def _refusal() -> _UpstreamError:
    return _UpstreamError(400, NOT_SERVERLESS)


class _SlowListener(RecordingListener):
    def __init__(self, failure: Exception | None = None) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.failure = failure

    async def model_unavailable(self, unavailable_model: UnavailableModel) -> None:
        await self.release.wait()
        if self.failure is not None:
            raise self.failure
        await super().model_unavailable(unavailable_model)


@pytest.mark.parametrize("uses_deployment_credential", [True, False])
@pytest.mark.asyncio
async def test_a_refusal_reaches_the_listener_in_the_providers_words(uses_deployment_credential: bool) -> None:
    listener = RecordingListener()

    UnavailableModelReporter(listener).report(
        _refusal(), instance="together", model="m", uses_deployment_credential=uses_deployment_credential
    )
    await background_tasks_done()

    assert listener.models == [UnavailableModel("together", "m", NOT_SERVERLESS, uses_deployment_credential)]


@pytest.mark.asyncio
async def test_a_failure_that_is_not_a_refusal_reaches_no_listener() -> None:
    listener = RecordingListener()

    UnavailableModelReporter(listener).report(
        _UpstreamError(404, "The model `x` does not exist"),
        instance="together",
        model="m",
        uses_deployment_credential=True,
    )
    await background_tasks_done()

    assert listener.models == []


@pytest.mark.asyncio
async def test_a_report_returns_before_the_listener_has_reacted() -> None:
    listener = _SlowListener()

    UnavailableModelReporter(listener).report(
        _refusal(), instance="together", model="m", uses_deployment_credential=True
    )
    reacted_on_return = list(listener.models)
    listener.release.set()
    await background_tasks_done()

    assert reacted_on_return == []
    assert len(listener.models) == 1


@pytest.mark.asyncio
async def test_a_refusal_the_listener_is_still_being_told_of_is_not_told_twice() -> None:
    listener = _SlowListener()
    reporter = UnavailableModelReporter(listener)

    reporter.report(_refusal(), instance="together", model="m", uses_deployment_credential=True)
    reporter.report(_refusal(), instance="together", model="m", uses_deployment_credential=True)
    reporter.report(_refusal(), instance="together", model="other", uses_deployment_credential=True)
    reporter.report(_refusal(), instance="together", model="m", uses_deployment_credential=False)
    listener.release.set()
    await background_tasks_done()

    assert [(heard.model, heard.uses_deployment_credential) for heard in listener.models] == [
        ("m", True),
        ("other", True),
        ("m", False),
    ]


@pytest.mark.parametrize("failure", [RuntimeError("boom"), TimeoutError()], ids=["unexpected", "database"])
@pytest.mark.asyncio
async def test_a_listener_that_fails_is_logged_and_told_of_the_next_refusal(
    failure: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    listener = _SlowListener(failure=failure)
    listener.release.set()
    reporter = UnavailableModelReporter(listener)

    reporter.report(_refusal(), instance="together", model="m", uses_deployment_credential=True)
    await background_tasks_done()
    listener.failure = None
    reporter.report(_refusal(), instance="together", model="m", uses_deployment_credential=True)
    await background_tasks_done()

    assert caplog.text.count("Could not act on unavailable model together:m") == 1
    assert listener.models == [UnavailableModel("together", "m", NOT_SERVERLESS, True)]


@pytest.mark.asyncio
async def test_a_listener_that_is_canceled_is_told_of_the_next_refusal() -> None:
    listener = _SlowListener()
    reporter = UnavailableModelReporter(listener)

    reporter.report(_refusal(), instance="together", model="m", uses_deployment_credential=True)
    (telling,) = asyncio.all_tasks() - {asyncio.current_task()}
    await asyncio.sleep(0)
    telling.cancel()
    await background_tasks_done()
    listener.release.set()
    reporter.report(_refusal(), instance="together", model="m", uses_deployment_credential=True)
    await background_tasks_done()

    assert telling.cancelled()
    assert listener.models == [UnavailableModel("together", "m", NOT_SERVERLESS, True)]


@pytest.mark.asyncio
async def test_the_app_reports_unavailable_models_to_the_health_checks() -> None:
    reporter = build_unavailable_model_reporter(GatewayConfig())

    with patch.object(UnavailableModelListener, "model_unavailable") as model_unavailable:
        reporter.report(_refusal(), instance="together", model="m", uses_deployment_credential=True)
        await background_tasks_done()

    model_unavailable.assert_awaited_once_with(UnavailableModel("together", "m", NOT_SERVERLESS, True))


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
