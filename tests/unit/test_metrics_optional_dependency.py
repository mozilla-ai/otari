"""Tests for the optional ``prometheus-client`` extra, with no database required.

``prometheus-client`` ships in the dev group, so the absent case cannot be
observed by importing it here; the first test runs it in a subprocess with the
module blocked, which is also what the OSS edition smoke gate exercises for real
(it installs no extras).
"""

import subprocess
import sys
import textwrap

import pytest

from gateway.core.config import GatewayConfig
from gateway.main import _validate_metrics_support

_WITHOUT_PROMETHEUS = textwrap.dedent(
    """
    import sys

    # None in sys.modules makes the import statement raise ImportError, which is
    # the branch gateway.metrics falls back on.
    sys.modules["prometheus_client"] = None

    import gateway.metrics as m

    assert m.PROMETHEUS_AVAILABLE is False, "expected the fallback branch"

    # Each metric is declared beside the code that increments it, so the absent
    # case has to hold across those modules rather than in gateway.metrics alone.
    from gateway.api import deps
    from gateway.api.routes import _pipeline, _platform
    from gateway.core import database
    from gateway import rate_limit
    from gateway.services import budget_service, log_writer

    # Every recorder stays callable, so a caller on a hot path needs no guard.
    _pipeline.record_tokens("prov", "model", 100, 50)
    _pipeline.record_cost("prov", "model", 0.25)
    _pipeline.record_inline_cost_settlement("attached")
    _platform.record_abandoned_attempt("prov", "model", "timeout", 0)
    deps.record_auth_failure("invalid_key")
    rate_limit.RATE_LIMIT_HITS.inc()
    budget_service.BUDGET_EXCEEDED.inc()
    log_writer.QUEUE_DEPTH.set(3)
    log_writer.BATCH_SIZE.labels(writer="usage").observe(12)
    log_writer.ROWS.labels(writer="usage", result="ok").inc()

    # labels() returns the same stand-in rather than a child, so chaining works.
    assert _pipeline.TOKENS.labels(provider="a", model="b", type="input") is _pipeline.TOKENS

    # The custom pool collector subclasses Collector and is registered at import
    # time; both have to survive the fallback, and its samples still build.
    collector = database._PoolCollector()
    m.REGISTRY.register(collector)

    assert m.generate_latest(m.REGISTRY) == b""

    print("OK")
    """
)


def test_metrics_module_imports_without_prometheus_client() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _WITHOUT_PROMETHEUS],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "OK" in result.stdout


def test_prometheus_is_available_in_the_dev_environment() -> None:
    """Guards the inverse of the test above: the dev group must install the extra.

    Without it the family-pinning tests in ``test_gateway_metrics.py`` would pass
    against no-op metrics that expose nothing.
    """
    from gateway.metrics import PROMETHEUS_AVAILABLE

    assert PROMETHEUS_AVAILABLE is True


def test_enable_metrics_without_prometheus_refuses_to_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gateway.metrics.PROMETHEUS_AVAILABLE", False)
    config = GatewayConfig(enable_metrics=True)

    with pytest.raises(ValueError, match=r"pip install gateway\[metrics\]"):
        _validate_metrics_support(config)


def test_metrics_disabled_without_prometheus_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default (metrics off, extra absent) is the silent no-op path, not an error."""
    monkeypatch.setattr("gateway.metrics.PROMETHEUS_AVAILABLE", False)
    config = GatewayConfig(enable_metrics=False)

    _validate_metrics_support(config)


def test_enable_metrics_with_prometheus_starts() -> None:
    _validate_metrics_support(GatewayConfig(enable_metrics=True))
