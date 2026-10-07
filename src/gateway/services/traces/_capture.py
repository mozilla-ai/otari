"""Where a request's trace is kept while it is served, and how usage rows reach it.

A request registers its :class:`RequestTrace` when the preamble has resolved who
it belongs to, and the trace middleware takes it back once the response has been
sent, which is the one point every request passes exactly once, streamed or not,
refused or served. In between, each usage row settlement writes for the request
becomes an LLM span on its trace, so the span carries exactly what billing
recorded without any settlement path having to know traces exist.
"""

from gateway.log_config import logger
from gateway.models.usage import UsageLog
from gateway.services.log_writer import LogWriter
from gateway.services.traces._collector import RequestTrace


class RequestTraces:
    """The traces of the requests this process is serving, by ``Otari-Request-ID``.

    No lock: every mutation is a single dict operation on the event loop's
    thread. An entry lives as long as its response, so the map is bounded by
    real concurrency, as the in-flight registry is.
    """

    def __init__(self) -> None:
        self._open: dict[str, RequestTrace] = {}

    def begin(self, trace: RequestTrace) -> None:
        self._open[trace.request_id] = trace

    def get(self, request_id: str | None) -> RequestTrace | None:
        return self._open.get(request_id) if request_id is not None else None

    def finish(self, request_id: str | None) -> RequestTrace | None:
        """Take a request's trace out of the registry. Tolerates an id never registered."""
        return self._open.pop(request_id, None) if request_id is not None else None

    def __len__(self) -> int:
        return len(self._open)

    def clear(self) -> None:
        """Forget every open trace. For a test boot on a registry that outlives the boot."""
        self._open.clear()


class TracingLogWriter:
    """A usage log writer that also records each row as an LLM span of its request's trace.

    Wraps whichever writer the deployment runs and changes nothing about how a row
    is written: the span is recorded first, in memory, and the row then goes to
    the wrapped writer exactly as before.
    """

    def __init__(self, inner: LogWriter, traces: RequestTraces) -> None:
        self._inner = inner
        self._traces = traces

    async def put(self, log: UsageLog) -> None:
        trace = self._traces.get(log.request_group_id)
        if trace is not None:
            # The row is the billing record: tracing it must never be why it is not written.
            try:
                trace.record_llm_call(log)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Usage row not traced (%s)", type(exc).__name__)
        await self._inner.put(log)

    async def start(self) -> None:
        await self._inner.start()

    async def stop(self) -> None:
        await self._inner.stop()
