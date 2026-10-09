"""Where a request's trace is kept while it is served.

A request registers its :class:`RequestTrace` when the preamble has resolved who
it belongs to, and the trace middleware takes it back once the response has been
sent, which is the one point every request passes exactly once, streamed or not,
refused or served.
"""

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
