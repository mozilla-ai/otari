"""The search and fetch tool calls one request may make, shared by every routing attempt (design H, step 1).

A search tool call and a fetch tool call each take a slot, an invalid one
included; enriching a search's hits takes none. Past ``web_search_max_calls``
the call is refused with a tool error the model can recover from, and the
request goes on.
"""

SEARCH_CALL_LIMIT_TEXT = "[tool error] web search call limit reached for this request"
FETCH_CALL_LIMIT_TEXT = "[tool error] web fetch call limit reached for this request"


class WebToolCallCap:
    """One request's allowance of search and fetch tool calls."""

    __slots__ = ("_limit", "_used")

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._used = 0

    @property
    def used(self) -> int:
        """The slots taken so far."""
        return self._used

    def claim(self) -> bool:
        """Take a slot for one call; ``False``, taking none, once every slot is taken."""
        if self._used >= self._limit:
            return False
        self._used += 1
        return True
