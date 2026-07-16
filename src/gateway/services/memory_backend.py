"""Dispatch memory tool calls to the platform's memory endpoints.

A backend the tool-use loop in :mod:`gateway.services.mcp_loop` dispatches to whenever the
model emits a ``memory_search`` / ``memory_store`` / ``memory_forget`` call. The gateway
advertises these tools to the model (when the request opts in with an ``otari_memory`` tool
entry and the workspace has memory enabled), intercepts the calls, and forwards them to the
platform's dual-auth ``/gateway/memory/{search,store,forget}`` endpoints, where the mem0
engine and all tenant memory data live.

Memory is platform-only: resolving the caller's workspace + member needs both the gateway
token (machine identity) and the user token, so this backend is wired only in hybrid mode.

This backend satisfies the same duck-typed protocol the MCP loop uses for tool dispatch
(``openai_tools``, ``owns_tool``, ``purpose_hints``, ``call_tool``), so the loop accepts it
as a ``pool`` without any refactor.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)

MEMORY_SEARCH_TOOL_NAME = "memory_search"
MEMORY_STORE_TOOL_NAME = "memory_store"
MEMORY_FORGET_TOOL_NAME = "memory_forget"
_MEMORY_TOOL_NAMES = frozenset({MEMORY_SEARCH_TOOL_NAME, MEMORY_STORE_TOOL_NAME, MEMORY_FORGET_TOOL_NAME})

_DEFAULT_TIMEOUT_S = 10.0

_DEFAULT_PURPOSE_HINT = (
    "Use `memory_search` to recall durable facts the user shared in earlier sessions before "
    "answering a question that depends on them; `memory_store` to save a new durable fact the "
    "user shares about themselves or their preferences; `memory_forget` to delete a stored "
    "memory (by an id returned from memory_search or memory_store) when the user asks you to."
)


class MemoryNotReachableError(RuntimeError):
    """Raised when the platform memory endpoint can't be reached or returns malformed data."""


class MemoryBackend:
    """Async context manager that owns an HTTP client for the platform memory endpoints.

    Usage::

        async with MemoryBackend(base_url="http://backend:8000/api/v1/gateway/memory",
                                  gateway_token=gw, user_token=tk) as backend:
            result = await mcp_tool_loop(completion_kwargs=kwargs, pool=backend, max_iterations=N)
    """

    def __init__(
        self,
        *,
        base_url: str,
        gateway_token: str | None,
        user_token: str,
        purpose_hint: str | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._gateway_token = gateway_token
        self._user_token = user_token
        self._purpose_hint = purpose_hint or _DEFAULT_PURPOSE_HINT
        self._timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None
        self._stack: AsyncExitStack = AsyncExitStack()

    async def __aenter__(self) -> MemoryBackend:
        self._client = await self._stack.enter_async_context(httpx.AsyncClient(timeout=self._timeout_s))
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        await self._stack.aclose()

    # ----- duck-typed protocol the MCP loop uses on `pool` -----

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": MEMORY_SEARCH_TOOL_NAME,
                    "description": (
                        "Search your long-term memory for durable facts relevant to a query "
                        "(things the user told you in earlier sessions). Returns matching "
                        "memories, each with an id you can pass to memory_forget."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What to recall. Use natural language.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": MEMORY_STORE_TOOL_NAME,
                    "description": (
                        "Save a durable fact to long-term memory so it can be recalled in future "
                        "conversations. Store one concise, self-contained fact per call."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The fact to remember, phrased so it stands alone.",
                            },
                        },
                        "required": ["content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": MEMORY_FORGET_TOOL_NAME,
                    "description": (
                        "Delete a memory you previously stored, identified by an id returned from "
                        "memory_search or memory_store. Use only when the user asks to be forgotten."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "memory_id": {
                                "type": "string",
                                "description": "The id of the memory to delete.",
                            },
                        },
                        "required": ["memory_id"],
                    },
                },
            },
        ]

    def owns_tool(self, name: str) -> bool:
        return name in _MEMORY_TOOL_NAMES

    def purpose_hints(self) -> list[tuple[str, str]]:
        return [("memory", self._purpose_hint)]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if self._client is None:
            raise RuntimeError("MemoryBackend not entered as an async context manager")
        if name == MEMORY_SEARCH_TOOL_NAME:
            return await self._search(arguments)
        if name == MEMORY_STORE_TOOL_NAME:
            return await self._store(arguments)
        if name == MEMORY_FORGET_TOOL_NAME:
            return await self._forget(arguments)
        raise KeyError(f"MemoryBackend does not own tool {name!r}")

    # ----- internals -----

    def _headers(self) -> dict[str, str]:
        return {"X-Gateway-Token": self._gateway_token or "", "X-User-Token": self._user_token}

    async def _post(self, path: str, body: dict[str, Any]) -> httpx.Response:
        assert self._client is not None
        try:
            return await self._client.post(
                f"{self._base_url}{path}", headers=self._headers(), json=body, timeout=self._timeout_s
            )
        except httpx.HTTPError as exc:
            raise MemoryNotReachableError(f"memory call failed against {self._base_url}{path}: {exc}") from exc

    async def _search(self, arguments: dict[str, Any]) -> str:
        query = (arguments.get("query") or "").strip()
        if not query:
            return "[tool error] empty query"
        response = await self._post("/search", {"query": query})
        if response.status_code != 200:
            return "[tool error] memory search is unavailable"
        facts = response.json().get("facts") or []
        if not facts:
            return "No relevant memories found."
        lines = ["Relevant memories:"]
        for fact in facts:
            memory_id = fact.get("id")
            text = fact.get("memory") or ""
            lines.append(f"- (id: {memory_id}) {text}" if memory_id else f"- {text}")
        return "\n".join(lines)

    async def _store(self, arguments: dict[str, Any]) -> str:
        content = (arguments.get("content") or "").strip()
        if not content:
            return "[tool error] empty content"
        response = await self._post("/store", {"content": content})
        if response.status_code != 200:
            return "[tool error] could not store memory"
        payload = response.json()
        memory_id = payload.get("id")
        return f"Stored memory (id: {memory_id})." if memory_id else "Stored memory."

    async def _forget(self, arguments: dict[str, Any]) -> str:
        memory_id = (arguments.get("memory_id") or "").strip()
        if not memory_id:
            return "[tool error] missing memory_id"
        response = await self._post("/forget", {"memory_id": memory_id})
        if response.status_code == 204:
            return f"Deleted memory {memory_id}."
        if response.status_code == 404:
            return f"[tool error] no memory with id {memory_id} to delete"
        return "[tool error] could not delete memory"
