# Bring MCP / sandbox / web_search / routing parity to `/v1/messages` and `/v1/responses`

## Context

`/v1/chat/completions` (`src/gateway/api/routes/chat.py`) supports all four recently added features — MCP tool-loop, sandbox code execution, SearXNG web search, and platform-mode multi-provider routing/fallback — across both streaming and non-streaming paths.

`/v1/messages` (Anthropic, `routes/messages.py`) and `/v1/responses` (OpenAI Responses, `routes/responses.py`) currently just pass kwargs through to `amessages` / `aresponses`. None of those four features are wired in, on either stream or non-stream. A user invoking the Anthropic-shaped API with `tools: [{"type": "code_execution_20250825"}]` gets whatever the provider does natively — there is no gateway-side sandbox dispatch, no MCP-server orchestration, no SearXNG bridge, no multi-attempt fallback.

The root cause is format coupling: `mcp_tool_loop` / `mcp_tool_loop_stream` (`src/gateway/services/mcp_loop.py`) speak only OpenAI chat-completions message shape (`tool_calls`, `role:"tool"`). The decision (per user) is **native loops per format** — write Anthropic-shaped and Responses-shaped tool loops that share the existing `pool` duck-type (`openai_tools` / `owns_tool` / `purpose_hints` / `call_tool`) and reuse the already-shape-agnostic streaming fallback (`iterate_streaming_attempts` in `gateway/streaming.py`) and the chat.py non-stream `route.attempts` lock-in pattern.

Outcome: a request sent to any of the three endpoints with gateway-managed tools (or in platform mode with multiple route attempts) gets identical behavior — the only differences are wire shape on the boundary.

## Approach

Three pieces of work, sequenced so each is independently testable.

### 1. Per-format tool-loop functions

Add two new files alongside `src/gateway/services/mcp_loop.py`:

- `src/gateway/services/mcp_loop_messages.py` — `anthropic_tool_loop` and `anthropic_tool_loop_stream`
- `src/gateway/services/mcp_loop_responses.py` — `responses_tool_loop` and `responses_tool_loop_stream`

Each new module:

- Mirrors the public signatures of `mcp_tool_loop` / `mcp_tool_loop_stream`: `completion_kwargs: dict`, `pool` (duck-typed), `max_iterations: int`, and (non-stream only) `on_first_response: Callable[[], None] | None = None` so platform-mode pre-lock-in fallback works the same way.
- Calls `amessages(**kwargs)` / `aresponses(**kwargs)` instead of `acompletion`.
- Reuses `MAX_TOOL_ITERATIONS_CAP`, `DEFAULT_MAX_TOOL_ITERATIONS`, and `MaxToolIterationsExceeded` from `mcp_loop.py` (re-export, don't duplicate).
- Reuses `pool.owns_tool` / `pool.call_tool` / `pool.purpose_hints` unchanged.

#### Tool-definition format conversion

`pool.openai_tools` returns `[{"type":"function","function":{"name","description","parameters"}}]`. We need two converters next to `mcp_loop.py` (one helper module: `src/gateway/services/tool_format.py`):

- `openai_to_anthropic_tools(tools)` → `[{"name","description","input_schema": parameters}]`
- `openai_to_responses_tools(tools)` → `[{"type":"function","name","description","parameters"}]` (flat Responses shape — verify against `any_llm.types.responses` before locking)

Standalone converter avoids touching `MCPClientPool` / `SandboxBackend` / `WebSearchBackend`. If the Responses shape turns out to differ in non-trivial ways across providers (e.g. `tools` accepts native shape only for OpenAI, with adapters elsewhere), promote it to backend-side properties later.

#### Anthropic non-stream loop (`anthropic_tool_loop`)

Loop body per iteration:

1. Call `amessages(**kwargs)` → `MessageResponse`.
2. After the first successful response, fire `on_first_response` once (matches `mcp_tool_loop` line 226 lock-in semantics).
3. Walk `result.content` for `ToolUseBlock` entries (`type == "tool_use"`).
4. Partition into gateway-owned (`pool.owns_tool(block.name) is True`) vs foreign. If any foreign blocks exist, return the response unchanged (matches chat.py "foreign tool returns to caller" behavior at `mcp_loop.py:201`).
5. For each owned block: `output = await pool.call_tool(block.name, block.input)`; collect `{"type":"tool_result","tool_use_id": block.id,"content": output}`.
6. Append assistant message `{"role":"assistant","content": result.content}` and user message `{"role":"user","content": tool_results}` to `kwargs["messages"]`.
7. Fold usage tokens (`result.usage.input_tokens` / `output_tokens`) into a running total just like `mcp_loop.py` does.
8. If `result.stop_reason != "tool_use"` or no owned blocks: return the final `MessageResponse`, with the accumulated usage merged back into it.

Raise `MaxToolIterationsExceeded` if the loop hits `max_iterations` without a non-tool stop.

#### Anthropic stream loop (`anthropic_tool_loop_stream`)

This is the hardest piece. Per round:

1. Set `kwargs["stream"] = True`, call `amessages(...)`, iterate the event stream.
2. Forward every event downstream **except** those belonging to a tool_use content block (track via `content_block_start.content_block.type == "tool_use"` keyed on `index`).
3. For tool_use blocks: buffer `input_json_delta.partial_json` chunks until `content_block_stop`; parse JSON, store `{id, name, input}`.
4. On `message_stop`: if any buffered tool_use blocks exist AND all owned by the pool, execute them, append messages exactly like the non-stream loop, **suppress** the terminal `message_delta`/`message_stop` from being forwarded (so the client sees one continuous stream), and start the next round.
5. If foreign tool_use blocks exist, forward everything (including the buffered tool_use events) and exit.
6. If no tool_use: forward `message_stop` and exit.

Adopt the same "buffer + drop terminal chunk if loop continues; forward if exiting" pattern documented at `mcp_loop.py:194-197`.

#### Responses non-stream loop (`responses_tool_loop`)

1. Call `aresponses(**kwargs)` → `Response`.
2. Fire `on_first_response` once.
3. Walk `result.output` for items with `type == "function_call"` (gateway-managed tools are always exposed as function tools; native server tools like `web_search_call`/`code_interpreter_call` would belong to the upstream provider and we don't intercept them).
4. Partition by `pool.owns_tool(item.name)`. If foreign present, return unchanged.
5. For each owned call: `output = await pool.call_tool(item.name, json.loads(item.arguments))`; append `{"type":"function_call_output","call_id": item.call_id,"output": output}` to `kwargs["input_data"]` (which is a list of input items per Responses API).
6. Append the assistant `function_call` items to the input as well so the next call sees the full history.
7. Loop until no owned `function_call` items or `result.status == "completed"` with no tool calls.

Fold `result.usage` across rounds.

#### Responses stream loop (`responses_tool_loop_stream`)

Mirror the Anthropic stream design. Forward events except for owned `function_call` items; accumulate `response.function_call_arguments.delta` until `response.output_item.done` for that item; on `response.completed`, if buffered owned calls exist, execute and start the next round (suppress terminal events). If foreign, forward and exit.

Stream event names from `openai.types.responses` — confirm exact strings during implementation (`output_item.added`, `function_call_arguments.delta`, `output_item.done`, `completed`).

#### Purpose-hint injection

`inject_purpose_hints` (`mcp_loop.py:45`) operates on chat-completions messages and merges into a system message. For the other formats:

- **Anthropic** (`messages.py`): `system` is a separate top-level field on the request. New helper `inject_purpose_hints_anthropic(call_kwargs, hints, header)`: builds the hint preamble, prepends to `call_kwargs["system"]` (handling both string and content-block-list cases).
- **Responses** (`responses.py`): `instructions` is a separate field; if absent, prepend to the first text input. New helper `inject_purpose_hints_responses(call_kwargs, hints, header)`.

Both helpers live alongside the existing one in `mcp_loop.py` (or `tool_format.py`) and share the header-resolution logic (env `GATEWAY_TOOLS_HEADER`, etc.).

### 2. Wire features into the endpoint handlers

#### `routes/messages.py`

Refactor `create_message` to mirror `routes/chat.py:855-960` (tool extraction) and `:1409-1462` (non-stream dispatch) / `:945` (stream dispatch via `_run_streaming_with_fallback`):

- Extract `MessagesRequest.tools` for `code_execution_*` and `web_search_*` types using the existing `_extract_code_execution_tool` / `_extract_web_search_tool` helpers (already format-agnostic — they only look at the `type` string). Move them to a shared `chat_tools.py` if they currently live in `chat.py`. Also recognize `mcp_servers` if/when MessagesRequest grows that field — for the first cut, gate it behind a `mcp_servers: list[dict[str, Any]] | None = None` addition to the pydantic model so Anthropic's native shape stays valid.
- Build the backend (`MCPClientPool` / `SandboxBackend` / `_build_web_search_backend`) using the same `async with` pattern.
- Convert the pool's `openai_tools` via `openai_to_anthropic_tools`, set `call_kwargs["tools"]` to the result (leaving any user-supplied function tools merged in if applicable — match chat.py's behavior in `_strip_gateway_fields`).
- Call `anthropic_tool_loop(...)` (non-stream) or `anthropic_tool_loop_stream(...)` (stream) where the current `amessages(...)` calls are.
- All existing usage-logging, error handling, and rate-limit-header code stays exactly the same — only the inner call changes.

#### `routes/responses.py`

Same structure with `responses_tool_loop` / `responses_tool_loop_stream`. Verify whether `aresponses` accepts `tools` for non-OpenAI providers (it likely doesn't — gate by `provider_class.SUPPORTS_RESPONSES_TOOLS` if such a flag exists, otherwise let the upstream error surface).

### 3. Platform-mode routing/fallback for the two new endpoints

`iterate_streaming_attempts` (`gateway/streaming.py:117`) and the non-stream `route.attempts` loop in chat.py are already message-shape-agnostic — they only care about `build_stream` / `acompletion`-equivalent callable, first-chunk timeout, retryable-error classification.

For the new endpoints, factor the non-stream attempt loop out of `chat.py:1178-1392` into a shared helper. The chat.py inner body is mostly: build `pool` once, iterate `route.attempts`, for each attempt construct `completion_kwargs`, call the loop, classify error / break / continue, track `locked_in`. Extract:

- `src/gateway/api/routes/_platform.py` (new) — `async def run_platform_attempts(*, route, build_kwargs_for_attempt, run_attempt, classify_error) -> Any`. `run_attempt` is the per-format coroutine factory (returns `anthropic_tool_loop(...)` for messages, `responses_tool_loop(...)` for responses, `mcp_tool_loop(...)` for chat).
- `_run_streaming_with_fallback` in chat.py already takes `build_stream_for_attempt`-style callbacks; either generalize it the same way or write `_run_streaming_with_fallback_messages` / `_responses` that share `iterate_streaming_attempts` and differ only in the per-attempt builder.

Keep the extraction conservative — start with code-duplication-with-different-callable-injection if the abstraction starts pulling in unrelated chat-specific logic. The shape-agnostic core (`iterate_streaming_attempts`) doesn't need to change at all.

Provider resolution for the new endpoints reuses whatever chat.py uses today (look for `route = await _resolve_platform_route(...)` around `chat.py:925`).

## Critical files

**Edit:**
- `src/gateway/api/routes/messages.py` — wire tool extraction + native loops + platform routing.
- `src/gateway/api/routes/responses.py` — same.
- `src/gateway/api/routes/chat.py` — extract shared helpers (tool extraction, platform attempts) so messages/responses can import them without circular deps. Move `_extract_code_execution_tool`, `_extract_web_search_tool`, `_is_*_tool_type` to `chat_tools.py` (new) or `_helpers.py`.

**New:**
- `src/gateway/services/mcp_loop_messages.py`
- `src/gateway/services/mcp_loop_responses.py`
- `src/gateway/services/tool_format.py` — converters and per-format `inject_purpose_hints_*`.
- `src/gateway/api/routes/_platform.py` (or extend `_helpers.py`) — shared platform-attempts runner.

**Reuse (no edits):**
- `src/gateway/services/mcp_loop.py` — `MAX_TOOL_ITERATIONS_CAP`, `DEFAULT_MAX_TOOL_ITERATIONS`, `MaxToolIterationsExceeded`, `inject_purpose_hints`.
- `src/gateway/services/mcp_pool.py`, `sandbox_backend.py`, `web_search_backend.py` — duck-typed pool interface; do not modify.
- `src/gateway/streaming.py` — `iterate_streaming_attempts`, `streaming_generator`, `ANTHROPIC_STREAM_FORMAT`, `RESPONSES_STREAM_FORMAT`.

**New tests:**
- `tests/unit/test_mcp_loop_messages.py` — mirror every test in `tests/unit/test_mcp_loop.py` (18 tests) for the Anthropic loop: empty/text-only path, tool execution, multi-round, max-iter, foreign tool, mixed tools, `on_first_response` semantics, stream-vs-non-stream parity, tool-failure-as-message.
- `tests/unit/test_mcp_loop_responses.py` — same set for Responses.
- `tests/integration/test_platform_mode_messages.py` — mirror `test_platform_mode_chat.py`'s `test_platform_mode_tool_loop_*` trio (non-stream pre-lock-in fallback, non-stream no-fallback-after-lock-in, streaming pre-lock-in fallback).
- `tests/integration/test_platform_mode_responses.py` — same trio.
- `tests/integration/test_messages_sandbox.py` / `test_messages_web_search.py` / `test_messages_mcp.py` — standalone-mode integration for each tool (mirroring whatever lives for chat.py).
- `tests/integration/test_responses_sandbox.py` / `test_responses_web_search.py` / `test_responses_mcp.py` — same.

## Suggested PR breakdown

Landed as one PR this is ~4500-5500 LOC added (most of it tests) plus 200-400 LOC moved. Four sequential PRs along these seams:

### PR 1 — Helper extraction (warmup, no behavior change)

- Move `_is_web_search_tool_type`, `_is_code_execution_tool_type`, `_extract_code_execution_tool`, `_extract_web_search_tool`, `_resolve_sandbox_purpose_hint`, `_resolve_web_search_purpose_hint`, `_build_web_search_backend`, `_strip_gateway_fields` out of `chat.py` into a new `src/gateway/api/routes/_tools.py` (or extend `_helpers.py`).
- Move the non-stream `route.attempts` loop body (`chat.py:1178-1392`) into `src/gateway/api/routes/_platform.py` as a generic `run_platform_attempts(route, build_attempt, run_attempt, classify_error)` runner. `chat.py` becomes the first caller. Streaming side stays in `chat.py` for now.
- ~300 LOC moved, 0 LOC of new behavior. Full existing test suite must pass unchanged.

### PR 2 — Anthropic messages parity (standalone mode only)

- Add `src/gateway/services/tool_format.py` with `openai_to_anthropic_tools` and `inject_purpose_hints_anthropic`.
- Add `src/gateway/services/mcp_loop_messages.py` with `anthropic_tool_loop` and `anthropic_tool_loop_stream`.
- Wire MCP / sandbox / web_search into `routes/messages.py` for both stream and non-stream **standalone-mode** paths only. No platform-mode routing yet — platform requests fall through to today's passthrough behavior.
- New tests: `tests/unit/test_mcp_loop_messages.py` (mirror all 18 cases from `test_mcp_loop.py`), `tests/integration/test_messages_{sandbox,web_search,mcp}.py` (mirror chat.py standalone integration).
- ~1500-2000 LOC.

### PR 3 — Responses parity (standalone mode only)

- Add `openai_to_responses_tools` and `inject_purpose_hints_responses` to `tool_format.py`.
- Add `src/gateway/services/mcp_loop_responses.py` with `responses_tool_loop` and `responses_tool_loop_stream`.
- Wire all three tool modes into `routes/responses.py`'s stream + non-stream standalone paths.
- New tests: `tests/unit/test_mcp_loop_responses.py`, `tests/integration/test_responses_{sandbox,web_search,mcp}.py`.
- Symmetric with PR 2; can be done in parallel by a different engineer if needed.
- ~1500-2000 LOC.

### PR 4 — Platform-mode routing for the two new endpoints

- Generalize `_run_streaming_with_fallback` in `chat.py` so it accepts a per-format builder, or add `_run_streaming_with_fallback_messages` / `_responses` that share `iterate_streaming_attempts` with chat.py.
- Use the `run_platform_attempts` runner from PR 1 to handle non-stream lock-in / pre-lock-in fallback in messages.py and responses.py.
- New tests: `tests/integration/test_platform_mode_messages.py` and `test_platform_mode_responses.py`, each mirroring the three-test fallback suite from `test_platform_mode_chat.py` (pre-lock-in fallback non-stream, no-fallback-after-lock-in, pre-lock-in fallback streaming).
- ~800-1200 LOC.

### Notes on the breakdown

- **Why standalone before platform**: the tool-loop is the load-bearing piece; once it exists and is tested in standalone mode, platform routing is "wrap the per-attempt call". Doing platform first would mean the runner has nothing format-specific to dispatch to and would just be scaffolding.
- **Why split Anthropic from Responses**: they share zero code except `tool_format.py` (added in PR 2, extended in PR 3). The two streams have different event shapes and error modes — reviewing one in isolation is much easier than reviewing them together.
- **Risk concentration**: PR 2 and PR 3 each introduce one new streaming state machine. Those are the PRs that need the most careful review and the most aggressive test coverage. PR 1 and PR 4 are mostly mechanical.

## Verification

1. **Lint and type check**: `make lint` (or `uv run ruff check` + `uv run mypy src`).
2. **Unit tests**: `uv run pytest tests/unit/test_mcp_loop_messages.py tests/unit/test_mcp_loop_responses.py -v`. Must reach test-parity with `test_mcp_loop.py`.
3. **Integration tests**: `uv run pytest tests/integration/test_platform_mode_messages.py tests/integration/test_platform_mode_responses.py -v` — focus on the lock-in semantics (the same way chat.py tests do).
4. **Full suite**: `uv run pytest` — make sure chat.py behavior is untouched.
5. **Local manual smoke** with `docker-compose up`:
   - Send `/v1/messages` request with `tools: [{"type":"code_execution_20250825"}]`. Confirm the sandbox executes Python (set `GATEWAY_SANDBOX_URL`).
   - Repeat with `tools: [{"type":"web_search_20250305"}]` (set `GATEWAY_WEB_SEARCH_URL`).
   - Send `/v1/messages` with `mcp_servers: [...]`. Confirm the MCP tool round-trip.
   - Repeat all three for `/v1/responses`.
   - Run each in both `stream: true` and `stream: false` mode.
   - In platform mode with a route that lists two providers and the first set to fail: confirm fallback happens on both endpoints, both stream and non-stream.

## Open question to settle during implementation

`MessagesRequest` doesn't currently accept `mcp_servers`. Adding it would mean diverging from Anthropic's wire shape (Anthropic has its own `mcp_servers` field at the top level of the Messages API — confirm whether the gateway can passthrough-compatible naming). Same question for `ResponsesRequest` (which uses `extra="allow"`, so the field is already accepted but unvalidated). If Anthropic's native `mcp_servers` shape matches what `MCPClientPool` expects, we may be able to use it directly without a model change.
