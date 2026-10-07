import type { TraceDetail, TraceSpan, TraceSummary } from "@/client"

export function traceSummary(
  overrides: Partial<TraceSummary> = {},
): TraceSummary {
  return {
    trace_id: "s-0123456789abcdef",
    workspace_id: "22222222-2222-2222-2222-222222222222",
    user_id: "alice",
    api_key_id: null,
    session_source: "harness",
    harness: "claude-code",
    name: null,
    started_at: "2026-10-07T12:00:00Z",
    last_activity_at: "2026-10-07T12:01:10Z",
    step_count: 2,
    span_count: 5,
    error_count: 0,
    input_tokens: 20,
    output_tokens: 4,
    cost: 0.02,
    ...overrides,
  }
}

export function traceSpan(overrides: Partial<TraceSpan> = {}): TraceSpan {
  return {
    span_id: "req-1",
    parent_span_id: null,
    kind: "step",
    origin: "gateway",
    name: "step",
    operation: null,
    start_time: "2026-10-07T12:00:00Z",
    end_time: "2026-10-07T12:00:10Z",
    duration_ms: 10_000,
    approximate: false,
    outcome: "ok",
    recovered: false,
    error_class: null,
    opens_turn: false,
    model: null,
    provider: null,
    input_tokens: null,
    output_tokens: null,
    cost: null,
    tool_name: null,
    tool_type: null,
    tool_call_id: null,
    request_id: null,
    attributes: {},
    has_content: false,
    ...overrides,
  }
}

// One turn of two requests: an LLM call each, and a tool the agent ran between them.
export function traceDetail(overrides: Partial<TraceDetail> = {}): TraceDetail {
  return {
    summary: traceSummary(),
    state: "idle",
    turns: [
      {
        index: 0,
        state: "completed",
        continued: false,
        started_at: "2026-10-07T12:00:00Z",
        ended_at: "2026-10-07T12:01:10Z",
        step_ids: ["req-1", "req-2"],
        tool_calls: 1,
        llm_calls: 2,
        errors: 0,
        input_tokens: 20,
        output_tokens: 4,
        cost: 0.02,
      },
    ],
    spans: [
      traceSpan({ span_id: "req-1", opens_turn: true }),
      traceSpan({
        span_id: "llm-1",
        parent_span_id: "req-1",
        kind: "llm",
        name: "chat",
        model: "gpt-5",
        input_tokens: 10,
        output_tokens: 2,
        cost: 0.01,
        duration_ms: 9_000,
      }),
      traceSpan({
        span_id: "call-toolu_1",
        parent_span_id: "req-1",
        kind: "tool",
        name: "execute_tool:Bash",
        tool_name: "Bash",
        tool_type: "client",
        approximate: true,
        start_time: "2026-10-07T12:00:10Z",
        end_time: "2026-10-07T12:01:00Z",
        duration_ms: 50_000,
      }),
      traceSpan({
        span_id: "req-2",
        start_time: "2026-10-07T12:01:00Z",
        end_time: "2026-10-07T12:01:10Z",
      }),
      traceSpan({
        span_id: "llm-2",
        parent_span_id: "req-2",
        kind: "llm",
        name: "chat",
        model: "gpt-5",
        outcome: "error",
        recovered: true,
        start_time: "2026-10-07T12:01:00Z",
      }),
    ],
    truncated: false,
    ...overrides,
  }
}
