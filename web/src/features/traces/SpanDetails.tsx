import type { ReactNode } from "react"
import type { TraceSpan } from "@/client"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { spanLabel, spanOutcomeLabel } from "@/features/traces/traceModel"
import {
  formatCost,
  formatDateTime,
  formatLatency,
  formatTokens,
} from "@/shared/helpers/format"

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <dt className="text-caption">{label}</dt>
      <dd className="truncate text-body tabular-nums">{children}</dd>
    </div>
  )
}

// What one span says about itself, and, where a person would look for the prompt
// or the tool's output, why it is not there: no content is recorded.
export function SpanDetails({ span }: { span: TraceSpan }) {
  const duration = formatLatency(span.duration_ms)
  return (
    <div className="flex flex-col gap-4">
      <h4 className="text-title">{spanLabel(span)}</h4>
      <dl className="grid grid-cols-2 gap-3">
        <Fact label="Outcome">
          {spanOutcomeLabel(span)}
          {span.error_class ? ` (${span.error_class})` : ""}
        </Fact>
        <Fact label="Duration">
          {duration
            ? span.approximate
              ? `≈ ${duration}, inferred`
              : duration
            : "Unknown"}
        </Fact>
        <Fact label="Started">
          {span.start_time ? formatDateTime(span.start_time) : "Unknown"}
        </Fact>
        {span.model ? <Fact label="Model">{span.model}</Fact> : null}
        {span.provider ? <Fact label="Provider">{span.provider}</Fact> : null}
        {span.input_tokens != null ? (
          <Fact label="Tokens">
            {formatTokens(span.input_tokens)} in ·{" "}
            {formatTokens(span.output_tokens ?? 0)} out
          </Fact>
        ) : null}
        {span.cost != null ? (
          <Fact label="Cost">{formatCost(span.cost)}</Fact>
        ) : null}
        {span.tool_type ? (
          <Fact label="Ran by">
            {span.tool_type === "client" ? "The agent" : "The gateway"}
          </Fact>
        ) : null}
        {span.request_id ? (
          <Fact label="Request ID">{span.request_id}</Fact>
        ) : null}
      </dl>
      <div className="flex flex-col gap-2">
        <h5 className="text-overline">Input and output</h5>
        <EmptyMessage>
          Content is not captured. Traces record what ran, how long it took and
          what it cost, never the prompt, the model's output or a tool's input
          and output.
        </EmptyMessage>
      </div>
    </div>
  )
}
