import type { IconType } from "react-icons"
import {
  FiCornerDownRight,
  FiCpu,
  FiLink,
  FiMessageSquare,
  FiShield,
  FiShuffle,
  FiTool,
  FiUser,
} from "react-icons/fi"
import type { TraceSpan } from "@/client"
import { Chip } from "@/design-system/indicators/Chip"
import {
  type SpanNode,
  spanLabel,
  spanOutcomeLabel,
  spanTone,
  type TraceTree,
  turnStatus,
} from "@/features/traces/traceModel"
import {
  formatCost,
  formatLatency,
  formatTokens,
} from "@/shared/helpers/format"

const KIND_ICON: Record<string, IconType> = {
  step: FiMessageSquare,
  llm: FiCpu,
  tool: FiTool,
  routing_attempt: FiShuffle,
  guardrail: FiShield,
  mcp_connect: FiLink,
  agent: FiUser,
}

function SpanFacts({ span }: { span: TraceSpan }) {
  const duration = formatLatency(span.duration_ms)
  return (
    <span className="flex flex-wrap items-center gap-x-3 gap-y-1 text-caption tabular-nums">
      {duration ? (
        <span>{span.approximate ? `≈ ${duration}` : duration}</span>
      ) : null}
      {span.cost != null && span.cost > 0 ? (
        <span>{formatCost(span.cost)}</span>
      ) : null}
      {span.input_tokens != null ? (
        <span>
          {formatTokens((span.input_tokens ?? 0) + (span.output_tokens ?? 0))}{" "}
          tokens
        </span>
      ) : null}
    </span>
  )
}

function SpanRow({
  node,
  selectedSpanId,
  onSelect,
}: {
  node: SpanNode
  selectedSpanId: string
  onSelect: (spanId: string) => void
}) {
  const { span } = node
  const Icon = KIND_ICON[span.kind] ?? FiCornerDownRight
  const isSelected = span.span_id === selectedSpanId
  return (
    <li className="flex flex-col gap-1">
      <button
        type="button"
        aria-current={isSelected ? "true" : undefined}
        onClick={() => onSelect(span.span_id)}
        className={`flex min-h-11 w-full min-w-0 items-center gap-3 rounded-md px-2 py-1.5 text-left transition-colors motion-reduce:transition-none ${
          isSelected ? "bg-primary-subtle" : "hover:bg-surface-alt"
        }`}
      >
        <Icon aria-hidden className="size-4 shrink-0 text-muted" />
        <span className="flex min-w-0 flex-1 flex-col gap-0.5">
          <span className="truncate text-body">{spanLabel(span)}</span>
          <SpanFacts span={span} />
        </span>
        {span.outcome !== "ok" ? (
          <Chip tone={spanTone(span)} size="sm" className="shrink-0">
            {spanOutcomeLabel(span)}
          </Chip>
        ) : null}
      </button>
      {node.children.length > 0 ? (
        <SpanList
          nodes={node.children}
          selectedSpanId={selectedSpanId}
          onSelect={onSelect}
          nested
        />
      ) : null}
    </li>
  )
}

function SpanList({
  nodes,
  selectedSpanId,
  onSelect,
  nested = false,
}: {
  nodes: SpanNode[]
  selectedSpanId: string
  onSelect: (spanId: string) => void
  nested?: boolean
}) {
  return (
    <ul
      className={`flex flex-col gap-1 ${nested ? "ml-3 border-l border-border pl-3" : ""}`}
    >
      {nodes.map((node) => (
        <SpanRow
          key={node.span.span_id}
          node={node}
          selectedSpanId={selectedSpanId}
          onSelect={onSelect}
        />
      ))}
    </ul>
  )
}

// The session as a tree: each turn, the requests it made, and inside each request
// the LLM calls, tools, routing attempts and checks that ran for it.
export function SpanTree({
  tree,
  selectedSpanId,
  onSelect,
}: {
  tree: TraceTree
  selectedSpanId: string
  onSelect: (spanId: string) => void
}) {
  return (
    <div className="flex flex-col gap-4">
      {tree.turns.map(({ turn, steps }) => {
        const status = turnStatus(turn)
        return (
          <section
            key={turn.index}
            aria-label={`Turn ${turn.index + 1}`}
            className="flex flex-col gap-2"
          >
            <header className="flex flex-wrap items-center gap-2">
              <h4 className="text-title">
                {turn.continued ? "Earlier steps" : `Turn ${turn.index + 1}`}
              </h4>
              <Chip tone={status.tone} size="sm">
                {status.label}
              </Chip>
              <span className="text-caption tabular-nums">
                {turn.llm_calls} LLM · {turn.tool_calls} tools ·{" "}
                {formatCost(turn.cost)}
              </span>
            </header>
            <SpanList
              nodes={steps}
              selectedSpanId={selectedSpanId}
              onSelect={onSelect}
            />
          </section>
        )
      })}
      {tree.roots.length > 0 ? (
        <section aria-label="Agent spans" className="flex flex-col gap-2">
          <h4 className="text-title">Agent spans</h4>
          <SpanList
            nodes={tree.roots}
            selectedSpanId={selectedSpanId}
            onSelect={onSelect}
          />
        </section>
      ) : null}
    </div>
  )
}
