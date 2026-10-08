import { useState } from "react"
import type { IconType } from "react-icons"
import {
  FiChevronDown,
  FiChevronRight,
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
import { IconButton } from "@/design-system/actions/IconButton"
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

// Each kind keeps one glyph and one categorical slot, in the slots' own order,
// so a tool reads as a tool at a glance anywhere in a long session. The label
// beside it always says the same thing in words.
const KIND_ICON: Record<string, { icon: IconType; tint: string }> = {
  step: { icon: FiMessageSquare, tint: "text-chart-cat-1" },
  tool: { icon: FiTool, tint: "text-chart-cat-2" },
  agent: { icon: FiUser, tint: "text-chart-cat-3" },
  guardrail: { icon: FiShield, tint: "text-chart-cat-4" },
  llm: { icon: FiCpu, tint: "text-chart-cat-5" },
  routing_attempt: { icon: FiShuffle, tint: "text-chart-cat-6" },
  mcp_connect: { icon: FiLink, tint: "text-chart-cat-7" },
}
const OTHER_KIND = { icon: FiCornerDownRight, tint: "text-chart-cat-other" }

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
  const { icon: Icon, tint } = KIND_ICON[span.kind] ?? OTHER_KIND
  const isSelected = span.span_id === selectedSpanId
  const hasChildren = node.children.length > 0
  const [isOpen, setIsOpen] = useState(true)
  return (
    <li className="flex flex-col gap-0.5">
      <div className="flex min-w-0 items-center">
        {/* A lane of its own on every row, so labels line up whether or not a
            row can collapse. */}
        <span className="flex w-11 shrink-0 justify-center">
          {hasChildren ? (
            <IconButton
              label={`${isOpen ? "Collapse" : "Expand"} ${spanLabel(span)}`}
              aria-expanded={isOpen}
              size="sm"
              className="text-subtle"
              onPress={() => setIsOpen((value) => !value)}
            >
              {isOpen ? (
                <FiChevronDown aria-hidden className="size-3.5" />
              ) : (
                <FiChevronRight aria-hidden className="size-3.5" />
              )}
            </IconButton>
          ) : null}
        </span>
        <button
          type="button"
          aria-current={isSelected ? "true" : undefined}
          onClick={() => onSelect(span.span_id)}
          className={`flex min-h-11 w-full min-w-0 items-center gap-3 rounded-md px-2 py-1.5 text-left transition-colors motion-reduce:transition-none ${
            isSelected ? "bg-primary-subtle" : "hover:bg-surface-alt"
          }`}
        >
          <Icon aria-hidden className={`size-4 shrink-0 ${tint}`} />
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
      </div>
      {hasChildren && isOpen ? (
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
      className={`flex flex-col gap-0.5 ${nested ? "ml-3 border-border-strong border-l pl-2" : ""}`}
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
