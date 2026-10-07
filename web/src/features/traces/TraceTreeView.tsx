import { useMemo, useState } from "react"
import { FiChevronRight } from "react-icons/fi"

import { IconButton } from "@/design-system/actions/IconButton"
import { Chip } from "@/design-system/indicators/Chip"
import { formatCost, formatNumber } from "@/shared/helpers/format"

import { formatDuration } from "./format"
import { TypeMark } from "./observationTypes"
import { buildTree, flattenTreeOrder } from "./tree"
import { summarizeTrace, type Observation, type Trace } from "./types"

function Metrics({
  durationMs,
  cost,
  tokens,
}: {
  durationMs: number
  cost: number
  tokens: number
}) {
  const duration = formatDuration(durationMs)
  if (!duration && !cost && !tokens) return null
  return (
    <span className="flex items-center gap-x-3 truncate text-mono-caption whitespace-nowrap text-muted">
      {duration && durationMs > 0 ? <span>{duration}</span> : null}
      {cost ? (
        <span>{formatCost(cost)}</span>
      ) : tokens ? (
        <span>{formatNumber(tokens)} tokens</span>
      ) : null}
    </span>
  )
}

function LevelChip({ level }: { level: Observation["level"] }) {
  if (level === "WARNING") return <Chip tone="warning">Warning</Chip>
  if (level === "ERROR") return <Chip tone="danger">Error</Chip>
  if (level === "DEBUG") return <Chip>Debug</Chip>
  return null
}

const ROW =
  "flex min-w-0 flex-1 flex-col gap-0.5 py-1 pr-2 pl-1 text-left outline-none focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-foreground"

export function TraceTreeView({
  trace,
  selectedId,
  onSelect,
}: {
  trace: Trace
  selectedId: string | null
  onSelect: (observationId: string | null) => void
}) {
  const [collapsed, setCollapsed] = useState<ReadonlySet<string>>(new Set())
  const roots = useMemo(() => buildTree(trace.observations), [trace])
  const items = useMemo(
    () => flattenTreeOrder(roots, collapsed),
    [roots, collapsed],
  )
  const summary = summarizeTrace(trace)

  const toggle = (id: string) =>
    setCollapsed((current) => {
      const next = new Set(current)
      if (!next.delete(id)) next.add(id)
      return next
    })

  return (
    <div role="tree" aria-label="Observations" className="flex flex-col py-1">
      <div
        role="treeitem"
        aria-selected={selectedId === null}
        aria-level={1}
        className={`flex items-start gap-1 pl-3 ${selectedId === null ? "bg-surface-subtle" : "hover:bg-surface-alt"}`}
      >
        <span className="grid size-6 shrink-0 place-items-center pt-1">
          <TypeMark type="TRACE" />
        </span>
        <button
          type="button"
          className={ROW}
          aria-label={`Trace, ${trace.name}, ${formatDuration(summary.latencyMs)}`}
          onClick={() => onSelect(null)}
        >
          <span className="truncate text-body">{trace.name}</span>
          <Metrics
            durationMs={summary.latencyMs}
            cost={summary.totalCost}
            tokens={summary.totalTokens}
          />
        </button>
      </div>

      {items.map(({ node, treeLines, isLast }) => {
        const { obs } = node
        const hasChildren = node.children.length > 0
        const isOpen = !collapsed.has(obs.id)
        const selected = selectedId === obs.id
        const tokens = (obs.promptTokens ?? 0) + (obs.completionTokens ?? 0)
        return (
          <div
            key={obs.id}
            role="treeitem"
            aria-level={node.depth + 2}
            aria-selected={selected}
            aria-expanded={hasChildren ? isOpen : undefined}
            className={`flex items-stretch pl-3 ${selected ? "bg-surface-subtle" : "hover:bg-surface-alt"}`}
          >
            {treeLines.map((continues, level) => (
              <span
                // biome-ignore lint/suspicious/noArrayIndexKey: fixed-width gutters
                key={level}
                aria-hidden="true"
                className={`w-4 shrink-0 ${continues ? "border-l border-border" : ""}`}
                style={{ marginLeft: level === 0 ? "0.75rem" : 0 }}
              />
            ))}
            <span
              aria-hidden="true"
              className="relative w-3 shrink-0"
              style={{ marginLeft: treeLines.length === 0 ? "0.75rem" : 0 }}
            >
              <span
                className={`absolute top-0 left-0 h-3.5 w-3 border-b border-l border-border ${isLast ? "" : ""}`}
              />
              {!isLast ? (
                <span className="absolute top-3.5 bottom-0 left-0 border-l border-border" />
              ) : null}
            </span>
            {hasChildren ? (
              <span className="otari-toolbar contents">
                <IconButton
                  label={isOpen ? "Collapse" : "Expand"}
                  size="sm"
                  className="shrink-0 md:min-h-6 md:min-w-6"
                  onPress={() => toggle(obs.id)}
                >
                  <FiChevronRight
                    aria-hidden="true"
                    className={`size-3.5 transition-transform duration-150 ease-smooth motion-reduce:transition-none ${isOpen ? "rotate-90" : ""}`}
                  />
                </IconButton>
              </span>
            ) : (
              <span className="w-11 shrink-0 md:w-6" />
            )}
            <span className="grid size-6 shrink-0 place-items-center pt-1">
              <TypeMark type={obs.type} />
            </span>
            <button
              type="button"
              className={ROW}
              aria-label={`${obs.type.charAt(0)}${obs.type.slice(1).toLowerCase()}, ${obs.name}, ${formatDuration(obs.endMs - obs.startMs)}`}
              onClick={() => onSelect(obs.id)}
            >
              <span className="flex min-w-0 items-center gap-2">
                <span className="truncate text-body">{obs.name}</span>
                <LevelChip level={obs.level} />
              </span>
              <Metrics
                durationMs={obs.endMs - obs.startMs}
                cost={(obs.inputCost ?? 0) + (obs.outputCost ?? 0)}
                tokens={tokens}
              />
            </button>
          </div>
        )
      })}
    </div>
  )
}
