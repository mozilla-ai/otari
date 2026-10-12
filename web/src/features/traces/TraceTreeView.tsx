import { useMemo } from "react"
import { FiChevronRight } from "react-icons/fi"

import { Button } from "@/design-system/actions/Button"
import { IconButton } from "@/design-system/actions/IconButton"
import { formatCost, formatNumber } from "@/shared/helpers/format"

import { formatDuration } from "./format"
import { LevelChip } from "./LevelChip"
import { TypeMark } from "./observationTypes"
import { matchesSearch } from "./search"
import { buildTree, flattenTreeOrder } from "./tree"
import { observationCost, summarizeTrace, type Trace } from "./types"

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

const ROW =
  "flex min-w-0 flex-1 flex-col py-0.5 pr-1 pl-2 text-left outline-none focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-foreground"

// A level indents 16px: an 8px elbow plus half a 16px mark, so every connector
// runs through the centre of its parent's mark and meets its own at mid-height.
const MARK =
  "relative flex w-4 shrink-0 items-center justify-center self-stretch"

function DropLine() {
  return (
    <span
      aria-hidden="true"
      className="absolute top-[calc(50%+0.625rem)] bottom-0 left-1/2 border-l border-border"
    />
  )
}

export function TraceTreeView({
  trace,
  selectedId,
  onSelect,
  collapsed,
  onToggle,
  query,
  onClearQuery,
}: {
  trace: Trace
  selectedId: string | null
  onSelect: (observationId: string | null) => void
  collapsed: ReadonlySet<string>
  onToggle: (observationId: string) => void
  query: string
  onClearQuery: () => void
}) {
  const roots = useMemo(() => buildTree(trace.observations), [trace])
  const items = useMemo(
    () => flattenTreeOrder(roots, collapsed),
    [roots, collapsed],
  )
  const summary = summarizeTrace(trace)

  const toggle = onToggle

  // As in Langfuse, a query turns the tree into a flat list of the matches.
  if (query.trim()) {
    const matches = [...trace.observations]
      .sort((a, b) => a.startMs - b.startMs)
      .filter((obs) => matchesSearch(obs, query))
    if (matches.length === 0) {
      return (
        <div className="flex flex-col items-center gap-2 p-6 text-center">
          <p className="text-body text-muted">No results found</p>
          <p className="text-caption text-subtle">
            Try searching by type, title, or id
          </p>
          <Button size="sm" onPress={onClearQuery}>
            Clear search
          </Button>
        </div>
      )
    }
    return (
      <div
        role="listbox"
        aria-label="Search results"
        className="flex flex-col py-1"
      >
        {matches.map((obs) => {
          const selected = selectedId === obs.id
          return (
            <button
              key={obs.id}
              type="button"
              role="option"
              aria-selected={selected}
              aria-label={`${obs.type.charAt(0)}${obs.type.slice(1).toLowerCase()}, ${obs.name}, ${formatDuration(obs.endMs - obs.startMs)}`}
              onClick={() => onSelect(obs.id)}
              className={`flex min-h-11 w-full items-center gap-1 px-3 text-left outline-none focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-foreground ${selected ? "bg-primary-subtle" : "hover:bg-surface-alt"}`}
            >
              <span className={MARK}>
                <TypeMark type={obs.type} />
              </span>
              <span className="flex min-w-0 flex-1 flex-col py-0.5 pl-2">
                <span className="flex min-w-0 items-center gap-2">
                  <span className="truncate text-body">{obs.name}</span>
                  <LevelChip level={obs.level} />
                </span>
                <Metrics
                  durationMs={obs.endMs - obs.startMs}
                  cost={observationCost(obs)}
                  tokens={(obs.promptTokens ?? 0) + (obs.completionTokens ?? 0)}
                />
              </span>
            </button>
          )
        })}
      </div>
    )
  }

  return (
    <div role="tree" aria-label="Observations" className="flex flex-col py-1">
      <div
        role="treeitem"
        aria-selected={selectedId === null}
        aria-level={1}
        className={`flex min-h-11 items-center pl-3 ${selectedId === null ? "bg-primary-subtle" : "hover:bg-surface-alt"}`}
      >
        <span className={MARK}>
          <TypeMark type="TRACE" />
          {items.length > 0 ? <DropLine /> : null}
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
        <span className="ml-auto size-11 shrink-0 md:size-6" />
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
            className={`flex min-h-11 items-stretch pl-5 ${selected ? "bg-primary-subtle" : "hover:bg-surface-alt"}`}
          >
            {treeLines.map((continues, level) => (
              <span
                // biome-ignore lint/suspicious/noArrayIndexKey: fixed-width gutters
                key={level}
                aria-hidden="true"
                className={`w-4 shrink-0 ${continues ? "border-l border-border" : ""}`}
              />
            ))}
            <span aria-hidden="true" className="relative w-2 shrink-0">
              <span className="absolute top-0 left-0 h-1/2 w-1.5 border-b border-l border-border" />
              {!isLast ? (
                <span className="absolute top-1/2 bottom-0 left-0 border-l border-border" />
              ) : null}
            </span>
            <span className={MARK}>
              <TypeMark type={obs.type} />
              {hasChildren && isOpen ? <DropLine /> : null}
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
                cost={observationCost(obs)}
                tokens={tokens}
              />
            </button>
            {hasChildren ? (
              <span className="otari-toolbar contents">
                <IconButton
                  label={isOpen ? "Collapse" : "Expand"}
                  size="sm"
                  className="ml-auto size-11 shrink-0 self-center md:size-6 md:min-h-6 md:min-w-6"
                  onPress={() => toggle(obs.id)}
                >
                  <FiChevronRight
                    aria-hidden="true"
                    className={`size-3.5 transition-transform duration-150 ease-smooth motion-reduce:transition-none ${isOpen ? "rotate-90" : ""}`}
                  />
                </IconButton>
              </span>
            ) : (
              <span className="ml-auto size-11 shrink-0 md:size-6" />
            )}
          </div>
        )
      })}
    </div>
  )
}
