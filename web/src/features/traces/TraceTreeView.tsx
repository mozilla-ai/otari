import { type KeyboardEvent, useMemo } from "react"
import { FiChevronRight } from "react-icons/fi"

import { IconButton } from "@/design-system/actions/IconButton"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { formatCost, formatNumber } from "@/shared/helpers/format"

import { formatDuration } from "./format"
import { LevelChip } from "./LevelChip"
import { OBSERVATION_TYPES, TypeMark } from "./observationTypes"
import { moveRowFocus } from "./rowFocus"
import { matchesSearch } from "./search"
import { buildTree, flattenTreeOrder } from "./tree"
import { summarizeTrace, type Trace } from "./types"

function Metrics({
  durationMs,
  cost,
  tokens,
}: {
  durationMs: number
  cost: number
  tokens: number
}) {
  if (durationMs <= 0 && !cost && !tokens) return null
  return (
    <span className="flex items-center gap-x-3 truncate text-mono-caption whitespace-nowrap text-muted">
      {durationMs > 0 ? <span>{formatDuration(durationMs)}</span> : null}
      {cost ? (
        <span>{formatCost(cost)}</span>
      ) : tokens ? (
        <span>{formatNumber(tokens)} tokens</span>
      ) : null}
    </span>
  )
}

// The row's button spans everything but the chevron, so the gutter and the type
// mark open the span too.
const ROW =
  "otari-trace-ring flex min-h-11 min-w-0 flex-1 items-stretch text-left"
const TEXT = "flex min-w-0 flex-1 flex-col justify-center py-0.5 pr-1 pl-2"

// A level indents 16px: an 8px elbow plus half a 16px mark, so every connector
// runs through the center of its parent's mark and meets its own at mid-height.
const MARK =
  "relative flex w-4 shrink-0 items-center justify-center self-stretch"

// Holds a childless row's text where a chevron would be, so names align.
const CHEVRON_SLOT = <span aria-hidden="true" className="size-11 shrink-0" />

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
}: {
  trace: Trace
  selectedId: string | null
  onSelect: (observationId: string | null) => void
  collapsed: ReadonlySet<string>
  onToggle: (observationId: string) => void
  query: string
}) {
  const roots = useMemo(() => buildTree(trace.observations), [trace])
  const items = useMemo(
    () => flattenTreeOrder(roots, collapsed),
    [roots, collapsed],
  )
  const summary = summarizeTrace(trace)

  // A query turns the tree into a flat list of the matches: while searching, a
  // match's ancestors are in the way rather than context.
  if (query.trim()) {
    const matches = [...trace.observations]
      .sort((a, b) => a.startMs - b.startMs)
      .filter((obs) => matchesSearch(obs, query))
    if (matches.length === 0) {
      return <EmptyMessage>No spans match this search.</EmptyMessage>
    }
    const tabStop = matches.some((obs) => obs.id === selectedId)
      ? selectedId
      : matches[0].id
    return (
      <ul
        aria-label="Search results"
        data-trace-rows
        className="flex flex-col py-1"
      >
        {matches.map((obs) => {
          const selected = selectedId === obs.id
          return (
            <li key={obs.id}>
              <button
                type="button"
                data-trace-row
                tabIndex={obs.id === tabStop ? 0 : -1}
                aria-current={selected ? "true" : undefined}
                aria-label={`${OBSERVATION_TYPES[obs.type].label}, ${obs.name}, ${formatDuration(obs.endMs - obs.startMs)}`}
                onClick={() => onSelect(obs.id)}
                onKeyDown={moveRowFocus}
                className={`otari-trace-ring flex min-h-11 w-full items-center gap-1 px-3 text-left ${selected ? "bg-primary-subtle" : "hover:bg-surface-alt"}`}
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
                    cost={obs.cost ?? 0}
                    tokens={
                      (obs.promptTokens ?? 0) + (obs.completionTokens ?? 0)
                    }
                  />
                </span>
              </button>
            </li>
          )
        })}
      </ul>
    )
  }

  // The open row takes Tab; one hidden under a collapsed parent hands it to the
  // session row.
  const tabStop = items.some(({ node }) => node.obs.id === selectedId)
    ? selectedId
    : null

  return (
    <ul
      aria-label="Observations"
      data-trace-rows
      className="flex flex-col py-1"
    >
      <li
        className={`flex ${selectedId === null ? "bg-primary-subtle" : "hover:bg-surface-alt"}`}
      >
        <button
          type="button"
          data-trace-row
          tabIndex={tabStop === null ? 0 : -1}
          aria-current={selectedId === null ? "true" : undefined}
          aria-label={`Trace, ${trace.name}, ${formatDuration(summary.latencyMs)}`}
          onClick={() => onSelect(null)}
          onKeyDown={moveRowFocus}
          className={`${ROW} pl-3`}
        >
          <span className={MARK}>
            <TypeMark type="TRACE" />
            {items.length > 0 ? <DropLine /> : null}
          </span>
          <span className={TEXT}>
            <span className="truncate text-body">{trace.name}</span>
            <Metrics
              durationMs={summary.latencyMs}
              cost={summary.totalCost}
              tokens={summary.totalTokens}
            />
          </span>
        </button>
        {CHEVRON_SLOT}
      </li>

      {items.map(({ node, treeLines, isLast }) => {
        const { obs } = node
        const hasChildren = node.children.length > 0
        const isOpen = !collapsed.has(obs.id)
        const selected = selectedId === obs.id
        const tokens = (obs.promptTokens ?? 0) + (obs.completionTokens ?? 0)
        const onKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
          // Right opens a parent and Left closes it, so the chevrons need no
          // tab stops of their own.
          if (
            hasChildren &&
            (event.key === "ArrowRight" || event.key === "ArrowLeft") &&
            isOpen === (event.key === "ArrowLeft")
          ) {
            event.preventDefault()
            onToggle(obs.id)
            return
          }
          moveRowFocus(event)
        }
        return (
          <li
            key={obs.id}
            className={`flex ${selected ? "bg-primary-subtle" : "hover:bg-surface-alt"}`}
          >
            <button
              type="button"
              data-trace-row
              tabIndex={tabStop === obs.id ? 0 : -1}
              aria-current={selected ? "true" : undefined}
              aria-label={`${OBSERVATION_TYPES[obs.type].label}, ${obs.name}, ${formatDuration(obs.endMs - obs.startMs)}`}
              onClick={() => onSelect(obs.id)}
              onKeyDown={onKeyDown}
              className={`${ROW} pl-5`}
            >
              {treeLines.map((continues, level) => (
                <span
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
              <span className={TEXT}>
                <span className="flex min-w-0 items-center gap-2">
                  <span className="truncate text-body">{obs.name}</span>
                  <LevelChip level={obs.level} />
                </span>
                <Metrics
                  durationMs={obs.endMs - obs.startMs}
                  cost={obs.cost ?? 0}
                  tokens={tokens}
                />
              </span>
            </button>
            {hasChildren ? (
              <span className="otari-toolbar flex shrink-0 items-center">
                <IconButton
                  label={isOpen ? "Collapse" : "Expand"}
                  size="sm"
                  aria-expanded={isOpen}
                  excludeFromTabOrder
                  className="shrink-0"
                  onPress={() => onToggle(obs.id)}
                >
                  <FiChevronRight
                    aria-hidden="true"
                    className={`size-3.5 transition-transform duration-150 ease-smooth motion-reduce:transition-none ${isOpen ? "rotate-90" : ""}`}
                  />
                </IconButton>
              </span>
            ) : (
              CHEVRON_SLOT
            )}
          </li>
        )
      })}
    </ul>
  )
}
