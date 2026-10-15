import { useEffect, useMemo, useRef, useState } from "react"
import {
  FiAlignLeft,
  FiCheck,
  FiChevronRight,
  FiChevronsDown,
  FiChevronsUp,
  FiClock,
  FiCopy,
} from "react-icons/fi"

import { IconButton } from "@/design-system/actions/IconButton"
import { SearchField } from "@/design-system/forms/SearchField"
import { copyToClipboard } from "@/design-system/helpers/clipboard"
import { Tooltip } from "@/design-system/overlays/Tooltip"

import { formatDuration, formatRelativeTime } from "./format"
import { type Format, IOBlock } from "./IOBlock"
import { LevelChip } from "./LevelChip"
import { TypeMark } from "./observationTypes"
import { matchesSearch } from "./search"
import { buildTree, flattenTreeOrder } from "./tree"
import type { Trace } from "./types"

const INDENT_PX = 12

const gridClass = (showMs: boolean) =>
  `grid ${showMs ? "grid-cols-[1rem_minmax(0,1fr)_3rem_5rem_4.5rem]" : "grid-cols-[1rem_minmax(0,1fr)_3rem_3rem_4.5rem]"} items-center gap-2 px-3`

// A toggle keeps one name and says its state through `aria-pressed`; only the
// tooltip reads as the action.
function ToolbarButton({
  label,
  tooltip = label,
  pressed,
  onPress,
  children,
}: {
  label: string
  tooltip?: string
  pressed?: boolean
  onPress: () => void
  children: React.ReactNode
}) {
  return (
    <Tooltip content={tooltip}>
      <IconButton
        label={label}
        size="sm"
        aria-pressed={pressed}
        className={pressed ? "bg-surface-subtle" : ""}
        onPress={onPress}
      >
        {children}
      </IconButton>
    </Tooltip>
  )
}

export function LogView({ trace, format }: { trace: Trace; format: Format }) {
  const [query, setQuery] = useState("")
  const [indent, setIndent] = useState(false)
  const [showMs, setShowMs] = useState(false)
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(new Set())
  const [copied, setCopied] = useState(false)
  const copiedTimer = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  )
  useEffect(() => () => clearTimeout(copiedTimer.current), [])

  const items = useMemo(
    () => flattenTreeOrder(buildTree(trace.observations)),
    [trace],
  )
  const visible = useMemo(
    () => items.filter(({ node }) => matchesSearch(node.obs, query)),
    [items, query],
  )

  const allExpanded =
    visible.length > 0 && visible.every(({ node }) => expanded.has(node.obs.id))

  const toggleRow = (id: string) =>
    setExpanded((current) => {
      const next = new Set(current)
      if (!next.delete(id)) next.add(id)
      return next
    })

  const copy = async () => {
    // The shared helper falls back where the Clipboard API is missing, as on
    // the plain-HTTP origins a gateway is often served from.
    if (!(await copyToClipboard(JSON.stringify(trace.observations, null, 2))))
      return
    setCopied(true)
    clearTimeout(copiedTimer.current)
    copiedTimer.current = setTimeout(() => setCopied(false), 1000)
  }

  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <div className="otari-toolbar flex h-11 shrink-0 items-center gap-1 border-b border-border px-2">
        <SearchField
          label="Search observations"
          placeholder="Search observations..."
          value={query}
          onChange={setQuery}
          className="min-w-0 flex-1"
        />
        <ToolbarButton
          label="Indentation"
          tooltip={indent ? "Hide indentation" : "Show indentation"}
          pressed={indent}
          onPress={() => setIndent((v) => !v)}
        >
          <FiAlignLeft aria-hidden="true" className="size-4" />
        </ToolbarButton>
        <ToolbarButton
          label="Milliseconds"
          tooltip={showMs ? "Hide milliseconds" : "Show milliseconds"}
          pressed={showMs}
          onPress={() => setShowMs((v) => !v)}
        >
          <FiClock aria-hidden="true" className="size-4" />
        </ToolbarButton>
        <ToolbarButton
          label={allExpanded ? "Collapse all" : "Expand all"}
          onPress={() =>
            setExpanded(
              allExpanded
                ? new Set()
                : new Set(visible.map(({ node }) => node.obs.id)),
            )
          }
        >
          {allExpanded ? (
            <FiChevronsUp aria-hidden="true" className="size-4" />
          ) : (
            <FiChevronsDown aria-hidden="true" className="size-4" />
          )}
        </ToolbarButton>
        <ToolbarButton label="Copy as JSON" onPress={copy}>
          {copied ? (
            <FiCheck aria-hidden="true" className="size-4" />
          ) : (
            <FiCopy aria-hidden="true" className="size-4" />
          )}
        </ToolbarButton>
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        <div
          className={`${gridClass(showMs)} sticky top-0 z-10 min-h-6 border-b border-border bg-surface py-0.5 text-overline`}
        >
          <span />
          <span>Observation</span>
          <span className="text-right">Depth</span>
          <span className="text-right">Start</span>
          <span className="text-right">Duration</span>
        </div>
        {visible.map(({ node }) => {
          const { obs } = node
          const isOpen = expanded.has(obs.id)
          const childCount = node.children.length
          return (
            <div key={obs.id} className="border-b border-border">
              <button
                type="button"
                aria-expanded={isOpen}
                onClick={() => toggleRow(obs.id)}
                className={`otari-trace-ring ${gridClass(showMs)} min-h-11 w-full text-left hover:bg-surface-alt md:min-h-6`}
              >
                <FiChevronRight
                  aria-hidden="true"
                  className={`size-3.5 text-muted transition-transform duration-150 ease-smooth motion-reduce:transition-none ${isOpen ? "rotate-90" : ""}`}
                />
                <span
                  className="flex min-w-0 items-center gap-2"
                  style={{ paddingLeft: indent ? node.depth * INDENT_PX : 0 }}
                >
                  <TypeMark type={obs.type} />
                  <span className="truncate text-body">
                    {obs.name} ({obs.id.slice(0, 8)})
                  </span>
                  <LevelChip level={obs.level} />
                  {childCount > 0 ? (
                    <span className="shrink-0 text-caption text-subtle">
                      {childCount} {childCount === 1 ? "item" : "items"}
                    </span>
                  ) : null}
                </span>
                <span className="text-right text-mono-caption text-muted">
                  L{node.depth}
                </span>
                <span className="text-right text-mono-caption text-muted">
                  {formatRelativeTime(obs.startMs, showMs)}
                </span>
                <span className="text-right text-mono-caption text-muted">
                  {formatDuration(obs.endMs - obs.startMs)}
                </span>
              </button>
              {isOpen ? (
                <div className="flex flex-col gap-4 border-t border-border bg-surface-alt px-4 py-3">
                  {obs.input == null && obs.output == null ? (
                    <span className="text-caption">No input or output</span>
                  ) : null}
                  {obs.input != null ? (
                    <IOBlock label="Input" value={obs.input} format={format} />
                  ) : null}
                  {obs.output != null ? (
                    <IOBlock
                      label="Output"
                      value={obs.output}
                      format={format}
                    />
                  ) : null}
                </div>
              ) : null}
            </div>
          )
        })}
      </div>
    </div>
  )
}
