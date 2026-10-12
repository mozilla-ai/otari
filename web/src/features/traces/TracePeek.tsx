import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  FiChevronDown,
  FiChevronUp,
  FiMaximize2,
  FiMinimize2,
  FiX,
} from "react-icons/fi"

import { CopyableValue } from "@/design-system/actions/CopyField"
import { IconButton } from "@/design-system/actions/IconButton"
import { Kbd } from "@/design-system/indicators/Kbd"
import { Segmented } from "@/design-system/navigation/Segmented"
import { Tab, TabRow } from "@/design-system/navigation/TabRow"
import { Tooltip } from "@/design-system/overlays/Tooltip"
import { formatCost, formatNumber } from "@/shared/helpers/format"

import {
  formatDuration,
  formatRelativeTime,
  formatTraceTimestamp,
} from "./format"
import { type Format, IOBlock } from "./IOBlock"
import { LogView } from "./LogView"
import { NavHeader, type NavView } from "./NavHeader"
import { TypeMark } from "./observationTypes"
import { TraceTimelineView } from "./TraceTimelineView"
import { TraceTreeView } from "./TraceTreeView"
import {
  type Observation,
  observationCost,
  summarizeTrace,
  type Trace,
} from "./types"

type PeekTab = "preview" | "attributes" | "log"

// Langfuse shows Attributes only while an observation is selected
// (TraceDetailView.tsx).
const ALL_TABS: { id: PeekTab; label: string }[] = [
  { id: "preview", label: "Preview" },
  { id: "attributes", label: "Attributes" },
  { id: "log", label: "Log View" },
]

const FORMAT_OPTIONS = [
  { value: "formatted", label: "Formatted" },
  { value: "raw", label: "Raw" },
]

const MIN_PX = 640 // 40rem
const MAX_FRACTION = 0.9
const DEFAULT_FRACTION = 0.5
const DEFAULT_CAP_PX = 1400 // 87.5rem
const EXIT_MS = 250

function defaultWidth(): number {
  return Math.max(
    MIN_PX,
    Math.min(window.innerWidth * DEFAULT_FRACTION, DEFAULT_CAP_PX),
  )
}

// A 4px bar in a 12px strip straddling an edge: transparent at rest, drawn on
// hover, focus and drag. Shared by the panel's outer edge and the inner divider.
function ResizeHandle({
  label,
  className,
  onDrag,
  onStep,
  valueNow,
  valueMin,
  valueMax,
}: {
  label: string
  className: string
  onDrag: (clientX: number) => void
  onStep: (direction: -1 | 1) => void
  valueNow?: number
  valueMin?: number
  valueMax?: number
}) {
  const [dragging, setDragging] = useState(false)
  return (
    // biome-ignore lint/a11y/useSemanticElements: an <hr> cannot take focus or a drag
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      aria-valuenow={valueNow}
      aria-valuemin={valueMin}
      aria-valuemax={valueMax}
      tabIndex={0}
      className={`group absolute inset-y-0 z-10 w-3 cursor-ew-resize touch-none outline-none ${className}`}
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture(event.pointerId)
        setDragging(true)
      }}
      onPointerMove={(event) => {
        if (dragging) onDrag(event.clientX)
      }}
      onPointerUp={() => setDragging(false)}
      onPointerCancel={() => setDragging(false)}
      onKeyDown={(event) => {
        if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
          event.preventDefault()
          onStep(event.key === "ArrowLeft" ? -1 : 1)
        }
      }}
    >
      <span
        className={`absolute inset-y-0 left-1/2 w-1 -translate-x-1/2 transition-colors group-hover:bg-control-border group-focus-visible:bg-control-border motion-reduce:transition-none ${
          dragging ? "bg-control-border" : "bg-transparent"
        }`}
      />
    </div>
  )
}

function Breakdown({ rows }: { rows: [string, string][] }) {
  return (
    <dl className="grid grid-cols-[auto_auto] gap-x-4 gap-y-0.5 text-mono-caption">
      {rows.map(([key, value]) => (
        <div key={key} className="contents">
          <dt>{key}</dt>
          <dd className="text-right">{value}</dd>
        </div>
      ))}
    </dl>
  )
}

const DOTTED =
  "underline decoration-dotted decoration-control-border underline-offset-4"

const STATUS_TITLE: Record<Observation["level"], string> = {
  ERROR: "Error",
  WARNING: "Warning",
  DEBUG: "Debug",
  DEFAULT: "Status",
}

const STATUS_TONE: Record<Observation["level"], string> = {
  ERROR: "bg-danger-subtle text-danger",
  WARNING: "bg-warning-subtle text-warning",
  DEBUG: "bg-surface-alt text-muted",
  DEFAULT: "bg-surface-alt text-muted",
}

function PreviewBody({
  input,
  output,
  statusMessage,
  level,
  format,
}: {
  input: unknown
  output: unknown
  statusMessage?: string
  level: Observation["level"]
  format: Format
}) {
  if (input == null && output == null && !statusMessage) {
    return <span className="text-caption text-subtle">No input or output</span>
  }
  return (
    <>
      {statusMessage ? (
        <section className="flex flex-col gap-1.5">
          <h3 className="text-overline text-muted">{STATUS_TITLE[level]}</h3>
          <p
            className={`px-3 py-2 whitespace-pre-wrap break-words ${STATUS_TONE[level]}`}
          >
            {statusMessage}
          </p>
        </section>
      ) : null}
      <IOBlock label="Input" value={input} format={format} />
      <IOBlock label="Output" value={output} format={format} />
    </>
  )
}

function Attributes({
  trace,
  observation,
}: {
  trace: Trace
  observation: Observation
}) {
  const rows: [string, string | undefined][] = [
    ["Type", observation.type],
    ["Level", observation.level],
    ["Status message", observation.statusMessage],
    ["Model", observation.model],
    ["Provider", observation.provider],
    [
      "Prompt tokens",
      observation.promptTokens != null
        ? formatNumber(observation.promptTokens)
        : undefined,
    ],
    [
      "Completion tokens",
      observation.completionTokens != null
        ? formatNumber(observation.completionTokens)
        : undefined,
    ],
    [
      "Total cost",
      observation.cost != null || observation.inputCost != null
        ? formatCost(observationCost(observation))
        : undefined,
    ],
    ["Start", formatRelativeTime(observation.startMs, true)],
    ["Duration", formatDuration(observation.endMs - observation.startMs)],
    ["ID", observation.id],
    ["Trace ID", trace.id],
    ...Object.entries(observation.metadata ?? {}),
  ]
  return (
    <dl className="grid grid-cols-[9rem_1fr] gap-x-4 gap-y-2">
      {rows
        .filter(([, value]) => value)
        .map(([key, value]) => (
          <div key={key} className="contents">
            <dt className="text-muted">{key}</dt>
            <dd className="break-words text-body">{value}</dd>
          </div>
        ))}
    </dl>
  )
}

export function TracePeek({
  trace,
  onClose,
  onPrevious,
  onNext,
}: {
  trace: Trace
  onClose: () => void
  onPrevious?: () => void
  onNext?: () => void
}) {
  const [tab, setTab] = useState<PeekTab>("preview")
  const [format, setFormat] = useState<Format>("formatted")
  const [width, setWidth] = useState(defaultWidth)
  const [expanded, setExpanded] = useState(false)
  const [phase, setPhase] = useState<"entering" | "open" | "leaving">(
    "entering",
  )
  const [navFraction, setNavFraction] = useState(0.4)
  const [navView, setNavView] = useState<NavView>("tree")
  const [navCollapsed, setNavCollapsed] = useState(false)
  const [navQuery, setNavQuery] = useState("")
  const [collapsedIds, setCollapsedIds] = useState<ReadonlySet<string>>(
    new Set(),
  )
  const bodyRef = useRef<HTMLDivElement>(null)
  const summary = summarizeTrace(trace)
  const [selection, setSelection] = useState<{
    traceId: string
    id: string | null
  } | null>(null)
  // Cleared whenever the peeked trace changes, so every trace opens on its root.
  if (selection && selection.traceId !== trace.id) setSelection(null)
  const selectedId = selection?.id ?? null
  const setSelectedId = (id: string | null) =>
    setSelection({ traceId: trace.id, id })
  const observation =
    trace.observations.find((o) => o.id === selectedId) ?? null
  const tabs = observation
    ? ALL_TABS
    : ALL_TABS.filter((t) => t.id !== "attributes")
  const parentIds = useMemo(
    () => [
      ...new Set(
        trace.observations
          .map((o) => o.parentObservationId)
          .filter((id): id is string => id !== null),
      ),
    ],
    [trace],
  )
  const everythingCollapsed =
    parentIds.length > 0 && parentIds.every((id) => collapsedIds.has(id))
  const toggleCollapsed = (id: string) =>
    setCollapsedIds((current) => {
      const next = new Set(current)
      if (!next.delete(id)) next.add(id)
      return next
    })
  const activeTab = tabs.some((t) => t.id === tab) ? tab : "preview"
  const shown = observation
    ? {
        name: observation.name,
        type: observation.type,
        timestamp: new Date(
          Date.parse(trace.timestamp) + observation.startMs,
        ).toISOString(),
        latencyMs: observation.endMs - observation.startMs,
        inputCost: observation.inputCost ?? 0,
        outputCost: observation.outputCost ?? 0,
        totalCost: observationCost(observation),
        promptTokens: observation.promptTokens ?? 0,
        completionTokens: observation.completionTokens ?? 0,
        totalTokens:
          (observation.promptTokens ?? 0) + (observation.completionTokens ?? 0),
        model: observation.model,
      }
    : {
        name: trace.name,
        type: "TRACE" as const,
        timestamp: trace.timestamp,
        latencyMs: summary.latencyMs,
        inputCost: summary.inputCost,
        outputCost: summary.outputCost,
        totalCost: summary.totalCost,
        promptTokens: summary.promptTokens,
        completionTokens: summary.completionTokens,
        totalTokens: summary.totalTokens,
        model: summary.models[0],
      }

  // setTimeout rather than requestAnimationFrame: a hidden tab never fires rAF.
  useEffect(() => {
    const timer = setTimeout(() => setPhase("open"), 20)
    return () => clearTimeout(timer)
  }, [])

  const close = useCallback(() => {
    setPhase("leaving")
    setTimeout(onClose, EXIT_MS)
  }, [onClose])

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null
      const typing =
        target?.tagName === "INPUT" ||
        target?.tagName === "TEXTAREA" ||
        target?.isContentEditable
      if (event.key === "Escape") close()
      if (typing || event.metaKey || event.ctrlKey || event.altKey) return
      if (event.key === "k") onPrevious?.()
      if (event.key === "j") onNext?.()
    }
    document.addEventListener("keydown", onKeyDown)
    return () => document.removeEventListener("keydown", onKeyDown)
  }, [close, onPrevious, onNext])

  const clampWidth = (px: number) =>
    Math.min(window.innerWidth * MAX_FRACTION, Math.max(MIN_PX, px))

  const shownWidth = expanded ? window.innerWidth * MAX_FRACTION : width
  const model = shown.model

  return (
    <aside
      aria-label={`Trace ${trace.name}`}
      style={{ "--peek-width": `${shownWidth}px` } as React.CSSProperties}
      className={`fixed top-14 right-0 bottom-0 z-30 flex w-full flex-col border-l border-border-strong bg-surface transition-transform duration-250 ease-out-fluid motion-reduce:transition-none md:w-(--peek-width) ${
        phase === "open" ? "translate-x-0" : "translate-x-full"
      }`}
    >
      <ResizeHandle
        label="Resize trace panel"
        className="-left-1 max-md:hidden"
        onDrag={(clientX) => {
          setExpanded(false)
          setWidth(clampWidth(window.innerWidth - clientX))
        }}
        onStep={(direction) => {
          setExpanded(false)
          setWidth(
            clampWidth(shownWidth - direction * window.innerWidth * 0.05),
          )
        }}
        valueNow={Math.round(shownWidth)}
        valueMin={MIN_PX}
        valueMax={Math.round(window.innerWidth * MAX_FRACTION)}
      />

      <div className="flex h-12 shrink-0 items-center justify-between gap-2 border-b border-border px-4">
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <span className="shrink-0 text-overline text-muted">Trace</span>
          <span className="flex min-w-0 [&>span]:min-w-0 [&>span>span:first-child]:min-w-0 [&>span>span:first-child]:truncate">
            <CopyableValue
              value={trace.id}
              label="Copy trace ID"
              className="text-mono-caption"
            />
          </span>
        </div>
        <div className="otari-toolbar flex shrink-0 items-center gap-1">
          <Tooltip
            content={
              <span className="flex items-center gap-2">
                Previous trace <Kbd keys={["K"]} />
              </span>
            }
          >
            <IconButton
              label="Previous trace"
              size="sm"
              onPress={onPrevious}
              isDisabled={!onPrevious}
            >
              <FiChevronUp aria-hidden="true" className="size-4" />
            </IconButton>
          </Tooltip>
          <Tooltip
            content={
              <span className="flex items-center gap-2">
                Next trace <Kbd keys={["J"]} />
              </span>
            }
          >
            <IconButton
              label="Next trace"
              size="sm"
              onPress={onNext}
              isDisabled={!onNext}
            >
              <FiChevronDown aria-hidden="true" className="size-4" />
            </IconButton>
          </Tooltip>
          <span className="max-md:hidden">
            <Tooltip content={expanded ? "Collapse" : "Expand"}>
              <IconButton
                label={expanded ? "Collapse" : "Expand"}
                size="sm"
                onPress={() => setExpanded((value) => !value)}
              >
                {expanded ? (
                  <FiMinimize2 aria-hidden="true" className="size-4" />
                ) : (
                  <FiMaximize2 aria-hidden="true" className="size-4" />
                )}
              </IconButton>
            </Tooltip>
          </span>
          <Tooltip content="Close">
            <IconButton label="Close" size="sm" onPress={close}>
              <FiX aria-hidden="true" className="size-4" />
            </IconButton>
          </Tooltip>
        </div>
      </div>

      <div className="flex shrink-0 flex-col gap-2 px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="grid size-6 shrink-0 place-items-center">
            <TypeMark type={shown.type} className="size-5" />
          </span>
          <h2 className="truncate text-title">{shown.name}</h2>
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-mono-caption text-muted">
          <span>{formatTraceTimestamp(shown.timestamp, true)}</span>
          {formatDuration(shown.latencyMs) ? (
            <span>{formatDuration(shown.latencyMs)}</span>
          ) : null}
          {shown.totalCost > 0 && shown.inputCost + shown.outputCost === 0 ? (
            <span>{formatCost(shown.totalCost)}</span>
          ) : shown.totalCost > 0 ? (
            <Tooltip
              content={
                <Breakdown
                  rows={[
                    ["Input", formatCost(shown.inputCost)],
                    ["Output", formatCost(shown.outputCost)],
                  ]}
                />
              }
            >
              {(props) => (
                <button type="button" {...props} className={DOTTED}>
                  {formatCost(shown.totalCost)}
                </button>
              )}
            </Tooltip>
          ) : null}
          {shown.totalTokens > 0 ? (
            <Tooltip
              content={
                <Breakdown
                  rows={[
                    ["Input", formatNumber(shown.promptTokens)],
                    ["Output", formatNumber(shown.completionTokens)],
                  ]}
                />
              }
            >
              {(props) => (
                <button type="button" {...props} className={DOTTED}>
                  {formatNumber(shown.totalTokens)} tokens
                </button>
              )}
            </Tooltip>
          ) : null}
          {model ? <span>{model}</span> : null}
        </div>
      </div>

      <div className="flex shrink-0 items-center justify-between gap-2 border-b border-border px-4 py-2">
        <TabRow>
          {tabs.map(({ id, label }) => (
            <Tab
              key={id}
              isActive={activeTab === id}
              onPress={() => setTab(id)}
            >
              {label}
            </Tab>
          ))}
        </TabRow>
        <Segmented
          size="sm"
          label="Format"
          options={FORMAT_OPTIONS}
          value={format}
          onChange={(next) => setFormat(next as Format)}
        />
      </div>

      <div ref={bodyRef} className="relative flex min-h-0 flex-1 overflow-auto">
        {activeTab === "log" ? (
          <LogView trace={trace} format={format} />
        ) : (
          <>
            <div
              style={{
                width: navCollapsed
                  ? "40px"
                  : navView === "timeline"
                    ? "max(280px, calc(100% - 360px))"
                    : `min(max(${navFraction * 100}%, 280px), calc(100% - 360px))`,
              }}
              className="relative flex shrink-0 flex-col border-r border-border max-md:hidden"
            >
              <NavHeader
                isCollapsed={navCollapsed}
                onToggleCollapsed={() => setNavCollapsed((value) => !value)}
                query={navQuery}
                onQueryChange={setNavQuery}
                everythingCollapsed={everythingCollapsed}
                onToggleAll={() =>
                  setCollapsedIds(
                    everythingCollapsed ? new Set() : new Set(parentIds),
                  )
                }
                view={navView}
                onViewChange={setNavView}
              />
              {navCollapsed ? null : navView === "tree" ? (
                <div className="min-h-0 flex-1 overflow-auto">
                  <TraceTreeView
                    trace={trace}
                    selectedId={observation?.id ?? null}
                    onSelect={setSelectedId}
                    collapsed={collapsedIds}
                    onToggle={toggleCollapsed}
                    query={navQuery}
                    onClearQuery={() => setNavQuery("")}
                  />
                </div>
              ) : (
                <TraceTimelineView
                  trace={trace}
                  selectedId={observation?.id ?? null}
                  onSelect={setSelectedId}
                  collapsed={collapsedIds}
                  query={navQuery}
                />
              )}
              <ResizeHandle
                label="Resize navigation pane"
                className={`-right-1.5 ${navCollapsed || navView === "timeline" ? "hidden" : ""}`}
                onDrag={(clientX) => {
                  const box = bodyRef.current?.getBoundingClientRect()
                  if (!box) return
                  setNavFraction(
                    Math.min(
                      0.7,
                      Math.max(0.2, (clientX - box.left) / box.width),
                    ),
                  )
                }}
                onStep={(direction) =>
                  setNavFraction((value) =>
                    Math.min(0.7, Math.max(0.2, value + direction * 0.04)),
                  )
                }
              />
            </div>
            <div className="flex min-w-0 flex-1 flex-col gap-4 overflow-auto p-4 md:min-w-[360px]">
              {activeTab === "attributes" && observation ? (
                <Attributes trace={trace} observation={observation} />
              ) : (
                <PreviewBody
                  input={observation ? observation.input : trace.input}
                  output={observation ? observation.output : trace.output}
                  statusMessage={observation?.statusMessage}
                  level={observation?.level ?? "DEFAULT"}
                  format={format}
                />
              )}
            </div>
          </>
        )}
      </div>
    </aside>
  )
}
