import {
  type ReactNode,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react"
import {
  FiChevronDown,
  FiChevronLeft,
  FiChevronUp,
  FiMaximize2,
  FiMinimize2,
  FiX,
} from "react-icons/fi"

import { Button } from "@/design-system/actions/Button"
import { CopyableValue } from "@/design-system/actions/CopyField"
import { IconButton } from "@/design-system/actions/IconButton"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Skeleton } from "@/design-system/feedback/Skeleton"
import { Kbd } from "@/design-system/indicators/Kbd"
import { Segmented } from "@/design-system/navigation/Segmented"
import { Tab, TabRow } from "@/design-system/navigation/TabRow"
import { Tooltip } from "@/design-system/overlays/Tooltip"
import { formatCost, formatNumber, formatPct } from "@/shared/helpers/format"
import { CostCard } from "./CostCard"
import { ExplainCard } from "./ExplainCard"
import {
  formatDuration,
  formatRelativeTime,
  formatTraceTimestamp,
} from "./format"
import { GateCard, VERDICT } from "./GateCard"
import { GatesView, stoppedBy } from "./GatesView"
import { GatewayPanel } from "./GatewayPanel"
import { type Format, IOBlock } from "./IOBlock"
import { LogView } from "./LogView"
import { NavHeader, type NavView } from "./NavHeader"
import { TypeMark } from "./observationTypes"
import type { SessionStats } from "./sessionStats"
import { TraceTimelineView } from "./TraceTimelineView"
import { TraceTreeView } from "./TraceTreeView"
import { type Observation, summarizeTrace, type Trace } from "./types"

type PeekTab = "preview" | "attributes" | "log" | "gates"

// Attributes belong to a span, so that tab shows only while one is selected.
const ALL_TABS: { id: PeekTab; label: string }[] = [
  { id: "preview", label: "Preview" },
  { id: "attributes", label: "Attributes" },
  { id: "log", label: "Log View" },
  { id: "gates", label: "Gates" },
]

const FORMAT_OPTIONS = [
  { value: "formatted", label: "Formatted" },
  { value: "raw", label: "Raw" },
]

const MIN_PX = 640 // 40rem
const MAX_FRACTION = 0.9
const DEFAULT_FRACTION = 0.5
const DEFAULT_CAP_PX = 1400 // 87.5rem
// One arrow press on the panel's edge, as a share of the window.
const WIDTH_STEP = 0.05
const EXIT_MS = 250

// The tree pane's share of the body, between these bounds, and one arrow
// press on its divider.
const NAV_DEFAULT_FRACTION = 0.4
const NAV_MIN_FRACTION = 0.2
const NAV_MAX_FRACTION = 0.7
const NAV_STEP = 0.04
// Whatever the share, the tree keeps NAV_MIN and the span beside it keeps
// DETAIL_MIN (`md:min-w-[22.5rem]` below); collapsed, the tree is its toggle.
const NAV_MIN = "17.5rem"
const DETAIL_MIN = "22.5rem"
const NAV_COLLAPSED = "2.5rem"

function defaultWidth(): number {
  return Math.max(
    MIN_PX,
    Math.min(window.innerWidth * DEFAULT_FRACTION, DEFAULT_CAP_PX),
  )
}

// A 4px bar in a 12px strip straddling an edge: transparent at rest, drawn on
// hover, focus and drag. Shared by the panel's outer edge and the inner divider.
// A drag writes at most once a frame, from the last pointer position.
function ResizeHandle({
  label,
  className,
  onDragStart,
  onDrag,
  onStep,
  valueNow,
  valueMin,
  valueMax,
}: {
  label: string
  className: string
  onDragStart?: () => void
  onDrag: (clientX: number) => void
  onStep: (direction: -1 | 1) => void
  valueNow: number
  valueMin: number
  valueMax: number
}) {
  const [dragging, setDragging] = useState(false)
  const frame = useRef(0)
  const latestX = useRef(0)
  useEffect(() => () => cancelAnimationFrame(frame.current), [])
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
      className={`otari-trace-ring group absolute inset-y-0 z-10 w-3 cursor-ew-resize touch-none ${className}`}
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture(event.pointerId)
        onDragStart?.()
        setDragging(true)
      }}
      onPointerMove={(event) => {
        if (!dragging) return
        latestX.current = event.clientX
        if (frame.current) return
        frame.current = requestAnimationFrame(() => {
          frame.current = 0
          onDrag(latestX.current)
        })
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

function Breakdown({ rows }: { rows: [string, string | undefined][] }) {
  return (
    <dl className="grid grid-cols-[auto_auto] gap-x-4 gap-y-0.5 text-mono-caption">
      {rows
        .filter(([, value]) => value)
        .map(([key, value]) => (
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
  DEFAULT: "Status",
}

const STATUS_TONE: Record<Observation["level"], string> = {
  ERROR: "bg-danger-subtle text-danger",
  WARNING: "bg-warning-subtle text-warning",
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
          <h3 className="text-overline">{STATUS_TITLE[level]}</h3>
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
        ? formatNumber(observation.promptTokens) +
          (observation.cachedTokens
            ? ` (${formatNumber(observation.cachedTokens)} cached)`
            : "")
        : undefined,
    ],
    [
      "Cache write tokens",
      observation.cacheWriteTokens
        ? formatNumber(observation.cacheWriteTokens)
        : undefined,
    ],
    [
      "Completion tokens",
      observation.completionTokens != null
        ? formatNumber(observation.completionTokens)
        : undefined,
    ],
    [
      "Reasoning tokens",
      observation.reasoningTokens
        ? formatNumber(observation.reasoningTokens)
        : undefined,
    ],
    [
      "Total cost",
      observation.cost != null ? formatCost(observation.cost) : undefined,
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

// The session's tool calls and gate runs, each with its breakdown on hover.
function SessionCounts({ stats }: { stats: SessionStats }) {
  const calls = stats.toolCalls
  const runs = stats.gates.runs
  return (
    <>
      {calls > 0 ? (
        <Tooltip
          content={
            <Breakdown
              rows={[
                ...stats.tools.map((tally): [string, string] => [
                  tally.tool,
                  formatNumber(tally.calls),
                ]),
                ...(stats.inSubagents > 0
                  ? [
                      ["In subagents", formatNumber(stats.inSubagents)] as [
                        string,
                        string,
                      ],
                    ]
                  : []),
              ]}
            />
          }
        >
          {(props) => (
            <button type="button" {...props} className={DOTTED}>
              {formatNumber(calls)} tool call{calls === 1 ? "" : "s"}
            </button>
          )}
        </Tooltip>
      ) : null}
      {runs > 0 ? (
        <Tooltip
          content={
            <Breakdown
              rows={Object.entries(stats.gates.verdicts).map(
                ([verdict, count]): [string, string] => [
                  VERDICT[verdict as keyof typeof VERDICT].word,
                  formatNumber(count),
                ],
              )}
            />
          }
        >
          {(props) => (
            <button type="button" {...props} className={DOTTED}>
              {formatNumber(runs)} gate{runs === 1 ? "" : "s"} run
            </button>
          )}
        </Tooltip>
      ) : null}
    </>
  )
}

export function TracePeek({
  trace,
  error,
  initialSpanId = "",
  onSpanChange,
  onClose,
  onPrevious,
  onNext,
  notice,
}: {
  trace: Trace
  /** Why the session's spans could not be read, when they could not. */
  error?: unknown
  /** The span to open on, as a link names it. Read once, when the panel opens. */
  initialSpanId?: string
  /** Reports each span picked in the panel ("" for the session), so the URL can follow it. */
  onSpanChange?: (spanId: string) => void
  onClose: () => void
  onPrevious?: () => void
  onNext?: () => void
  /** Said above the session and every span in it: who may read what it was about. */
  notice?: ReactNode
}) {
  const [tab, setTab] = useState<PeekTab>("preview")
  const [format, setFormat] = useState<Format>("formatted")
  const [width, setWidth] = useState(defaultWidth)
  const [expanded, setExpanded] = useState(false)
  const [entered, setEntered] = useState(false)
  // The session being closed, so a row opened during the exit is not closed
  // with it.
  const [leavingId, setLeavingId] = useState<string | null>(null)
  const [navFraction, setNavFraction] = useState(NAV_DEFAULT_FRACTION)
  const [navView, setNavView] = useState<NavView>("tree")
  const [navCollapsed, setNavCollapsed] = useState(false)
  const [navQuery, setNavQuery] = useState("")
  // The rows a person opened or closed, against the default below, so a tool
  // call a live session adds later starts closed like the rest.
  const [toggledIds, setToggledIds] = useState<ReadonlySet<string>>(new Set())
  const bodyRef = useRef<HTMLDivElement>(null)
  const bodyBox = useRef<DOMRect | null>(null)
  const headingRef = useRef<HTMLHeadingElement>(null)
  const backRef = useRef<HTMLButtonElement>(null)
  const navRef = useRef<HTMLDivElement>(null)
  const exitTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const shownTraceId = useRef(trace.id)
  useLayoutEffect(() => {
    shownTraceId.current = trace.id
  })
  const summary = summarizeTrace(trace)
  const loaded = trace.gateway !== undefined
  const leaving = leavingId === trace.id
  // Below md one pane shows at a time: picking a row (the session's included)
  // shows what it opened, and Back returns to the tree.
  const [selection, setSelection] = useState<{
    traceId: string
    id: string | null
    isDetailShown: boolean
  } | null>(() =>
    initialSpanId
      ? { traceId: trace.id, id: initialSpanId, isDetailShown: true }
      : null,
  )
  // Cleared whenever the peeked trace changes, so every other trace opens on its root.
  if (selection && selection.traceId !== trace.id) setSelection(null)
  const selectedId = selection?.id ?? null
  const isDetailShown = selection?.isDetailShown ?? false
  const setSelectedId = (id: string | null) => {
    setSelection({ traceId: trace.id, id, isDetailShown: true })
    onSpanChange?.(id ?? "")
  }
  const showTree = () =>
    setSelection({ traceId: trace.id, id: selectedId, isDetailShown: false })
  const observation =
    trace.observations.find((o) => o.id === selectedId) ?? null
  const gates = trace.gateway?.gates ?? []
  const blockedChecks = gates.reduce((sum, gate) => sum + stoppedBy(gate), 0)
  // Attributes only for a picked span, Gates only for a session a hook reported to.
  const tabs = ALL_TABS.filter(
    (t) =>
      (t.id !== "attributes" || observation) &&
      (t.id !== "gates" || gates.length > 0),
  )
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
  // A tool call and a group of gates start closed: their row says enough until
  // someone asks for more.
  const defaultCollapsed = useMemo(() => {
    const byId = new Map(trace.observations.map((o) => [o.id, o]))
    // Only a row whose children are all details of it, so a tool call another
    // agent made with work of its own under it never hides that work.
    const onlyDetails = new Map<string, boolean>()
    for (const o of trace.observations) {
      if (o.parentObservationId === null) continue
      onlyDetails.set(
        o.parentObservationId,
        (onlyDetails.get(o.parentObservationId) ?? true) && o.detail === true,
      )
    }
    return new Set(
      parentIds.filter((id) => {
        const row = byId.get(id)
        return (
          (row?.type === "TOOL" || row?.detail === true) &&
          onlyDetails.get(id) === true
        )
      }),
    )
  }, [trace, parentIds])
  const collapsedIds = useMemo(
    () =>
      new Set(
        parentIds.filter(
          (id) => defaultCollapsed.has(id) !== toggledIds.has(id),
        ),
      ),
    [parentIds, defaultCollapsed, toggledIds],
  )
  const setCollapsedIds = (collapsed: ReadonlySet<string>) =>
    setToggledIds(
      new Set(
        parentIds.filter(
          (id) => defaultCollapsed.has(id) !== collapsed.has(id),
        ),
      ),
    )
  const everythingCollapsed =
    parentIds.length > 0 && parentIds.every((id) => collapsedIds.has(id))
  // Opens a span from outside the tree (a gate, a gateway decision): selects it,
  // opens every collapsed parent above it and clears a search that hides it, then
  // scrolls it into view once that has rendered.
  const isRevealPending = useRef(false)
  const reveal = (spanId: string) => {
    const byId = new Map(trace.observations.map((o) => [o.id, o]))
    const ancestors = new Set<string>()
    let parent = byId.get(spanId)?.parentObservationId ?? null
    while (parent && !ancestors.has(parent)) {
      ancestors.add(parent)
      parent = byId.get(parent)?.parentObservationId ?? null
    }
    setCollapsedIds(
      new Set([...collapsedIds].filter((id) => !ancestors.has(id))),
    )
    setNavQuery("")
    setNavCollapsed(false)
    setTab("preview")
    setSelectedId(spanId)
    isRevealPending.current = true
  }
  useEffect(() => {
    if (!isRevealPending.current) return
    isRevealPending.current = false
    navRef.current
      ?.querySelector('[aria-current="true"]')
      ?.scrollIntoView({ block: "center" })
  })
  const toggleCollapsed = (id: string) =>
    setToggledIds((current) => {
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
        totalCost: observation.cost ?? 0,
        promptTokens: observation.promptTokens ?? 0,
        completionTokens: observation.completionTokens ?? 0,
        cachedTokens: observation.cachedTokens ?? 0,
        cacheWriteTokens: observation.cacheWriteTokens ?? 0,
        totalTokens:
          (observation.promptTokens ?? 0) + (observation.completionTokens ?? 0),
        model: observation.model,
      }
    : {
        name: trace.name,
        type: "TRACE" as const,
        timestamp: trace.timestamp,
        latencyMs: summary.latencyMs,
        totalCost: summary.totalCost,
        promptTokens: summary.promptTokens,
        completionTokens: summary.completionTokens,
        cachedTokens: summary.cachedTokens,
        cacheWriteTokens: summary.cacheWriteTokens,
        totalTokens: summary.totalTokens,
        model: summary.models[0],
      }

  // setTimeout rather than requestAnimationFrame: a hidden tab never fires rAF.
  useEffect(() => {
    const timer = setTimeout(() => setEntered(true), 20)
    return () => clearTimeout(timer)
  }, [])

  // Focus moves into the panel as it opens and back to whatever opened it as
  // it closes, so a keyboard user is neither left behind on the list nor
  // dropped to the top of the document.
  useEffect(() => {
    const opener =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null
    headingRef.current?.focus()
    return () => {
      clearTimeout(exitTimer.current)
      if (opener?.isConnected) opener.focus()
    }
  }, [])

  // A panel opened on a span opens the rows above it and brings it into view
  // once the spans arrive, as picking it from the panel would.
  const isSpanUnseen = useRef(initialSpanId !== "")
  useEffect(() => {
    if (!loaded || !isSpanUnseen.current || !initialSpanId) return
    isSpanUnseen.current = false
    reveal(initialSpanId)
  })

  // The pane swap below md moves focus with it, as ListDetail does. Only on a
  // change, and a no-op from md up, where the Back control is not displayed.
  const wasDetailShown = useRef(isDetailShown)
  useEffect(() => {
    if (wasDetailShown.current === isDetailShown) return
    wasDetailShown.current = isDetailShown
    if (isDetailShown) backRef.current?.focus()
    else
      navRef.current
        ?.querySelector<HTMLElement>('[aria-current="true"]')
        ?.focus()
  }, [isDetailShown])

  const close = () => {
    if (leaving) return
    const closing = trace.id
    setLeavingId(closing)
    clearTimeout(exitTimer.current)
    exitTimer.current = setTimeout(() => {
      if (shownTraceId.current === closing) onClose()
    }, EXIT_MS)
  }

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      // A field owns its own keys, Escape included: in the span search it
      // clears the query rather than closing the panel.
      if (event.defaultPrevented) return
      if (
        event.target instanceof Element &&
        event.target.closest("input, textarea, [contenteditable='true']")
      )
        return
      if (event.metaKey || event.ctrlKey || event.altKey) return
      if (event.key === "Escape") close()
      else if (event.key === "k") onPrevious?.()
      else if (event.key === "j") onNext?.()
    }
    document.addEventListener("keydown", onKeyDown)
    return () => document.removeEventListener("keydown", onKeyDown)
  })

  const clampWidth = (px: number) =>
    Math.min(window.innerWidth * MAX_FRACTION, Math.max(MIN_PX, px))
  const clampNav = (fraction: number) =>
    Math.min(NAV_MAX_FRACTION, Math.max(NAV_MIN_FRACTION, fraction))

  const shownWidth = expanded ? window.innerWidth * MAX_FRACTION : width
  const model = shown.model
  const navWidth = navCollapsed
    ? NAV_COLLAPSED
    : navView === "timeline"
      ? `max(${NAV_MIN}, calc(100% - ${DETAIL_MIN}))`
      : `min(max(${navFraction * 100}%, ${NAV_MIN}), calc(100% - ${DETAIL_MIN}))`

  return (
    <aside
      aria-label={`Trace ${trace.name}`}
      style={{ "--peek-width": `${shownWidth}px` } as React.CSSProperties}
      className={`fixed top-14 right-0 bottom-0 z-30 flex w-full flex-col border-l border-border-strong bg-surface transition-transform duration-250 ease-out-fluid motion-reduce:transition-none md:w-(--peek-width) ${
        entered && !leaving ? "translate-x-0" : "translate-x-full"
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
            clampWidth(shownWidth - direction * window.innerWidth * WIDTH_STEP),
          )
        }}
        valueNow={Math.round(shownWidth)}
        valueMin={MIN_PX}
        valueMax={Math.round(window.innerWidth * MAX_FRACTION)}
      />

      <div className="flex h-12 shrink-0 items-center justify-between gap-2 border-b border-border px-4">
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <span className="shrink-0 text-overline">Trace</span>
          <span className="flex min-w-0">
            <CopyableValue
              value={trace.id}
              label="trace ID"
              className="text-mono-caption"
              isTruncated
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
          {/* Focused only by the panel opening, as ListDetail's landing column
              is, so it draws no ring: it is not a control. */}
          <h2
            ref={headingRef}
            tabIndex={-1}
            className="truncate text-title outline-none"
          >
            {shown.name}
          </h2>
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-mono-caption text-muted">
          <span>{formatTraceTimestamp(shown.timestamp, true)}</span>
          {/* Until the spans arrive there is nothing to measure. */}
          {loaded ? <span>{formatDuration(shown.latencyMs)}</span> : null}
          {shown.totalCost > 0 ? (
            <span>{formatCost(shown.totalCost)}</span>
          ) : null}
          {shown.totalTokens > 0 ? (
            <Tooltip
              content={
                <Breakdown
                  rows={[
                    ["Input", formatNumber(shown.promptTokens)],
                    [
                      "Cached",
                      shown.promptTokens && shown.cachedTokens
                        ? formatPct(shown.cachedTokens / shown.promptTokens)
                        : undefined,
                    ],
                    [
                      "Written to the cache",
                      shown.cacheWriteTokens
                        ? formatNumber(shown.cacheWriteTokens)
                        : undefined,
                    ],
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
          {!observation && trace.gateway ? (
            <SessionCounts stats={trace.gateway.stats} />
          ) : null}
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
              {id === "gates" && blockedChecks > 0 ? (
                <>
                  {" "}
                  <span className="text-warning">{`· ${blockedChecks} blocked`}</span>
                </>
              ) : null}
            </Tab>
          ))}
        </TabRow>
        {observation || activeTab === "log" ? (
          <Segmented
            size="sm"
            label="Format"
            options={FORMAT_OPTIONS}
            value={format}
            onChange={(next) => setFormat(next as Format)}
          />
        ) : null}
      </div>

      <div ref={bodyRef} className="relative flex min-h-0 flex-1 overflow-auto">
        {error ? (
          <div className="flex-1 p-4">
            <ErrorBanner error={error} />
          </div>
        ) : !loaded ? (
          <div className="flex flex-1 flex-col gap-3 p-4">
            <Skeleton className="h-5 w-2/3" ariaLabel="Loading the spans" />
            <Skeleton className="h-5 w-1/2" />
            <Skeleton className="h-5 w-3/5" />
          </div>
        ) : activeTab === "log" ? (
          <LogView trace={trace} format={format} />
        ) : activeTab === "gates" ? (
          <GatesView gates={gates} onSelect={reveal} />
        ) : (
          <>
            <div
              ref={navRef}
              style={{ "--nav-width": navWidth } as React.CSSProperties}
              className={`relative shrink-0 flex-col border-border md:flex md:w-(--nav-width) md:border-r ${
                isDetailShown ? "hidden" : "flex w-full"
              }`}
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
                className={`-right-1.5 max-md:hidden ${navCollapsed || navView === "timeline" ? "hidden" : ""}`}
                onDragStart={() => {
                  bodyBox.current =
                    bodyRef.current?.getBoundingClientRect() ?? null
                }}
                onDrag={(clientX) => {
                  const box = bodyBox.current
                  if (box)
                    setNavFraction(clampNav((clientX - box.left) / box.width))
                }}
                onStep={(direction) =>
                  setNavFraction((value) =>
                    clampNav(value + direction * NAV_STEP),
                  )
                }
                valueNow={Math.round(navFraction * 100)}
                valueMin={NAV_MIN_FRACTION * 100}
                valueMax={NAV_MAX_FRACTION * 100}
              />
            </div>
            <div
              className={`min-w-0 flex-1 flex-col gap-4 overflow-auto p-4 md:flex md:min-w-[22.5rem] ${
                isDetailShown ? "flex" : "hidden"
              }`}
            >
              <div className="-mx-4 -mt-4 border-b border-border px-2 py-1 md:hidden">
                <Button ref={backRef} className="min-h-11" onPress={showTree}>
                  <FiChevronLeft aria-hidden="true" className="size-4" />
                  Back to the spans
                </Button>
              </div>
              {notice}
              {activeTab === "attributes" && observation ? (
                <Attributes trace={trace} observation={observation} />
              ) : !observation && trace.gateway ? (
                <GatewayPanel
                  gateway={trace.gateway}
                  startedAt={trace.timestamp}
                  onSelect={reveal}
                />
              ) : observation?.gate ? (
                <GateCard gate={observation.gate} />
              ) : observation ? (
                <>
                  {observation.explain ? (
                    <ExplainCard explanation={observation.explain} />
                  ) : null}
                  {observation.estimate ? (
                    <CostCard cost={observation.estimate} />
                  ) : null}
                  {observation.explain &&
                  observation.input == null &&
                  observation.output == null &&
                  !observation.statusMessage ? null : (
                    <PreviewBody
                      input={observation.input}
                      output={observation.output}
                      statusMessage={observation.statusMessage}
                      level={observation.level}
                      format={format}
                    />
                  )}
                </>
              ) : null}
            </div>
          </>
        )}
      </div>
    </aside>
  )
}
