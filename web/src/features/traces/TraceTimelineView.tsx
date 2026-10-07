import {
  useCallback,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react"
import { FiMaximize, FiMinus, FiPlus, FiType } from "react-icons/fi"

import { IconButton } from "@/design-system/actions/IconButton"
import { Tooltip } from "@/design-system/overlays/Tooltip"
import { formatCost } from "@/shared/helpers/format"

import { formatDuration } from "./format"
import { OBSERVATION_TYPES, TypeMark } from "./observationTypes"
import { buildTree, flattenTreeOrder } from "./tree"
import type { Observation, Trace } from "./types"

// Structure after Langfuse's TraceTimelineDense (MIT, Copyright (c) 2023-2026
// ClickHouse, Inc.): a gutter of type icons, a lane of bars on a shared time
// axis, labels beside the bars, dashed idle bands. This port is horizontal zoom
// only; the viewport lens, marquee and row zoom are not carried over.

const GUTTER_COLLAPSED = 28
const GUTTER_EXPANDED = 160
const BAR_HEIGHT = 18
const LABEL_GAP = 6
const MIN_ZOOM = 1
const MAX_ZOOM = 32
const ZOOM_STEP = 1.6
const IDLE_MIN_MS = 250
const IDLE_LABEL_MIN_PX = 56
const MIN_CLIPPED_LABEL_PX = 72

// Rough text widths at the label sizes (13px caption, 12px mono caption).
const NAME_PX = 7
const METRIC_PX = 7.2

function niceStep(totalMs: number, lanePx: number): number {
  const targetCount = Math.max(2, Math.floor(lanePx / 90))
  const raw = totalMs / targetCount
  const pow = 10 ** Math.floor(Math.log10(raw))
  for (const m of [1, 2, 5, 10]) if (m * pow >= raw) return m * pow
  return 10 * pow
}

function tickLabel(ms: number): string {
  if (ms === 0) return "0s"
  if (ms < 1000) return `${Math.round(ms)}ms`
  if (ms < 60_000) return `${+(ms / 1000).toFixed(1)}s`
  const s = Math.round(ms / 1000)
  return `${Math.floor(s / 60)}m${s % 60 ? ` ${s % 60}s` : ""}`
}

// Stretches of the trace where no leaf observation is running.
function idleGaps(observations: Observation[], totalMs: number) {
  const parents = new Set(observations.map((o) => o.parentObservationId))
  const leaves = observations
    .filter((o) => !parents.has(o.id) && o.endMs > o.startMs)
    .sort((a, b) => a.startMs - b.startMs)
  const gaps: { start: number; end: number }[] = []
  let cursor = 0
  for (const leaf of leaves) {
    if (leaf.startMs - cursor >= IDLE_MIN_MS) {
      gaps.push({ start: cursor, end: leaf.startMs })
    }
    cursor = Math.max(cursor, leaf.endMs)
  }
  if (totalMs - cursor >= IDLE_MIN_MS)
    gaps.push({ start: cursor, end: totalMs })
  return gaps
}

function ToolbarButton({
  label,
  isDisabled,
  onPress,
  children,
}: {
  label: string
  isDisabled?: boolean
  onPress: () => void
  children: ReactNode
}) {
  return (
    <Tooltip content={label}>
      <IconButton
        label={label}
        size="sm"
        isDisabled={isDisabled}
        className="md:min-h-7 md:min-w-7"
        onPress={onPress}
      >
        {children}
      </IconButton>
    </Tooltip>
  )
}

export function TraceTimelineView({
  trace,
  selectedId,
  onSelect,
}: {
  trace: Trace
  selectedId: string | null
  onSelect: (observationId: string | null) => void
}) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const [viewWidth, setViewWidth] = useState(0)
  const [zoom, setZoom] = useState(1)
  const [showLabels, setShowLabels] = useState(false)
  const centerRatio = useRef<number | null>(null)

  const measure = useCallback((element: HTMLDivElement | null) => {
    scrollRef.current = element
    if (!element) return
    setViewWidth(element.clientWidth)
    const observer = new ResizeObserver(() => setViewWidth(element.clientWidth))
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  const items = useMemo(
    () => flattenTreeOrder(buildTree(trace.observations)),
    [trace],
  )
  const totalMs = Math.max(1, ...trace.observations.map((o) => o.endMs))
  const gutter = showLabels ? GUTTER_EXPANDED : GUTTER_COLLAPSED
  const baseLane = Math.max(viewWidth - gutter, 120)
  const lanePx = baseLane * zoom
  const gaps = useMemo(
    () => idleGaps(trace.observations, totalMs),
    [trace, totalMs],
  )
  const step = niceStep(totalMs, lanePx)
  const ticks: number[] = []
  for (let t = 0; t <= totalMs; t += step) ticks.push(t)

  const x = (ms: number) => (ms / totalMs) * lanePx

  const zoomBy = (factor: number) => {
    const el = scrollRef.current
    if (el) {
      centerRatio.current =
        (el.scrollLeft + (el.clientWidth - gutter) / 2) / lanePx
    }
    setZoom((z) => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z * factor)))
  }

  // Keep what was in the middle of the view in the middle after a zoom.
  useLayoutEffect(() => {
    const el = scrollRef.current
    if (!el || centerRatio.current === null) return
    el.scrollLeft = Math.max(
      0,
      centerRatio.current * lanePx - (el.clientWidth - gutter) / 2,
    )
    centerRatio.current = null
  }, [lanePx, gutter])

  const fitted = zoom === MIN_ZOOM

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="otari-toolbar flex shrink-0 items-center gap-1 border-b border-border px-2 py-1">
        <ToolbarButton
          label="Zoom out"
          isDisabled={fitted}
          onPress={() => zoomBy(1 / ZOOM_STEP)}
        >
          <FiMinus aria-hidden="true" className="size-4" />
        </ToolbarButton>
        <ToolbarButton
          label="Zoom in"
          isDisabled={zoom >= MAX_ZOOM}
          onPress={() => zoomBy(ZOOM_STEP)}
        >
          <FiPlus aria-hidden="true" className="size-4" />
        </ToolbarButton>
        <ToolbarButton
          label="Show labels"
          onPress={() => setShowLabels((v) => !v)}
        >
          <FiType aria-hidden="true" className="size-4" />
        </ToolbarButton>
        <ToolbarButton
          label={fitted ? "Whole trace already fits" : "Fit whole trace"}
          isDisabled={fitted}
          onPress={() => {
            setZoom(MIN_ZOOM)
            if (scrollRef.current) scrollRef.current.scrollLeft = 0
          }}
        >
          <FiMaximize aria-hidden="true" className="size-4" />
        </ToolbarButton>
      </div>

      <div ref={measure} className="min-h-0 flex-1 overflow-auto">
        <div style={{ width: gutter + lanePx }} className="relative min-h-full">
          {/* Axis */}
          <div className="sticky top-0 z-20 flex h-6 border-b border-border bg-surface">
            <div
              style={{ width: gutter }}
              className="sticky left-0 z-10 shrink-0 bg-surface"
            />
            <div className="relative" style={{ width: lanePx }}>
              {ticks.map((t) => (
                <span
                  key={t}
                  style={{ left: x(t) }}
                  className="absolute top-0 h-full border-l border-border pl-1 text-mono-micro leading-6 text-muted"
                >
                  {tickLabel(t)}
                </span>
              ))}
              {gaps.map((gap) =>
                x(gap.end) - x(gap.start) >= IDLE_LABEL_MIN_PX ? (
                  <span
                    key={`g${gap.start}`}
                    style={{
                      left: x(gap.start),
                      width: x(gap.end) - x(gap.start),
                    }}
                    className="absolute bottom-0 truncate text-center text-mono-micro text-subtle"
                  >
                    {formatDuration(gap.end - gap.start)} idle
                  </span>
                ) : null,
              )}
            </div>
          </div>

          {/* Idle bands, behind the rows */}
          <div
            aria-hidden="true"
            style={{ left: gutter, width: lanePx }}
            className="pointer-events-none absolute top-6 bottom-0"
          >
            {gaps.map((gap) => (
              <span
                key={gap.start}
                style={{
                  left: x(gap.start),
                  width: Math.max(2, x(gap.end) - x(gap.start)),
                }}
                className="absolute inset-y-0 border-x border-dashed border-border-strong"
              />
            ))}
          </div>

          <div role="listbox" aria-label="Timeline">
            {items.map(({ node }) => {
              const { obs } = node
              const selected = selectedId === obs.id
              const vocab = OBSERVATION_TYPES[obs.type]
              const duration = obs.endMs - obs.startMs
              const left = x(obs.startMs)
              const isPoint = vocab.barClass === null
              const width = isPoint ? 0 : Math.max(2, x(duration))
              const cost = (obs.inputCost ?? 0) + (obs.outputCost ?? 0)
              const metric = [
                formatDuration(duration),
                cost ? formatCost(cost) : null,
              ]
                .filter(Boolean)
                .join(" ")
              const labelPx =
                obs.name.length * NAME_PX + metric.length * METRIC_PX + 8
              const spaceAfter = lanePx - (left + width + LABEL_GAP)
              const spaceBefore = left - LABEL_GAP
              const side: "after" | "before" | null =
                labelPx <= spaceAfter
                  ? "after"
                  : labelPx <= spaceBefore
                    ? "before"
                    : Math.max(spaceAfter, spaceBefore) >= MIN_CLIPPED_LABEL_PX
                      ? spaceAfter >= spaceBefore
                        ? "after"
                        : "before"
                      : null
              const room = side === "after" ? spaceAfter : spaceBefore
              const accessibleName = `${vocab.label}, ${obs.name}, ${formatDuration(duration) ?? ""}`
              return (
                // biome-ignore lint/a11y/useSemanticElements: a row of a listbox holding positioned children
                <div
                  key={obs.id}
                  role="option"
                  aria-selected={selected}
                  aria-label={accessibleName}
                  tabIndex={0}
                  onClick={() => onSelect(obs.id)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault()
                      onSelect(obs.id)
                    }
                  }}
                  className={`group flex h-11 cursor-pointer outline-none focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-foreground md:h-7 ${selected ? "bg-surface-subtle" : "hover:bg-surface-alt"}`}
                >
                  <div
                    style={{ width: gutter }}
                    className={`sticky left-0 z-10 flex shrink-0 items-center gap-1.5 overflow-hidden ${showLabels ? "pr-2" : "justify-center"} ${selected ? "bg-surface-subtle" : "bg-surface group-hover:bg-surface-alt"}`}
                  >
                    <span
                      style={{
                        marginLeft: showLabels ? 8 + node.depth * 8 : 0,
                      }}
                      className="grid shrink-0 place-items-center"
                    >
                      <TypeMark type={obs.type} className="size-3" />
                    </span>
                    {showLabels ? (
                      <span className="truncate text-caption">{obs.name}</span>
                    ) : null}
                  </div>
                  <div className="relative" style={{ width: lanePx }}>
                    {isPoint ? (
                      <span
                        style={{ left: left - 3 }}
                        className={`absolute top-1/2 size-1.5 -translate-y-1/2 rotate-45 bg-current ${vocab.inkClass} ${selected ? "outline outline-1 outline-offset-2 outline-foreground" : ""}`}
                      />
                    ) : (
                      <span
                        style={{ left, width, height: BAR_HEIGHT }}
                        className={`absolute top-1/2 -translate-y-1/2 ${vocab.barClass} ${selected ? "outline outline-1 outline-foreground" : ""}`}
                      />
                    )}
                    {side ? (
                      <span
                        style={{
                          maxWidth: room,
                          ...(side === "after"
                            ? { left: left + width + LABEL_GAP }
                            : { right: lanePx - left + LABEL_GAP }),
                        }}
                        className="absolute top-1/2 flex -translate-y-1/2 items-baseline gap-1.5 overflow-hidden whitespace-nowrap"
                      >
                        <span className="truncate text-caption text-foreground">
                          {obs.name}
                        </span>
                        <span className="shrink-0 text-mono-caption text-muted">
                          {metric}
                        </span>
                      </span>
                    ) : null}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
