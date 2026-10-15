import {
  type ReactNode,
  useCallback,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react"
import { FiMaximize, FiMinus, FiPlus } from "react-icons/fi"

import { IconButton } from "@/design-system/actions/IconButton"
import { Tooltip } from "@/design-system/overlays/Tooltip"
import { formatCost, formatLatency } from "@/shared/helpers/format"

import { OBSERVATION_TYPES, type TypeKey, TypeMark } from "./observationTypes"
import { moveRowFocus } from "./rowFocus"
import { matchesSearch } from "./search"
import { buildTree, flattenTreeOrder, rolledUpCosts } from "./tree"
import type { Trace } from "./types"

// Structure after Langfuse's dense trace timeline (see LANGFUSE-LICENSE.md): a
// name gutter with connectors, a lane of bars on a shared time axis with
// gridlines, duration and cost on the bars, dashed idle bands.

const RAIL_WIDTH = 28
const GUTTER_MIN = 112
const GUTTER_MAX = 200
const GUTTER_FRACTION = 0.42
const LANE_MIN = 140
const INDENT = 14
const CONNECTOR_OFFSET = 6
const BAR_MAX_HEIGHT = 18
const LABEL_GAP = 6
const BAR_INSET = 4
const MIN_ZOOM = 1
const MAX_ZOOM = 32
const ZOOM_STEP = 1.6
const TICK_MIN_PX = 56
const IDLE_MIN_MS = 250
const IDLE_LABEL_MIN_PX = 56
const IDLE_MIN_PX = 14
// Fira Code is monospace: 0.6em, so 11px text is 6.6px a glyph.
const MICRO_GLYPH_PX = 6.6
const LABEL_ITEM_GAP = 8

function niceStep(totalMs: number, lanePx: number): number {
  const targetCount = Math.max(1, Math.floor(lanePx / TICK_MIN_PX))
  const raw = totalMs / targetCount
  const pow = 10 ** Math.floor(Math.log10(raw))
  for (const m of [1, 2, 5, 10]) if (m * pow >= raw) return m * pow
  return 10 * pow
}

// Stretches of the trace where no leaf observation is running.
function idleGaps(
  observations: Trace["observations"],
  totalMs: number,
): { start: number; end: number }[] {
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

function ZoomButton({
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
        isIconOnly
        isDisabled={isDisabled}
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
  collapsed,
  query,
}: {
  trace: Trace
  selectedId: string | null
  onSelect: (observationId: string | null) => void
  collapsed: ReadonlySet<string>
  query: string
}) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const [viewWidth, setViewWidth] = useState(0)
  const [zoom, setZoom] = useState(1)
  const centerRatio = useRef<number | null>(null)

  const measure = useCallback((element: HTMLDivElement | null) => {
    scrollRef.current = element
    if (!element) return
    setViewWidth(element.clientWidth)
    const observer = new ResizeObserver(() => setViewWidth(element.clientWidth))
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  const roots = useMemo(() => buildTree(trace.observations), [trace])
  const items = useMemo(
    () => flattenTreeOrder(roots, collapsed),
    [roots, collapsed],
  )
  const costs = useMemo(() => rolledUpCosts(roots), [roots])
  const totalMs = Math.max(1, ...trace.observations.map((o) => o.endMs))
  const totalCost = roots.reduce(
    (sum, root) => sum + (costs.get(root.obs.id) ?? 0),
    0,
  )

  // The gutter gives way to the icon rail when the lane would drop under its
  // minimum.
  const content = Math.max(viewWidth - 2, 0)
  const wanted = Math.min(
    Math.max(content * GUTTER_FRACTION, GUTTER_MIN),
    GUTTER_MAX,
  )
  const hasNames = content - wanted >= LANE_MIN
  const gutter = hasNames ? Math.round(wanted) : RAIL_WIDTH
  const baseLane = Math.max(viewWidth - gutter, 60)
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
  const hasQuery = query.trim().length > 0
  // The open row takes Tab; one hidden under a collapsed parent hands it to the
  // session row.
  const tabStop = items.some(({ node }) => node.obs.id === selectedId)
    ? selectedId
    : null

  const row = ({
    id,
    type,
    name,
    depth,
    treeLines,
    isLast,
    startMs,
    endMs,
    cost,
    dimmed,
  }: {
    id: string | null
    type: TypeKey
    name: string
    depth: number
    treeLines: boolean[]
    isLast: boolean
    startMs: number
    endMs: number
    cost: number
    dimmed: boolean
  }) => {
    const selected = selectedId === id
    const vocab = OBSERVATION_TYPES[type]
    const duration = endMs - startMs
    const left = x(startMs)
    const width = Math.max(2, x(duration))
    const durationText = formatLatency(duration)
    const costText = cost ? formatCost(cost) : ""
    const labelPx =
      (durationText.length + costText.length) * MICRO_GLYPH_PX +
      (costText ? LABEL_ITEM_GAP : 0)
    const spaceAfter = lanePx - (left + width + LABEL_GAP)
    const spaceBefore = left - LABEL_GAP
    const placement: "inside" | "after" | "before" | null =
      labelPx + LABEL_ITEM_GAP <= width
        ? "inside"
        : labelPx <= spaceAfter
          ? "after"
          : labelPx <= spaceBefore
            ? "before"
            : null
    const indent = depth * INDENT
    return (
      <li key={id ?? "session"}>
        <button
          type="button"
          data-trace-row
          tabIndex={tabStop === id ? 0 : -1}
          aria-current={selected ? "true" : undefined}
          aria-label={`${vocab.label}, ${name}, ${durationText}`}
          onClick={() => onSelect(id)}
          onKeyDown={moveRowFocus}
          className={`otari-trace-ring group relative flex h-11 w-full cursor-pointer text-left md:h-[1.625rem] ${dimmed ? "opacity-30" : ""}`}
        >
          {selected ? (
            <span
              aria-hidden="true"
              className="pointer-events-none absolute inset-0 z-0 bg-primary-subtle"
            />
          ) : (
            <span
              aria-hidden="true"
              className="pointer-events-none absolute inset-0 z-0 hidden group-hover:block group-hover:bg-surface-alt"
            />
          )}
          <span
            style={{ width: gutter }}
            className="sticky left-0 z-10 shrink-0 overflow-hidden bg-surface"
          >
            {selected ? (
              <span
                aria-hidden="true"
                className="absolute inset-0 bg-primary-subtle"
              />
            ) : null}
            {hasNames ? (
              <>
                {treeLines.map((continues, a) =>
                  a >= 1 && continues ? (
                    <span
                      key={a}
                      aria-hidden="true"
                      style={{
                        left: (a - 1) * INDENT + CONNECTOR_OFFSET,
                      }}
                      className="absolute inset-y-0 w-px bg-border-strong"
                    />
                  ) : null,
                )}
                {depth >= 1 ? (
                  <>
                    <span
                      aria-hidden="true"
                      style={{
                        left: (depth - 1) * INDENT + CONNECTOR_OFFSET,
                        width: INDENT - CONNECTOR_OFFSET,
                      }}
                      className="absolute top-0 h-1/2 border-b border-l border-border-strong"
                    />
                    {!isLast ? (
                      <span
                        aria-hidden="true"
                        style={{
                          left: (depth - 1) * INDENT + CONNECTOR_OFFSET,
                        }}
                        className="absolute top-1/2 bottom-0 w-px bg-border-strong"
                      />
                    ) : null}
                  </>
                ) : null}
                <span
                  style={{ paddingLeft: indent + 2 }}
                  className="relative flex h-full items-center gap-1 pr-1"
                >
                  <TypeMark type={type} className="size-3 shrink-0" />
                  <span
                    title={name}
                    className="truncate text-shell-secondary text-foreground"
                  >
                    {name}
                  </span>
                </span>
              </>
            ) : (
              <span className="grid h-full place-items-center">
                <TypeMark type={type} className="size-3" />
              </span>
            )}
          </span>
          <span className="relative block" style={{ width: lanePx }}>
            <span
              style={{
                left,
                width,
                height: `min(calc(100% - 1px), ${BAR_MAX_HEIGHT}px)`,
              }}
              className={`absolute top-1/2 -translate-y-1/2 ${vocab.barClass} ${selected ? "otari-trace-mark--selected" : ""}`}
            />
            {placement ? (
              <span
                style={
                  placement === "inside"
                    ? {
                        left: left + BAR_INSET,
                        maxWidth: width - BAR_INSET,
                      }
                    : placement === "after"
                      ? { left: left + width + LABEL_GAP }
                      : { right: lanePx - left + LABEL_GAP }
                }
                className={`absolute top-1/2 z-[1] flex -translate-y-1/2 items-center gap-2 overflow-hidden text-mono-micro whitespace-nowrap ${placement === "inside" ? "text-trace-bar-ink" : "text-muted"}`}
              >
                <span>{durationText}</span>
                {costText ? <span>{costText}</span> : null}
              </span>
            ) : null}
          </span>
        </button>
      </li>
    )
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="otari-toolbar flex h-11 shrink-0 items-center gap-1 border-b border-border px-1">
        <ZoomButton
          label="Zoom out"
          isDisabled={fitted}
          onPress={() => zoomBy(1 / ZOOM_STEP)}
        >
          <FiMinus aria-hidden="true" className="size-3" />
        </ZoomButton>
        <ZoomButton
          label="Zoom in"
          isDisabled={zoom >= MAX_ZOOM}
          onPress={() => zoomBy(ZOOM_STEP)}
        >
          <FiPlus aria-hidden="true" className="size-3" />
        </ZoomButton>
        <ZoomButton
          label={fitted ? "Whole trace already fits" : "Fit whole trace"}
          isDisabled={fitted}
          onPress={() => {
            setZoom(MIN_ZOOM)
            if (scrollRef.current) scrollRef.current.scrollLeft = 0
          }}
        >
          <FiMaximize aria-hidden="true" className="size-3" />
        </ZoomButton>
      </div>

      <div ref={measure} className="min-h-0 flex-1 overflow-auto">
        <div style={{ width: gutter + lanePx }} className="relative min-h-full">
          <div className="sticky top-0 z-20 flex h-[1.125rem] border-b border-border bg-surface">
            <div
              style={{ width: gutter }}
              className="sticky left-0 z-10 shrink-0 bg-surface"
            />
            <div className="relative" style={{ width: lanePx }}>
              {ticks.map((t) => (
                <span
                  key={t}
                  style={{ left: x(t) }}
                  className="absolute inset-y-0 border-l border-border-strong"
                >
                  <span className="absolute left-1 text-mono-micro leading-[1.125rem] whitespace-nowrap text-subtle">
                    {formatLatency(t)}
                  </span>
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
                    {formatLatency(gap.end - gap.start)} idle
                  </span>
                ) : null,
              )}
            </div>
          </div>

          <div
            aria-hidden="true"
            style={{ left: gutter, width: lanePx }}
            className="pointer-events-none absolute top-[1.125rem] bottom-0"
          >
            {ticks.map((t) => (
              <span
                key={t}
                style={{ left: x(t) }}
                className="pointer-events-none absolute inset-y-0 border-l border-border"
              />
            ))}
            {gaps
              .filter((gap) => x(gap.end) - x(gap.start) >= IDLE_MIN_PX)
              .map((gap) => (
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

          <ul
            aria-label="Timeline"
            data-trace-rows
            className="relative flex flex-col"
          >
            {row({
              id: null,
              type: "TRACE",
              name: trace.name,
              depth: 0,
              treeLines: [],
              isLast: true,
              startMs: 0,
              endMs: totalMs,
              cost: totalCost,
              dimmed: false,
            })}
            {items.map(({ node, treeLines, isLast }) =>
              row({
                id: node.obs.id,
                type: node.obs.type,
                name: node.obs.name,
                depth: node.depth,
                treeLines,
                isLast,
                startMs: node.obs.startMs,
                endMs: node.obs.endMs,
                cost: costs.get(node.obs.id) ?? 0,
                dimmed: hasQuery && !matchesSearch(node.obs, query),
              }),
            )}
          </ul>
        </div>
      </div>
    </div>
  )
}
