import { useCallback, useState } from "react"

import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { Chip } from "@/design-system/indicators/Chip"
import { Tooltip } from "@/design-system/overlays/Tooltip"
import {
  buildContextMap,
  type ContextBlock,
  type ContextKind,
  type ContextMap,
  fitBlockLabel,
  formatBlockTokens,
  formatShare,
  formatUsedPercent,
  formatWindow,
} from "./contextMap"
import { type Placed, squarify } from "./squarify"
import type { Observation } from "./types"

// Each kind's color is a chart token. The tile and its border are that color
// mixed into the surface, so the title keeps the page's own ink on both themes.
const TONE: Record<ContextKind, string> = {
  system: "[--tile:var(--color-chart-cat-other)]",
  tools: "[--tile:var(--color-chart-cat-1)]",
  files: "[--tile:var(--color-chart-cat-6)]",
  messages: "[--tile:var(--color-chart-cat-3)]",
  toolResults: "[--tile:var(--color-chart-cat-2)]",
  cacheRead: "[--tile:var(--color-chart-ramp-3)]",
  cacheWrite: "[--tile:var(--color-chart-ramp-4)]",
  fresh: "[--tile:var(--color-chart-cat-other)]",
  free: "",
}

const TILE =
  "bg-[color-mix(in_oklab,var(--tile)_26%,var(--color-surface))] border-[color-mix(in_oklab,var(--tile)_60%,var(--color-surface))] hover:border-(--tile) focus-visible:border-(--tile)"
// A diagonal hatch in the border color, on the surface: nothing is in this space.
const HATCH =
  "bg-surface bg-[repeating-linear-gradient(135deg,transparent_0,transparent_0.375rem,var(--color-border)_0.375rem,var(--color-border)_0.4375rem)] border-dashed border-border hover:border-border-strong focus-visible:border-border-strong"

// Blocks move and resize as the window or the split changes. Geometry is what
// animates here (a treemap's layout is its data), so this is the documented
// exception to transform-and-opacity, bounded to a handful of elements.
const MOTION =
  "transition-[left,top,width,height,transform,box-shadow,border-color,opacity] duration-300 ease-out starting:opacity-0 motion-reduce:transition-none"

const GAP = 2
// The container's own border, top and bottom, which the blocks do not lie on.
const BORDER = 2
const MIN_BORDERED = 4
const TOOLTIP_CLEARANCE = 8

function mapHeight(width: number): number {
  return Math.round(Math.min(360, Math.max(200, width * 0.5)))
}

// A block whose title is its category (Free, Tool results) would say it twice.
function summary(block: ContextBlock): string {
  const amount = `${formatBlockTokens(block.tokens)} tokens · ${formatShare(block.share)}`
  return block.category === block.title
    ? amount
    : `${block.category} · ${amount}`
}

function BlockTooltip({ block }: { block: ContextBlock }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="font-semibold">{block.title}</span>
      <span>{summary(block)}</span>
      {block.notes.map((note) => (
        <span key={note}>{note}</span>
      ))}
    </div>
  )
}

function Swatch({ block }: { block: ContextBlock }) {
  return (
    <span
      aria-hidden="true"
      className={`size-3 shrink-0 border ${block.kind === "free" ? HATCH : `${TONE[block.kind]} ${TILE}`}`}
    />
  )
}

/** The parts by name, for a map too small to write them all on its blocks. */
function Legend({ blocks }: { blocks: ContextBlock[] }) {
  return (
    <ul
      aria-label="Context parts"
      className="m-0 flex list-none flex-col gap-1 p-0 text-caption"
    >
      {blocks.map((block) => (
        <li key={block.kind} className="flex items-center gap-2">
          <Swatch block={block} />
          <span className="min-w-0 flex-1 truncate text-heading">
            {block.title}
          </span>
          <span className="shrink-0 text-mono-caption">
            {`${formatBlockTokens(block.tokens)} · ${formatShare(block.share)}`}
          </span>
        </li>
      ))}
    </ul>
  )
}

function Block({ placed }: { placed: Placed<ContextBlock> }) {
  const block = placed.item
  // Lift the tooltip clear of the whole map, so it never covers a neighbour's label.
  const tooltipOffset = Math.round(placed.y + GAP / 2) + TOOLTIP_CLEARANCE
  // A sliver of a block can come out smaller than the gap; a negative size is invalid CSS.
  const width = Math.max(0, placed.width - GAP)
  const height = Math.max(0, placed.height - GAP)
  const isFree = block.kind === "free"
  const label = fitBlockLabel(block, width, height)
  const showTitle = label.text !== undefined
  // Padding and a border cannot be drawn inside a box smaller than they are, so a
  // block gets them only where it has the room, and never grows past its share.
  const hasRoom = width >= MIN_BORDERED && height >= MIN_BORDERED
  return (
    <Tooltip
      content={<BlockTooltip block={block} />}
      delay={0}
      placement="top"
      offset={tooltipOffset}
    >
      {(props) => (
        <button
          type="button"
          {...props}
          aria-label={`${block.title}: ${summary(block)}. ${block.notes.join(". ")}`}
          style={{
            left: placed.x + GAP / 2,
            top: placed.y + GAP / 2,
            width,
            height,
          }}
          className={`absolute flex min-w-0 cursor-default flex-col items-start justify-start gap-0.5 overflow-hidden text-left ${hasRoom ? "border" : ""} ${showTitle ? "p-2" : ""} hover:z-10 hover:-translate-y-px hover:shadow-elevation-md focus-visible:z-10 focus-visible:-translate-y-px focus-visible:shadow-elevation-md ${MOTION} ${isFree ? HATCH : `${TONE[block.kind]} ${TILE}`}`}
        >
          {showTitle ? (
            <span className="w-full truncate text-heading">{label.text}</span>
          ) : null}
          {label.count ? (
            <span className="w-full truncate text-caption">
              {formatBlockTokens(block.tokens)}
              {label.category ? ` · ${block.category}` : ""}
            </span>
          ) : null}
        </button>
      )}
    </Tooltip>
  )
}

function Treemap({ map }: { map: ContextMap }) {
  // Blocks wait for the first measurement, so they animate in at their real
  // size instead of shrinking from a guess.
  const [width, setWidth] = useState(0)
  const measure = useCallback((element: HTMLFieldSetElement | null) => {
    if (!element) return
    setWidth(Math.floor(element.clientWidth))
    const observer = new ResizeObserver(([entry]) => {
      if (entry && entry.contentRect.width > 0)
        setWidth(Math.floor(entry.contentRect.width))
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [])
  const height = mapHeight(width)
  const placed = squarify(
    map.blocks.map((block) => ({ ...block, value: block.tokens })),
    { x: 0, y: 0, width, height: height - BORDER },
  ).map(({ item, ...rect }) => ({ ...rect, item: item as ContextBlock }))
  // Once measured, a block with no room for its name and count sends every part to a list below.
  const needsLegend =
    width > 0 &&
    placed.some(
      ({ item, width: w, height: h }) =>
        !fitBlockLabel(item, Math.max(0, w - GAP), Math.max(0, h - GAP))
          .isComplete,
    )
  return (
    <div className="flex flex-col gap-3">
      <fieldset
        ref={measure}
        aria-label="Context window map"
        style={{ height }}
        className="relative m-0 min-w-0 w-full overflow-hidden border border-border bg-surface-subtle"
      >
        {placed.map((block) => (
          <Block key={block.item.kind} placed={block} />
        ))}
      </fieldset>
      {needsLegend ? <Legend blocks={map.blocks} /> : null}
    </div>
  )
}

/** One request's context window as a treemap, sized by tokens. */
export function ContextMapView({
  map,
  model,
  outputTokens,
}: {
  map: ContextMap | undefined
  model?: string
  /** The completion tokens of the same request, named beside the input the map is of. */
  outputTokens?: number
}) {
  if (!map)
    return (
      <EmptyMessage>
        This request recorded no input tokens, so there is nothing to map.
      </EmptyMessage>
    )
  const { windowTokens } = map
  const used = formatBlockTokens(map.used)
  return (
    <section aria-label="Context" className="flex flex-col gap-3">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h3 className="flex flex-wrap items-baseline gap-x-2 text-overline">
          Context
          {model ? (
            <span className="text-muted normal-case">
              {windowTokens !== undefined
                ? `${model} [${formatWindow(windowTokens)}]`
                : model}
            </span>
          ) : null}
        </h3>
        <div className="flex items-center gap-2 text-mono-caption">
          {windowTokens !== undefined ? (
            <>
              <span>{`${used} input / ${formatWindow(windowTokens)}`}</span>
              <Chip
                tone={map.used / windowTokens >= 0.9 ? "warning" : "neutral"}
              >
                {formatUsedPercent(map.used, windowTokens)}
              </Chip>
            </>
          ) : (
            <>
              <span>{`${used} input tokens`}</span>
              <Chip>Window unknown</Chip>
            </>
          )}
          {outputTokens ? (
            <span className="text-muted">{`+ ${formatBlockTokens(outputTokens)} output`}</span>
          ) : null}
        </div>
      </div>
      <Treemap map={map} />
      <div className="flex flex-col gap-1 text-caption">
        {outputTokens ? (
          <p>
            {`The map shows input tokens only. The ${formatBlockTokens(outputTokens)} output tokens of this request are counted in the span total, not here.`}
          </p>
        ) : null}
        {map.cache ? (
          <p>
            {`Exact from the provider: ${formatBlockTokens(map.cache.read)} read from the cache, ${formatBlockTokens(map.cache.write)} written to it, ${formatBlockTokens(map.cache.fresh)} fresh input. These overlap the parts above and are not added to them.`}
          </p>
        ) : null}
        {map.isEstimate ? (
          <p>
            <Chip size="sm">Estimated</Chip> Split estimated from request size,
            scaled to the provider's exact input-token count. The total is
            exact.
          </p>
        ) : (
          <p>
            The total and the cache split are the provider's own counts. A split
            by prompt, tools and messages is only kept for requests Otari
            served.
          </p>
        )}
        {windowTokens === undefined ? (
          <p>
            The context window of this model is unknown, so free space is not
            shown.
          </p>
        ) : null}
        {map.isOverWindow && windowTokens !== undefined ? (
          <p className="text-warning">
            {`The input is larger than the ${formatWindow(windowTokens)} window recorded for this model. It may have been compacted, or the window may be wrong, so free space is not shown.`}
          </p>
        ) : null}
      </div>
    </section>
  )
}

/** The Context tab of a model request. */
export function ContextTab({ observation }: { observation: Observation }) {
  return (
    <ContextMapView
      model={observation.model}
      outputTokens={observation.completionTokens}
      map={buildContextMap({
        inputTokens: observation.promptTokens,
        cacheReadTokens: observation.cachedTokens,
        cacheWriteTokens: observation.cacheWriteTokens,
        windowTokens: observation.contextWindow,
        weights: observation.context,
      })}
    />
  )
}
