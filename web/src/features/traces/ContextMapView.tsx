import { useCallback, useRef, useState } from "react"

import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { Tooltip } from "@/design-system/overlays/Tooltip"
import {
  blockTransition,
  type ContextBlock,
  type ContextKind,
  type ContextMap,
  ESTIMATED,
  fitBlockLabel,
  formatBlockShare,
  formatBlockTokens,
  formatWindow,
  mapNote,
} from "./contextMap"
import { type Placed, squarify } from "./squarify"

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
// exception to transform-and-opacity, bounded to a handful of elements, and
// only while the data changes: a container being resized snaps. The timings are
// inline (`blockTransition`), so the reduced-motion guard has to be important
// to win over them. Transitions rather than keyframes, so a block retargets from
// where it is when the data changes again.
const MOTION = "starting:opacity-0 motion-reduce:transition-none!"
// Hover lifts a block a pixel; a keyboard arriving at one does not move it, only
// outlines it. `hover:` is already gated on a hover-capable pointer.
const LIFT =
  "hover:z-10 hover:-translate-y-px motion-reduce:hover:translate-y-0 focus-visible:z-10"

const GAP = 2
// The container's own border, top and bottom, which the blocks do not lie on.
const BORDER = 2
const MIN_BORDERED = 4

function mapHeight(width: number): number {
  return Math.round(Math.min(240, Math.max(140, width * 0.37)))
}

// A block whose title is its category (Free, Tool results) would say it twice.
function summary(block: ContextBlock): string {
  const amount = `${formatBlockTokens(block.tokens)} tokens · ${formatBlockShare(block.share)}`
  return block.category === block.title
    ? amount
    : `${block.category} · ${amount}`
}

function Swatch({ block }: { block: ContextBlock }) {
  return (
    <span
      aria-hidden="true"
      className={`size-2 shrink-0 ${block.kind === "free" ? "border border-border-strong" : `${TONE[block.kind]} bg-(--tile)`}`}
    />
  )
}

function BlockTooltip({ block }: { block: ContextBlock }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="flex items-center gap-2">
        <Swatch block={block} />
        <span className="text-emphasis">{block.title}</span>
      </span>
      <span className="text-caption">{summary(block)}</span>
      {block.notes.map((note) => (
        <span key={note} className="text-caption text-subtle">
          {note}
        </span>
      ))}
    </div>
  )
}

function Block({
  placed,
  index,
  animateGeometry,
  isEstimate,
}: {
  placed: Placed<ContextBlock>
  index: number
  animateGeometry: boolean
  isEstimate: boolean
}) {
  const block = placed.item
  // A tap opens the card too: a touch screen has no hover, and a block too small
  // to write on has its numbers nowhere else.
  const [isOpen, setIsOpen] = useState(false)
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
      isOpen={isOpen}
      onOpenChange={setIsOpen}
    >
      {(props) => (
        <button
          type="button"
          {...props}
          onClick={(event) => {
            props.onClick?.(event)
            setIsOpen(true)
          }}
          aria-label={[
            `${block.title}: ${summary(block)}`,
            ...block.notes,
            ...(isEstimate && !isFree ? [ESTIMATED] : []),
          ].join(". ")}
          style={{
            left: placed.x + GAP / 2,
            top: placed.y + GAP / 2,
            width,
            height,
            ...blockTransition(index, animateGeometry),
          }}
          className={`absolute flex min-w-0 cursor-default flex-col items-start justify-start gap-0.5 overflow-hidden text-left ${hasRoom ? "border" : ""} ${showTitle ? "p-2" : ""} ${LIFT} ${MOTION} ${isFree ? HATCH : `${TONE[block.kind]} ${TILE}`}`}
        >
          {showTitle ? (
            <span className="w-full truncate text-emphasis">{label.text}</span>
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
  // A new width snaps the blocks; the next change of the data animates them again.
  const [isResizing, setIsResizing] = useState(false)
  const signature = `${map.windowTokens}|${map.blocks.map((block) => `${block.kind}:${block.tokens}`).join()}`
  const [seen, setSeen] = useState(signature)
  if (seen !== signature) {
    setSeen(signature)
    setIsResizing(false)
  }
  const measured = useRef(0)
  const measure = useCallback((element: HTMLFieldSetElement | null) => {
    if (!element) return
    measured.current = Math.floor(element.clientWidth)
    setWidth(measured.current)
    const observer = new ResizeObserver(([entry]) => {
      if (entry && entry.contentRect.width > 0) {
        const next = Math.floor(entry.contentRect.width)
        if (next !== measured.current) {
          if (measured.current !== 0) setIsResizing(true)
          measured.current = next
          setWidth(next)
        }
      }
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [])
  const height = mapHeight(width)
  const placed = squarify(
    map.blocks.map((block) => ({ ...block, value: block.tokens })),
    { x: 0, y: 0, width, height: height - BORDER },
  ).map(({ item, ...rect }) => ({ ...rect, item: item as ContextBlock }))
  return (
    <fieldset
      ref={measure}
      aria-label="Context window map"
      style={{ height }}
      className="relative m-0 min-w-0 w-full overflow-hidden border border-border bg-surface-subtle"
    >
      {placed.map((block, index) => (
        <Block
          key={block.item.kind}
          placed={block}
          index={index}
          animateGeometry={!isResizing}
          isEstimate={map.isEstimate}
        />
      ))}
    </fieldset>
  )
}

/** One request's context window as a treemap, sized by tokens, and what its numbers mean. */
export function ContextMapView({
  map,
  lead,
}: {
  map: ContextMap | undefined
  /** Which request the map is of, said before what its numbers mean. */
  lead?: string
}) {
  if (!map)
    return (
      <EmptyMessage>
        This request recorded no input tokens, so there is nothing to map.
      </EmptyMessage>
    )
  return (
    <>
      <Treemap map={map} />
      <p className="text-caption text-subtle">
        {lead ? `${lead} ${mapNote(map)}` : mapNote(map)}
      </p>
      {map.isOverWindow && map.windowTokens !== undefined ? (
        <p className="text-caption text-warning">
          {`More than the ${formatWindow(map.windowTokens)} window the catalog lists for this model, so free space is not shown. The request may have run with a larger window.`}
        </p>
      ) : null}
    </>
  )
}
