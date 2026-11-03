import {
  type ReactNode,
  type RefObject,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react"
import {
  FiChevronDown,
  FiChevronUp,
  FiMaximize2,
  FiMinimize2,
  FiX,
} from "react-icons/fi"

import { IconButton } from "@/design-system/actions/IconButton"
import { Kbd } from "@/design-system/indicators/Kbd"
import { Tooltip } from "@/design-system/overlays/Tooltip"

const MIN_PX = 640 // 40rem
const MAX_FRACTION = 0.9
const DEFAULT_FRACTION = 0.5
const DEFAULT_CAP_PX = 1400 // 87.5rem
// One arrow press on the panel's edge, as a share of the window.
const WIDTH_STEP = 0.05
const EXIT_MS = 250

function defaultWidth(): number {
  return Math.max(
    MIN_PX,
    Math.min(window.innerWidth * DEFAULT_FRACTION, DEFAULT_CAP_PX),
  )
}

/**
 * A 4px bar in a 12px strip straddling an edge: transparent at rest, drawn on
 * hover, focus and drag. A drag writes at most once a frame, from the last
 * pointer position.
 */
export function ResizeHandle({
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

/**
 * One record of a list, open in a panel fixed to the right of the page: the
 * frame the Sessions panel (`features/traces/TracePeek`) draws, for any list
 * whose rows are read one after another.
 *
 * Beside the list rather than over it from `md` up, so another row can be
 * picked while one is open; below `md` it covers the page, and the caller makes
 * the page `inert` while it does. The bar walks the list (also J and K), widens
 * the panel and closes it (also Escape). Focus moves into the panel as it opens
 * and back to whatever opened it as it closes.
 *
 * Presentational: which record is open, and which are before and after it,
 * belong to the caller.
 */
export function PeekPanel({
  label,
  kind,
  identity,
  noun,
  recordKey,
  focusRef,
  onClose,
  onPrevious,
  onNext,
  children,
}: {
  /** Names the panel's region. */
  label: string
  /** What the record is, as the bar's overline: "Request", "Trace". */
  kind: string
  /** The record's identity beside it, such as its id and a copy control. */
  identity?: ReactNode
  /** The record in the walk controls' names: "Previous request". */
  noun: string
  /** Which record is shown, so a close animating out does not close the next one. */
  recordKey: string
  /** Where focus lands as the panel opens; the panel itself without one. */
  focusRef?: RefObject<HTMLElement | null>
  onClose: () => void
  /** Opens the record above this one, or is absent at the top of the list. */
  onPrevious?: () => void
  /** Opens the record below this one, or is absent at the end of the list. */
  onNext?: () => void
  children: ReactNode
}) {
  const [width, setWidth] = useState(defaultWidth)
  const [expanded, setExpanded] = useState(false)
  const [entered, setEntered] = useState(false)
  const [leavingKey, setLeavingKey] = useState<string | null>(null)
  const panelRef = useRef<HTMLElement>(null)
  const exitTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const shownKey = useRef(recordKey)
  useLayoutEffect(() => {
    shownKey.current = recordKey
  })
  const leaving = leavingKey === recordKey

  // setTimeout rather than requestAnimationFrame: a hidden tab never fires rAF.
  useEffect(() => {
    const timer = setTimeout(() => setEntered(true), 20)
    return () => clearTimeout(timer)
  }, [])

  // biome-ignore lint/correctness/useExhaustiveDependencies: only as the panel opens; walking the list leaves focus where it is
  useEffect(() => {
    const opener =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null
    ;(focusRef?.current ?? panelRef.current)?.focus()
    return () => {
      clearTimeout(exitTimer.current)
      if (opener?.isConnected) opener.focus()
    }
  }, [])

  const close = () => {
    if (leaving) return
    const closing = recordKey
    setLeavingKey(closing)
    clearTimeout(exitTimer.current)
    exitTimer.current = setTimeout(() => {
      if (shownKey.current === closing) onClose()
    }, EXIT_MS)
  }

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      // A field owns its own keys, Escape included.
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
  const shownWidth = expanded ? window.innerWidth * MAX_FRACTION : width

  return (
    <aside
      ref={panelRef}
      aria-label={label}
      tabIndex={-1}
      style={{ "--peek-width": `${shownWidth}px` } as React.CSSProperties}
      className={`fixed top-14 right-0 bottom-0 z-30 flex w-full flex-col border-l border-border-strong bg-surface outline-none transition-transform duration-250 ease-out-fluid motion-reduce:transition-none md:w-(--peek-width) ${
        entered && !leaving ? "translate-x-0" : "translate-x-full"
      }`}
    >
      <ResizeHandle
        label={`Resize ${noun} panel`}
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
          <span className="shrink-0 text-overline">{kind}</span>
          {identity ? <span className="flex min-w-0">{identity}</span> : null}
        </div>
        <div className="otari-toolbar flex shrink-0 items-center gap-1">
          <Tooltip
            content={
              <span className="flex items-center gap-2">
                Previous {noun} <Kbd keys={["K"]} />
              </span>
            }
          >
            <IconButton
              label={`Previous ${noun}`}
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
                Next {noun} <Kbd keys={["J"]} />
              </span>
            }
          >
            <IconButton
              label={`Next ${noun}`}
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

      {children}
    </aside>
  )
}
