import {
  type KeyboardEvent,
  type PointerEvent,
  useEffect,
  useRef,
  useState,
} from "react"

import { AvatarFace } from "./AvatarFace"
import { face } from "./faces"
import {
  type Axis,
  EASE_OUT,
  isAtRest,
  LANDING,
  LENS,
  prefersReducedMotion,
  spring,
  springEasing,
  step,
} from "./motion"

/**
 * A honeycomb of faces behind an oval window, read like a picker wheel: a
 * frame sits in the middle, and whichever face rests in it is the choice.
 *
 * Drag it and it follows the finger 1:1; let go and it carries on with the
 * velocity you gave it, then lands on the face nearest where that momentum
 * would take it, on a spring, so a flick travels and a nudge settles. Tap a
 * face and it glides into the frame. Arrow keys step without motion. At rest
 * it is always the same shape: rows of 5, 8, 9, 8, 5 (3, 6, 7, 6, 3 on a
 * phone), the outer ring a little smaller and lighter.
 *
 * The lattice repeats every `COLS` by `ROWS` cells, one face per cell, so it
 * never runs out. The period is wider and taller than the window, so no face
 * is on show twice, and `ROWS` is even so the stagger lines up at the seam.
 */
const COLS = 12
const ROWS = 6

// A face fills 70% of its cell and rows sit one cell apart, so row and column
// gaps match and a face swollen by the lens still clears its neighbours. A
// narrow frame gets a smaller shape at full size rather than a shrunken one,
// which keeps every cell over the 44px touch floor.
const MAX_PITCH = 52
const MIDDLE_ROW_SPAN = 9.8
const COMPACT_BELOW = 480
const COMPACT_MIDDLE_ROW_SPAN = 7.6
const ROW_RATIO = 1
const TILE_RATIO = 0.7

// The window, in cells: half-sizes that leave exactly the shapes above inside
// when the lattice rests on a cell. The settled outer faces sit at about 0.88
// of it and the next ring out at about 1.0, so faces shrink and lighten
// towards `EDGE` and anything drifting past fades out before the next ring.
const WINDOW_COLS = 4.6
const COMPACT_WINDOW_COLS = 3.4
const WINDOW_ROWS = 2.6
const RING_SCALE = 0.74
const RING_OPACITY = 0.6
const EDGE = 0.92
const FADE = 0.04

// The lens: how much the face under the pointer (or in the frame) grows.
const LENS_GAIN = 0.16
const LENS_REACH_CELLS = 0.7

// How far ahead of a release the landing is aimed, in seconds of the release
// velocity: a flick of 1000px/s travels about four faces before it settles.
const PROJECTION = 0.2
const DRAG_SLOP = 6
// Pointer samples older than this do not count towards the release velocity,
// so a drag that stopped before letting go lands where it is.
const VELOCITY_WINDOW_MS = 80

// One-shot motion. The bloom plays once per opening, ring by ring outwards.
// A shuffle is a ripple from the frame outwards: each face folds away, swaps
// at the fold's deepest point, and unfolds as the new face, so the shape never
// breaks and nothing crosses the frame. The frame pulses on a stiff, slightly
// bouncy spring when a face lands in it.
const BLOOM_MS = 460
const BLOOM_RING_DELAY = 34
const BLOOM_LEAD = 70
const FLIP_MS = 380
const FLIP_RING_DELAY = 46
const FLIP_SWAP_AT = 0.45
const FLIP_EASE = "cubic-bezier(0.77, 0, 0.175, 1)"
const PULSE = springEasing(spring(520, 0.32))
const PRESS_SCALE = 0.94

export type ChangeSource = "drag" | "tap" | "key"

interface Cell {
  r: number
  c: number
}

function mod(n: number, m: number): number {
  return ((n % m) + m) % m
}

function cellCentre({ r, c }: Cell, pitch: number): { x: number; y: number } {
  return {
    x: c * pitch + (mod(r, 2) ? pitch / 2 : 0),
    y: r * pitch * ROW_RATIO,
  }
}

function cellAt(x: number, y: number, pitch: number): Cell {
  const r = Math.round(y / (pitch * ROW_RATIO))
  const c = Math.round((x - (mod(r, 2) ? pitch / 2 : 0)) / pitch)
  return { r, c }
}

function axis(value = 0): Axis {
  return { value, velocity: 0, target: value }
}

function lensGain(distance: number, reach: number): number {
  return 1 + LENS_GAIN * Math.exp(-(distance ** 2) / (2 * reach ** 2))
}

// A haptic tick where the platform has one (Android); nothing elsewhere.
function haptic() {
  navigator.vibrate?.(6)
}

export function AvatarField({
  order,
  value,
  onChange,
}: {
  /** Face id for each of the `COLS * ROWS` cells of one period, row-major. */
  order: readonly number[]
  value: number
  onChange: (id: number, source: ChangeSource) => void
}) {
  const frameRef = useRef<HTMLDivElement>(null)
  const reticleRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ width: 520, height: 300 })
  const isCompact = size.width < COMPACT_BELOW
  const pitch = Math.min(
    MAX_PITCH,
    size.width / (isCompact ? COMPACT_MIDDLE_ROW_SPAN : MIDDLE_ROW_SPAN),
  )
  const pitchRef = useRef(pitch)
  pitchRef.current = pitch
  const lastPitch = useRef(pitch)

  // The physics lives in refs and is stepped by one frame loop, which asks
  // for a render per frame while anything is moving.
  const offset = useRef({ x: axis(), y: axis(), isLanding: false })
  const lens = useRef({ x: axis(), y: axis() })
  const [, setFrame] = useState(0)
  const loop = useRef<number | null>(null)

  const [shown, setShown] = useState(order)
  // During a ripple, the order each face showed before it, and when the
  // ripple started: a face keeps its old look until the fold reaches it.
  const ripple = useRef<{ from: readonly number[]; start: number } | null>(null)
  const [pressed, setPressed] = useState<string | null>(null)

  const onChangeRef = useRef(onChange)
  onChangeRef.current = onChange
  const shownRef = useRef(shown)
  shownRef.current = shown
  const centreId = useRef(value)
  // While a tapped face glides into the frame, the faces it passes over are
  // not choices: the tap already chose.
  const gliding = useRef(false)
  const hasBloomed = useRef(false)
  const animated = useRef(new WeakSet<Element>())
  const drag = useRef<{
    id: number
    pointerType: string
    startX: number
    startY: number
    from: { x: number; y: number }
    moved: boolean
    samples: { t: number; x: number; y: number }[]
  } | null>(null)

  const idAt = (cell: Cell) =>
    shownRef.current[mod(cell.r, ROWS) * COLS + mod(cell.c, COLS)]

  const centredCell = () =>
    cellAt(-offset.current.x.value, -offset.current.y.value, pitchRef.current)

  const pulse = () => {
    if (prefersReducedMotion()) return
    reticleRef.current?.animate([{ scale: "1.12" }, { scale: "1" }], {
      duration: PULSE.duration,
      easing: PULSE.easing,
    })
  }

  // Report the face in the frame whenever it changes during a drag or a coast.
  const reportCentre = () => {
    if (gliding.current) return
    const id = idAt(centredCell())
    if (id === centreId.current) return
    centreId.current = id
    onChangeRef.current(id, "drag")
    if (drag.current?.pointerType === "touch") haptic()
  }

  const run = () => {
    if (loop.current !== null) return
    let last = performance.now()
    const frame = (now: number) => {
      const dt = Math.min(1 / 30, (now - last) / 1000)
      last = now
      const o = offset.current
      if (o.isLanding) {
        step(o.x, LANDING, dt)
        step(o.y, LANDING, dt)
        if (isAtRest(o.x) && isAtRest(o.y)) {
          o.x = axis(o.x.target)
          o.y = axis(o.y.target)
          o.isLanding = false
          gliding.current = false
          reportCentre()
          pulse()
          haptic()
        } else {
          reportCentre()
        }
      }
      const l = lens.current
      step(l.x, LENS, dt)
      step(l.y, LENS, dt)
      const isLensMoving = !(isAtRest(l.x, 0.2) && isAtRest(l.y, 0.2))
      setFrame((n) => n + 1)
      loop.current =
        o.isLanding || isLensMoving ? requestAnimationFrame(frame) : null
    }
    loop.current = requestAnimationFrame(frame)
  }

  const stopLanding = () => {
    const o = offset.current
    o.x = axis(o.x.value)
    o.y = axis(o.y.value)
    o.isLanding = false
    gliding.current = false
  }

  /** Land on `cell`, carrying `velocity` (px/s) into the spring. */
  const landOn = (cell: Cell, velocity = { x: 0, y: 0 }) => {
    const { x, y } = cellCentre(cell, pitchRef.current)
    const o = offset.current
    if (prefersReducedMotion()) {
      o.x = axis(-x)
      o.y = axis(-y)
      o.isLanding = false
      gliding.current = false
      reportCentre()
      setFrame((n) => n + 1)
      return
    }
    o.x.target = -x
    o.y.target = -y
    o.x.velocity = velocity.x
    o.y.velocity = velocity.y
    o.isLanding = true
    run()
  }

  const jumpTo = (cell: Cell) => {
    const { x, y } = cellCentre(cell, pitchRef.current)
    stopLanding()
    offset.current.x = axis(-x)
    offset.current.y = axis(-y)
    setFrame((n) => n + 1)
  }

  const aimLens = (x: number, y: number) => {
    const l = lens.current
    if (prefersReducedMotion()) {
      l.x = axis(x)
      l.y = axis(y)
      setFrame((n) => n + 1)
      return
    }
    l.x.target = x
    l.y.target = y
    run()
  }

  // The copy of `id` nearest the middle of what is on screen.
  const nearestCell = (id: number): Cell => {
    const slot = Math.max(0, shownRef.current.indexOf(id))
    const r0 = Math.floor(slot / COLS)
    const c0 = slot % COLS
    const view = { x: -offset.current.x.value, y: -offset.current.y.value }
    let best: Cell = { r: r0, c: c0 }
    let bestDistance = Number.POSITIVE_INFINITY
    const kr = Math.round((view.y / (pitch * ROW_RATIO) - r0) / ROWS)
    const kc = Math.round((view.x / pitch - c0) / COLS)
    for (let i = kr - 1; i <= kr + 1; i++) {
      for (let j = kc - 1; j <= kc + 1; j++) {
        const cell = { r: r0 + i * ROWS, c: c0 + j * COLS }
        const { x, y } = cellCentre(cell, pitch)
        const distance = Math.hypot(x - view.x, y - view.y)
        if (distance < bestDistance) {
          bestDistance = distance
          best = cell
        }
      }
    }
    return best
  }

  // A new frame width rescales the lattice; keep the same cell in the frame.
  useEffect(() => {
    const o = offset.current
    const cell = cellAt(-o.x.value, -o.y.value, lastPitch.current)
    lastPitch.current = pitch
    const { x, y } = cellCentre(cell, pitch)
    o.x = axis(-x)
    o.y = axis(-y)
    o.isLanding = false
    setFrame((n) => n + 1)
  }, [pitch])

  // Open on the current face, without motion: it is where the picker starts,
  // not a move the person made. The bloom is the only opening motion.
  // biome-ignore lint/correctness/useExhaustiveDependencies: once, on mount
  useEffect(() => {
    jumpTo(nearestCell(value))
    const done = window.setTimeout(() => {
      hasBloomed.current = true
    }, 120)
    return () => window.clearTimeout(done)
  }, [])

  // A new order is a shuffle. The face in the frame keeps its cell (the
  // dialog builds the order that way), so it stays put while the ripple runs
  // outwards from it.
  // biome-ignore lint/correctness/useExhaustiveDependencies: keyed on the order on purpose
  useEffect(() => {
    if (order === shown) return
    if (prefersReducedMotion()) {
      setShown(order)
      return
    }
    const start = performance.now()
    ripple.current = { from: shown, start }
    setShown(order)
    let longest = 0
    for (const node of frameRef.current?.querySelectorAll<HTMLElement>(
      "[data-ring]",
    ) ?? []) {
      const ring = Number(node.dataset.ring)
      if (ring === 0) continue
      const delay = ring * FLIP_RING_DELAY
      longest = Math.max(longest, delay)
      node.animate(
        [
          { transform: "scale(1)", opacity: 1, filter: "blur(0)" },
          {
            transform: "scale(0.55)",
            opacity: 0,
            filter: "blur(2px)",
            offset: FLIP_SWAP_AT,
          },
          { transform: "scale(1)", opacity: 1, filter: "blur(0)" },
        ],
        { duration: FLIP_MS, delay, easing: FLIP_EASE },
      )
    }
    // Re-render through the ripple so each face swaps as its fold bottoms out.
    const end = start + longest + FLIP_MS * FLIP_SWAP_AT + 32
    let raf = requestAnimationFrame(function frame(now) {
      setFrame((n) => n + 1)
      if (now < end) raf = requestAnimationFrame(frame)
      else ripple.current = null
    })
    return () => cancelAnimationFrame(raf)
  }, [order])

  // biome-ignore lint/correctness/useExhaustiveDependencies: listeners bind once; everything they touch is a ref
  useEffect(() => {
    const frame = frameRef.current
    if (!frame) return
    const observer = new ResizeObserver(([entry]) => {
      setSize({
        width: entry.contentRect.width,
        height: entry.contentRect.height,
      })
    })
    observer.observe(frame)
    // A wheel or trackpad pans the field and lands when it goes quiet.
    // Registered here because React's wheel listener is passive and cannot
    // stop the dialog scrolling underneath.
    let quiet = 0
    const onWheel = (event: WheelEvent) => {
      event.preventDefault()
      stopLanding()
      const o = offset.current
      o.x = axis(o.x.value - event.deltaX)
      o.y = axis(o.y.value - event.deltaY)
      reportCentre()
      setFrame((n) => n + 1)
      window.clearTimeout(quiet)
      quiet = window.setTimeout(() => landOn(centredCell()), 120)
    }
    frame.addEventListener("wheel", onWheel, { passive: false })
    return () => {
      observer.disconnect()
      frame.removeEventListener("wheel", onWheel)
      window.clearTimeout(quiet)
      if (loop.current !== null) cancelAnimationFrame(loop.current)
      loop.current = null
    }
  }, [])

  const local = (event: PointerEvent) => {
    const box = event.currentTarget.getBoundingClientRect()
    return {
      x: event.clientX - box.left - box.width / 2,
      y: event.clientY - box.top - box.height / 2,
    }
  }

  const cellUnder = (event: PointerEvent): string | null =>
    document
      .elementFromPoint(event.clientX, event.clientY)
      ?.closest<HTMLElement>("[data-cell]")?.dataset.cell ?? null

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    // One pointer at a time: a second finger mid-drag would make it jump.
    if (event.button !== 0 || drag.current) return
    stopLanding()
    event.currentTarget.setPointerCapture(event.pointerId)
    const o = offset.current
    drag.current = {
      id: event.pointerId,
      pointerType: event.pointerType,
      startX: event.clientX,
      startY: event.clientY,
      from: { x: o.x.value, y: o.y.value },
      moved: false,
      samples: [{ t: event.timeStamp, x: event.clientX, y: event.clientY }],
    }
    setPressed(cellUnder(event))
  }

  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (event.pointerType === "mouse" && !drag.current?.moved) {
      const at = local(event)
      aimLens(at.x, at.y)
    }
    const state = drag.current
    if (!state || state.id !== event.pointerId) return
    const dx = event.clientX - state.startX
    const dy = event.clientY - state.startY
    if (!state.moved && Math.hypot(dx, dy) < DRAG_SLOP) return
    if (!state.moved) {
      state.moved = true
      setPressed(null)
      // While dragging, the lens sits in the frame, as it always does on touch.
      aimLens(0, 0)
    }
    state.samples.push({
      t: event.timeStamp,
      x: event.clientX,
      y: event.clientY,
    })
    while (
      state.samples.length > 2 &&
      event.timeStamp - state.samples[0].t > VELOCITY_WINDOW_MS
    ) {
      state.samples.shift()
    }
    const o = offset.current
    o.x = axis(state.from.x + dx)
    o.y = axis(state.from.y + dy)
    reportCentre()
    setFrame((n) => n + 1)
  }

  const onPointerUp = (event: PointerEvent<HTMLDivElement>) => {
    const state = drag.current
    if (!state || state.id !== event.pointerId) return
    drag.current = null
    setPressed(null)
    if (!state.moved) {
      const hit = cellUnder(event)
      if (!hit) return
      const [r, c] = hit.split(":").map(Number)
      const id = idAt({ r, c })
      if (id !== centreId.current) {
        centreId.current = id
        gliding.current = true
        onChange(id, "tap")
      }
      landOn({ r, c })
      return
    }
    // Release velocity from the recent samples only, in px/s.
    const first = state.samples[0]
    const latest = state.samples[state.samples.length - 1]
    const span = Math.max(1, latest.t - first.t) / 1000
    const isFresh = event.timeStamp - latest.t < VELOCITY_WINDOW_MS
    const velocity = isFresh
      ? { x: (latest.x - first.x) / span, y: (latest.y - first.y) / span }
      : { x: 0, y: 0 }
    const o = offset.current
    const aim = {
      x: o.x.value + velocity.x * PROJECTION,
      y: o.y.value + velocity.y * PROJECTION,
    }
    landOn(cellAt(-aim.x, -aim.y, pitchRef.current), velocity)
  }

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const moves: Record<string, Cell> = {
      ArrowLeft: { r: 0, c: -1 },
      ArrowRight: { r: 0, c: 1 },
      ArrowUp: { r: -1, c: 0 },
      ArrowDown: { r: 1, c: 0 },
    }
    const move = moves[event.key]
    if (!move) return
    event.preventDefault()
    const now = centredCell()
    const next = { r: now.r + move.r, c: now.c + move.c }
    // No motion on a key press: it lands as fast as the key repeats.
    jumpTo(next)
    const id = idAt(next)
    centreId.current = id
    onChange(id, "key")
  }

  // The cells that can fall inside the window, with one cell of margin so
  // faces drift in from beyond its edge rather than appearing at it.
  const rowHeight = pitch * ROW_RATIO
  const tile = pitch * TILE_RATIO
  const windowX = (isCompact ? COMPACT_WINDOW_COLS : WINDOW_COLS) * pitch
  const windowY = WINDOW_ROWS * rowHeight
  const ox = offset.current.x.value
  const oy = offset.current.y.value
  const rMin = Math.floor((-windowY - oy) / rowHeight) - 1
  const rMax = Math.ceil((windowY - oy) / rowHeight) + 1
  const cMin = Math.floor((-windowX - ox) / pitch) - 1
  const cMax = Math.ceil((windowX - ox) / pitch) + 1
  const lensAt = { x: lens.current.x.value, y: lens.current.y.value }
  const reach = LENS_REACH_CELLS * pitch

  // Read per render, so flipping the OS setting takes effect without
  // reopening. Inline transitions outrank `motion-reduce:`; this does not.
  const isReduced = prefersReducedMotion()
  const isBlooming = !hasBloomed.current && !isReduced
  const renderedAt = performance.now()
  const tiles = []
  // The face resting in the frame, and its on-screen scale, so the frame can
  // hug it as the lens swells and eases it.
  let framedScale = 1
  for (let r = rMin; r <= rMax; r++) {
    for (let c = cMin; c <= cMax; c++) {
      const centre = cellCentre({ r, c }, pitch)
      const x = centre.x + ox
      const y = centre.y + oy
      const rim = Math.hypot(x / windowX, y / windowY)
      if (rim >= EDGE + FADE) continue
      const toRing = Math.min(1, Math.max(0, (rim - 0.6) / (EDGE - 0.6)))
      const opacity =
        rim <= EDGE
          ? 1 - (1 - RING_OPACITY) * toRing
          : RING_OPACITY * (1 - (rim - EDGE) / FADE)
      const scale =
        (1 - (1 - RING_SCALE) * toRing) *
        lensGain(Math.hypot(x - lensAt.x, y - lensAt.y), reach)
      if (Math.abs(x) < pitch / 2 && Math.abs(y) < rowHeight / 2) {
        framedScale = scale
      }
      const cellKey = `${r}:${c}`
      const ring = Math.round(Math.hypot(x / pitch, y / rowHeight))
      const pending =
        ripple.current &&
        renderedAt <
          ripple.current.start + ring * FLIP_RING_DELAY + FLIP_MS * FLIP_SWAP_AT
      const id = pending
        ? (ripple.current as { from: readonly number[] }).from[
            mod(r, ROWS) * COLS + mod(c, COLS)
          ]
        : idAt({ r, c })
      tiles.push(
        // Four layers, one job each, so no two motions fight over a
        // transform. The outer is the cell's place and follows a drag 1:1.
        // The next is the lens, driven per frame by its spring. Then the
        // press, a CSS transition. The innermost takes the one-shot
        // animations: the bloom and the shuffle's fold.
        <div
          key={cellKey}
          data-cell={cellKey}
          className="absolute top-1/2 left-1/2"
          style={{
            width: tile,
            height: tile,
            opacity,
            zIndex: scale > 1.05 ? 1 : 0,
            transform: `translate(${x - tile / 2}px, ${y - tile / 2}px)`,
          }}
        >
          <div className="size-full" style={{ transform: `scale(${scale})` }}>
            <div
              className="size-full"
              style={{
                transform:
                  pressed === cellKey && !isReduced
                    ? `scale(${PRESS_SCALE})`
                    : undefined,
                transition: isReduced
                  ? undefined
                  : `transform 160ms ${EASE_OUT}`,
              }}
            >
              <div
                data-ring={ring}
                ref={
                  isBlooming
                    ? (node) => {
                        if (!node || animated.current.has(node)) return
                        animated.current.add(node)
                        node.animate(
                          [
                            { opacity: 0, transform: "scale(0.8)" },
                            { opacity: 1, transform: "scale(1)" },
                          ],
                          {
                            duration: BLOOM_MS,
                            delay: BLOOM_LEAD + ring * BLOOM_RING_DELAY,
                            easing: EASE_OUT,
                            fill: "backwards",
                          },
                        )
                      }
                    : undefined
                }
                className="size-full"
              >
                <AvatarFace id={id} className="size-full" />
              </div>
            </div>
          </div>
        </div>,
      )
    }
  }

  const frameSize = tile * framedScale + 10

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div
        ref={frameRef}
        role="application"
        aria-roledescription="avatar picker"
        aria-label="Avatars. The face in the frame is your choice. Arrow keys move the frame; drag, scroll or tap a face to choose it."
        // biome-ignore lint/a11y/noNoninteractiveTabindex: the field takes arrow keys itself; `application` hands them to it
        tabIndex={0}
        onKeyDown={onKeyDown}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={() => {
          drag.current = null
          setPressed(null)
          landOn(centredCell())
        }}
        onPointerLeave={() => {
          if (!drag.current) aimLens(0, 0)
        }}
        // Fills the sheet on a phone, where the dialog is the whole screen; a
        // fixed band on a desk, where the dialog floats.
        className="otari-focus-ring relative min-h-[300px] flex-1 cursor-grab touch-none overflow-hidden select-none active:cursor-grabbing sm:h-[300px] sm:flex-none"
      >
        {tiles}
        {/* The frame does not move: faces pass through it. It hugs the face
            it holds as the lens swells and eases that face, and pulses once
            when a new face lands. It blooms in last, after the faces. */}
        <div
          aria-hidden
          className="pointer-events-none absolute top-1/2 left-1/2 z-10"
          style={{
            width: frameSize,
            height: frameSize,
            translate: `${-frameSize / 2}px ${-frameSize / 2}px`,
          }}
        >
          <div
            ref={(node) => {
              reticleRef.current = node
              if (!node || !isBlooming || animated.current.has(node)) return
              animated.current.add(node)
              node.animate(
                [
                  { opacity: 0, scale: "1.25" },
                  { opacity: 1, scale: "1" },
                ],
                {
                  duration: 380,
                  delay: BLOOM_LEAD + 120,
                  easing: EASE_OUT,
                  fill: "backwards",
                },
              )
            }}
            className="border-foreground size-full border-2"
          />
        </div>
      </div>
      <p aria-live="polite" className="sr-only">
        Selected: {face(value).label}
      </p>
    </div>
  )
}
