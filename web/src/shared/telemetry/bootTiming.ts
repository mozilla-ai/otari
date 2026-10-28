/**
 * The moments a load of the dashboard passes through, written as User Timing
 * marks, and the one read of them that `app/BootTimingReport.tsx` turns into a
 * `Dashboard Boot Timed` event.
 *
 * Not part of the telemetry seam and not swapped by a build: the marks are
 * facts about this bundle, and both a lab harness reading
 * `performance.getEntriesByType("mark")` and the event read the same ones, so
 * the number a lab run reports and the number production reports come from one
 * instrument.
 *
 * **Nothing here watches the page while it boots.** A mark is a timestamp the
 * app writes when it already knows the moment has come, and the entries are
 * read once, after the load has settled. An earlier probe built on a
 * MutationObserver slowed the boot it was measuring, which is the failure this
 * shape exists to rule out.
 */

import { useLayoutEffect } from "react"

/** The marks this bundle writes. Each is written at most once per load. */
export const BOOT_MARKS = {
  /** `GET /v1/bootstrap` answered, or failed (`main.tsx`). */
  BOOTSTRAP: "boot:bootstrap",
  /** `GET /v1/organizations/me` answered, a 404 included. */
  CONTEXT: "ctx:current",
  /** The signed-in shell or the sign-in screen reached a frame. */
  SHELL: "shell:painted",
  /** Overview's "Spend today" holds a figure and reached a frame. */
  KPIS: "overview:kpis",
  /** A kept answer was replaced by a different current one. */
  REMOUNT: "boot:remount",
} as const

/**
 * The marks a kept answer writes, as `<prefix><outcome>`. Nothing in this build
 * keeps an answer, so neither is ever written here and both read as `"off"`;
 * the names are fixed now so the build that adds them reports through this
 * reader unchanged.
 */
const KEPT_BOOTSTRAP_PREFIX = "boot:kept-bootstrap:"
const KEPT_CONTEXT_PREFIX = "ctx:kept:"

export type KeptOutcome = "hit" | "miss" | "off"
const KEPT_OUTCOMES: ReadonlySet<string> = new Set(["hit", "miss", "off"])

type NavType = "navigate" | "reload" | "back_forward"
const NAV_TYPES: ReadonlySet<string> = new Set([
  "navigate",
  "reload",
  "back_forward",
])

const EFFECTIVE_TYPES: ReadonlySet<string> = new Set([
  "slow-2g",
  "2g",
  "3g",
  "4g",
])

/** Rounded to this, so a value says when and not which millisecond. */
const ROUND_MS = 10
/** And capped here: anything later is a stalled load, not a slow one. */
const CAP_MS = 60_000

const listeners = new Map<string, Set<() => void>>()

function hasUserTiming(): boolean {
  return (
    typeof performance !== "undefined" &&
    typeof performance.mark === "function" &&
    typeof performance.getEntriesByName === "function"
  )
}

/**
 * Write `name` now, unless this load already has it.
 *
 * Once rather than every time, because most of the moments recur: the context
 * is read again on every refetch and the shell remounts on a sign-in. The first
 * one is the boot; a buffer that grew by one entry a minute for as long as the
 * tab stayed open would be a leak paid for a number nobody reads.
 */
export function markOnce(name: string): void {
  if (!hasUserTiming()) return
  if (performance.getEntriesByName(name, "mark").length > 0) return
  performance.mark(name)
  const waiting = listeners.get(name)
  if (!waiting) return
  listeners.delete(name)
  for (const listener of waiting) listener()
}

/**
 * Write `name` once the frame now being built has been presented.
 *
 * Two frames and not one: a callback in the first `requestAnimationFrame` runs
 * before that frame paints, so the second is the earliest point that is
 * reliably after it. A hidden tab presents no frame and runs neither, so the
 * mark is absent there rather than wrong, which is what the reader expects.
 */
export function markAfterPaint(name: string): void {
  if (typeof requestAnimationFrame !== "function") return
  requestAnimationFrame(() => {
    requestAnimationFrame(() => markOnce(name))
  })
}

/**
 * `markAfterPaint` from the first commit in which `ready` holds: the one that
 * mounted the component, by default.
 *
 * A layout effect rather than a plain one, because a passive effect can run
 * after the browser has already painted that commit, and the two frames would
 * then be counted from the wrong one.
 */
export function useMarkAfterPaint(name: string, ready = true): void {
  useLayoutEffect(() => {
    if (ready) markAfterPaint(name)
  }, [name, ready])
}

/**
 * Call `listener` once `name` is written, or straight away if it already is.
 * Returns the unsubscribe. A registry rather than an observer: it is told by
 * `markOnce`, which runs anyway, and costs nothing while nobody is waiting.
 */
export function whenMarked(name: string, listener: () => void): () => void {
  if (hasUserTiming() && performance.getEntriesByName(name, "mark").length) {
    queueMicrotask(listener)
    return () => undefined
  }
  const waiting = listeners.get(name) ?? new Set()
  waiting.add(listener)
  listeners.set(name, waiting)
  return () => {
    waiting.delete(listener)
  }
}

/** When `name` was first written, from navigation start, bounded. */
function markedAt(name: string): number | undefined {
  if (!hasUserTiming()) return undefined
  const [entry] = performance.getEntriesByName(name, "mark")
  if (!entry) return undefined
  const rounded = Math.round(entry.startTime / ROUND_MS) * ROUND_MS
  return Math.min(Math.max(rounded, 0), CAP_MS)
}

function keptOutcome(prefix: string): KeptOutcome {
  if (!hasUserTiming()) return "off"
  for (const entry of performance.getEntriesByType("mark")) {
    if (!entry.name.startsWith(prefix)) continue
    const outcome = entry.name.slice(prefix.length)
    if (KEPT_OUTCOMES.has(outcome)) return outcome as KeptOutcome
  }
  return "off"
}

function navType(): NavType | undefined {
  if (typeof performance.getEntriesByType !== "function") return undefined
  const [entry] = performance.getEntriesByType("navigation")
  const type = (entry as PerformanceNavigationTiming | undefined)?.type
  return type && NAV_TYPES.has(type) ? (type as NavType) : undefined
}

/**
 * The two client hints the event carries, each only where the browser offers
 * it. Neither is in TypeScript's DOM library yet (both are Chromium-only), so
 * they are read through the shape they have rather than assumed.
 */
interface ClientHints {
  userAgentData?: { mobile?: unknown }
  connection?: { effectiveType?: unknown }
}

function device(): "mobile" | "desktop" | undefined {
  const mobile = (navigator as ClientHints).userAgentData?.mobile
  if (typeof mobile !== "boolean") return undefined
  return mobile ? "mobile" : "desktop"
}

function effectiveType(): string | undefined {
  const type = (navigator as ClientHints).connection?.effectiveType
  return typeof type === "string" && EFFECTIVE_TYPES.has(type)
    ? type
    : undefined
}

/** What this load's marks and the browser say, read once. */
export interface BootTiming {
  t_shell_ms?: number
  t_kpi_ms?: number
  t_bootstrap_ms?: number
  t_context_ms?: number
  nav_type?: NavType
  kept_bootstrap: KeptOutcome
  kept_context: KeptOutcome
  remounted: boolean
  device?: "mobile" | "desktop"
  effective_type?: string
}

export function readBootTiming(): BootTiming {
  return {
    t_shell_ms: markedAt(BOOT_MARKS.SHELL),
    t_kpi_ms: markedAt(BOOT_MARKS.KPIS),
    t_bootstrap_ms: markedAt(BOOT_MARKS.BOOTSTRAP),
    t_context_ms: markedAt(BOOT_MARKS.CONTEXT),
    nav_type: navType(),
    kept_bootstrap: keptOutcome(KEPT_BOOTSTRAP_PREFIX),
    kept_context: keptOutcome(KEPT_CONTEXT_PREFIX),
    remounted: markedAt(BOOT_MARKS.REMOUNT) !== undefined,
    device: device(),
    effective_type: effectiveType(),
  }
}
