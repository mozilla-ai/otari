import { lazy, Suspense } from "react"
import type { OrbState } from "thinking-orbs"

import { usePrefersReducedMotion } from "@/shared/hooks/usePrefersReducedMotion"

/**
 * What the guide is doing right now, named for the role rather than for the
 * animation. Mapping the two is this module's whole job.
 */
export type SetupOrbPhase =
  // Before the first request lands.
  | "waiting"
  | "checking"
  | "stalled"
  // After it does.
  | "solved"

/**
 * Per-phase tuning.
 *
 * `searching` sweeps a scan meridian across a dotted globe, which reads as
 * watching for something that has not arrived; `working` puts particles on
 * tilted orbits, so a check in flight looks like effort rather than waiting.
 * `stalled` reuses the searching sweep at roughly half speed, so a check that
 * could not complete feels becalmed without going inert. `solving` scrambles
 * bands and clicks them back into place, which is the celebration beat.
 */
const ORB_BY_PHASE: Record<SetupOrbPhase, { state: OrbState; speed: number }> =
  {
    waiting: { state: "searching", speed: 0.4 },
    checking: { state: "working", speed: 0.9 },
    stalled: { state: "searching", speed: 0.22 },
    solved: { state: "solving", speed: 0.7 },
  }

/**
 * Loaded on demand, and only ever on this one screen. It is a canvas engine
 * with nine animations in it, and every page that is not the setup sheet has no
 * use for any of them.
 */
const ThinkingOrb = lazy(() =>
  import("thinking-orbs").then((module) => ({ default: module.ThinkingOrb })),
)

/**
 * The animated mark beside the guide's status line.
 *
 * Decorative, and `aria-hidden`: the sentence next to it is the accessible
 * account of this state, so exposing the canvas as a second image would only
 * make a screen reader announce it twice.
 *
 * `theme="auto"` resolves against the `data-theme` attribute `useTheme` writes
 * on `<html>`, so the ink follows the app's theme with nothing threaded through.
 *
 * Under `prefers-reduced-motion` it is frozen rather than removed. CSS cannot
 * reach inside a canvas, so this is one of the few places that has to ask the
 * preference itself; the frame it holds is still the right mark for the state.
 *
 * The fallback is a box of the orb's own size, so the row does not resize when
 * the chunk arrives.
 */
export function SetupOrb({ phase }: { phase: SetupOrbPhase }) {
  const prefersReducedMotion = usePrefersReducedMotion()
  const { state, speed } = ORB_BY_PHASE[phase]

  return (
    <Suspense fallback={<span aria-hidden className="size-10 shrink-0" />}>
      <span
        aria-hidden
        className="flex size-10 shrink-0 items-center justify-center"
      >
        {/* Rendered at the library's 64px preset and scaled into a 40px box,
            which is the size the band affords. The two shipped sizes are 64 and
            20, tuned separately rather than as a scale factor, and 20 is an
            inline-text mark rather than the subject of a status row. So the
            larger one is scaled: a dot rasterized slightly soft reads better
            here than a mark half the size the design draws. */}
        <ThinkingOrb
          aria-hidden="true"
          size={64}
          speed={speed}
          state={state}
          paused={prefersReducedMotion}
          theme="auto"
          style={{ transform: "scale(0.625)" }}
        />
      </span>
    </Suspense>
  )
}
