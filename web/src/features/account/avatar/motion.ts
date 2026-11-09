/**
 * The picker's motion vocabulary, in one place so every part of it moves with
 * the same physics.
 *
 * Anything that follows a hand (the drag's landing, the lens chasing the
 * pointer) is a spring stepped per frame, because a spring keeps the velocity
 * it is handed and can be retargeted mid-flight without a seam. Anything that
 * runs once to completion (the opening bloom, a shuffle's travel, the frame's
 * pulse) is a Web Animation or a CSS transition, which runs off the main
 * thread; the shuffle uses the same spring baked into a `linear()` curve.
 */

export interface Spring {
  stiffness: number
  damping: number
}

/** A spring by its feel: `bounce` 0 is critically damped, 0.2 overshoots a hair. */
export function spring(stiffness: number, bounce: number): Spring {
  return { stiffness, damping: 2 * Math.sqrt(stiffness) * (1 - bounce) }
}

/** Where a drag lands: firm enough to feel decided, with the faintest settle. */
export const LANDING = spring(240, 0.12)
/** The lens chasing the pointer: quick and without overshoot, so it never wobbles. */
export const LENS = spring(420, 0.04)

export interface Axis {
  value: number
  velocity: number
  target: number
}

/** One semi-implicit Euler step, in seconds. Stable at frame-sized steps. */
export function step(axis: Axis, { stiffness, damping }: Spring, dt: number) {
  const force =
    -stiffness * (axis.value - axis.target) - damping * axis.velocity
  axis.velocity += force * dt
  axis.value += axis.velocity * dt
}

export function isAtRest(axis: Axis, precision = 0.05): boolean {
  return (
    Math.abs(axis.value - axis.target) < precision &&
    Math.abs(axis.velocity) < precision * 10
  )
}

/**
 * A spring from 0 to 1, sampled into a CSS `linear()` easing, with the time it
 * takes to settle. Lets a one-shot transition move like the hand-driven parts.
 */
export function springEasing(
  { stiffness, damping }: Spring,
  samples = 40,
): { easing: string; duration: number } {
  const dt = 1 / 240
  const axis: Axis = { value: 0, velocity: 0, target: 1 }
  const trace: number[] = [0]
  let t = 0
  while (t < 2 && !(t > 0.1 && isAtRest(axis, 0.001))) {
    step(axis, { stiffness, damping }, dt)
    trace.push(axis.value)
    t += dt
  }
  const points: string[] = []
  for (let i = 0; i <= samples; i++) {
    const v = trace[Math.round((i / samples) * (trace.length - 1))]
    points.push(i === samples ? "1" : v.toFixed(4))
  }
  return {
    easing: `linear(${points.join(", ")})`,
    duration: Math.round(t * 1000),
  }
}

/** A strong ease-out, for things arriving: fast to respond, soft to land. */
export const EASE_OUT = "cubic-bezier(0.23, 1, 0.32, 1)"
