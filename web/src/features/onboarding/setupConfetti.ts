/**
 * The burst thrown from the success mark when a workspace's first request
 * lands.
 *
 * The numbers are tuned for a restrained payoff: 14 pieces is where the burst
 * reads as a celebration and still leaves the screen quiet. Denser turns it
 * into a party popper, sparser looks like a rendering bug.
 */
const CONFIG = {
  particleCount: 14,
  spread: 100,
  startVelocity: 32,
  /** Piece size multiplier. 1.3 is roughly a 13px square. */
  scalar: 1.3,
  gravity: 0.8,
  /** Air drag per frame. */
  decay: 0.9,
  /** Lifetime in frames. */
  ticks: 260,
}

/**
 * The palette, read off the running theme rather than written here.
 *
 * A hex at a call site cannot follow a theme and there are two of them, which
 * is the design system's layer rule and is enforced over this file. Reading the
 * tokens has the same shape as every other color in the product and gets the
 * dark theme's brighter teals for free. The two brand steps and the one green,
 * and nothing else: a celebration in six hues is not this product.
 */
const TOKENS = [
  "--color-primary",
  "--color-primary-hover",
  "--color-success",
  "--color-chart-ramp-4",
]

function paletteFromTheme(): string[] {
  if (typeof document === "undefined") return []
  const styles = getComputedStyle(document.documentElement)
  return TOKENS.map((token) => styles.getPropertyValue(token).trim()).filter(
    (value) => value !== "",
  )
}

/** Viewport-relative center of an element, the way canvas-confetti wants it. */
export function confettiOriginOf(element: Element): { x: number; y: number } {
  const { left, top, width, height } = element.getBoundingClientRect()
  return {
    x: (left + width / 2) / window.innerWidth,
    y: (top + height / 2) / window.innerHeight,
  }
}

/**
 * Once per page load, not once per mount.
 *
 * The success screen can mount more than once in a single load: any refetch
 * that flips the sheet between its states remounts the subtree, and keyed to
 * mount alone that reads as endless confetti. Module scope is the right
 * lifetime, because it resets on a real reload, which is also the only time a
 * second celebration would be warranted.
 */
let hasFired = false

/** Test seam. No user-facing caller resets this. */
export function resetSetupConfetti(): void {
  hasFired = false
}

/**
 * Fire the burst, loading the library on the way.
 *
 * Dynamically imported because this is the only screen in the product that
 * celebrates anything, and `disableForReducedMotion` is what makes it a no-op
 * for a reader who asked for less motion. That also keeps the Playwright
 * renders deterministic, since those runs force the preference.
 */
export async function fireSetupConfetti(origin: {
  x: number
  y: number
}): Promise<void> {
  if (hasFired) return
  hasFired = true
  const colors = paletteFromTheme()
  const { default: confetti } = await import("canvas-confetti")
  await confetti({
    ...CONFIG,
    origin,
    ...(colors.length > 0 ? { colors } : {}),
    shapes: ["square"],
    disableForReducedMotion: true,
  })
}
