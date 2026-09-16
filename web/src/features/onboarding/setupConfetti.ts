/** A restrained burst from the center of the settled success dialog. */
// Fourteen pieces read as a celebration without crowding the success message.
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
function confettiOriginOf(element: Element): { x: number; y: number } {
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

/** Measure the settled dialog, after its entrance and the lazy import finish. */
export async function fireSetupConfetti(
  element: Element,
  signal: AbortSignal,
): Promise<void> {
  if (hasFired || signal.aborted) return
  const animations: Animation[] = []
  // canvas-confetti's own default layer, the floor when no ancestor sets one.
  let overlayZIndex = 100
  for (let node: Element | null = element; node; node = node.parentElement) {
    animations.push(...(node.getAnimations?.() ?? []))
    const zIndex = Number(getComputedStyle(node).zIndex)
    if (Number.isFinite(zIndex)) overlayZIndex = Math.max(overlayZIndex, zIndex)
  }
  const [{ default: confetti }] = await Promise.all([
    import("canvas-confetti"),
    Promise.allSettled(
      animations
        .filter(
          (animation) => animation.effect?.getTiming().iterations !== Infinity,
        )
        .map((animation) => animation.finished),
    ),
  ])
  if (hasFired || signal.aborted || !element.isConnected) return
  hasFired = true
  const colors = paletteFromTheme()
  await confetti({
    ...CONFIG,
    origin: confettiOriginOf(element),
    // HeroUI's overlay sits above canvas-confetti's default layer.
    zIndex: overlayZIndex + 1,
    ...(colors.length > 0 ? { colors } : {}),
    shapes: ["square"],
    disableForReducedMotion: true,
  })
}
