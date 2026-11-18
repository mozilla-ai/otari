import { celebrationColors } from "@/features/onboarding/celebrationPalette"

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
  const colors = celebrationColors()
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
