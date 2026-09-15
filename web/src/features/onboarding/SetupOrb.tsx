import { lazy, Suspense, useEffect, useState } from "react"
import type { OrbState } from "thinking-orbs"

import { usePrefersReducedMotion } from "@/shared/hooks/usePrefersReducedMotion"

export type SetupOrbPhase = "waiting" | "checking" | "stalled" | "solved"

const ORB_BY_PHASE: Record<SetupOrbPhase, { state: OrbState; speed: number }> =
  {
    waiting: { state: "searching", speed: 0.4 },
    checking: { state: "working", speed: 0.9 },
    stalled: { state: "searching", speed: 0.22 },
    solved: { state: "solving", speed: 0.7 },
  }

// Paired with the setup-orb-layer animations in globals.css.
const CROSSFADE_MS = 420
const ThinkingOrb = lazy(() =>
  import("thinking-orbs").then((module) => ({ default: module.ThinkingOrb })),
)

type OrbLayer = { id: number; phase: SetupOrbPhase }

/** Crossfade canvas phases; the adjacent status text describes them accessibly. */
export function SetupOrb({ phase }: { phase: SetupOrbPhase }) {
  const prefersReducedMotion = usePrefersReducedMotion()
  const [layers, setLayers] = useState<OrbLayer[]>([{ id: 0, phase }])

  useEffect(() => {
    setLayers((current) => {
      const top = current[current.length - 1]
      if (top.phase === phase) return current
      const next = { id: top.id + 1, phase }
      return prefersReducedMotion ? [next] : [top, next]
    })
  }, [phase, prefersReducedMotion])

  useEffect(() => {
    if (layers.length < 2) return
    const timer = window.setTimeout(
      () => setLayers((current) => current.slice(-1)),
      prefersReducedMotion ? 0 : CROSSFADE_MS,
    )
    return () => window.clearTimeout(timer)
  }, [layers, prefersReducedMotion])

  return (
    <span
      aria-hidden
      className="relative flex size-10 shrink-0 items-center justify-center"
    >
      <Suspense fallback={null}>
        {layers.map((layer, index) => {
          const { state, speed } = ORB_BY_PHASE[layer.phase]
          return (
            <span
              key={layer.id}
              className={`absolute inset-0 flex items-center justify-center ${
                layers.length === 1
                  ? ""
                  : index === layers.length - 1
                    ? "setup-orb-layer-in"
                    : "setup-orb-layer-out"
              }`}
            >
              <span className="flex size-16 shrink-0 scale-[0.625] items-center justify-center">
                <ThinkingOrb
                  aria-hidden="true"
                  size={64}
                  speed={speed}
                  state={state}
                  paused={prefersReducedMotion}
                  theme="auto"
                />
              </span>
            </span>
          )
        })}
      </Suspense>
    </span>
  )
}
