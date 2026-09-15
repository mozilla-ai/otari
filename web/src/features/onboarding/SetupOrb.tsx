import { lazy, Suspense, useEffect, useState } from "react"
import type { OrbState } from "thinking-orbs"

import { usePrefersReducedMotion } from "@/shared/hooks/usePrefersReducedMotion"

export type SetupOrbPhase = "waiting" | "checking" | "stalled" | "solved"

// Searching watches for a request; working's tilted orbits convey effort.
// Stalled keeps the search alive at roughly half speed; solving is the payoff.
const ORB_BY_PHASE: Record<SetupOrbPhase, { state: OrbState; speed: number }> =
  {
    waiting: { state: "searching", speed: 0.4 },
    checking: { state: "working", speed: 0.9 },
    stalled: { state: "searching", speed: 0.22 },
    solved: { state: "solving", speed: 0.7 },
  }

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
      if (top.phase === phase) return prefersReducedMotion ? [top] : current
      const next = { id: top.id + 1, phase }
      return prefersReducedMotion ? [next] : [top, next]
    })
  }, [phase, prefersReducedMotion])

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
                    ? "animate-setup-orb-in motion-reduce:animate-none"
                    : "animate-setup-orb-out motion-reduce:animate-none"
              }`}
              onAnimationEnd={(event) => {
                if (
                  event.target !== event.currentTarget ||
                  index === layers.length - 1
                )
                  return
                setLayers((current) =>
                  current.filter(({ id }) => id !== layer.id),
                )
              }}
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
