import { useEffect, useRef } from "react"

/**
 * A 6px square. The page's one status mark, in every place it appears.
 *
 * `pulse` sends a ring of the dot's own color out from it every few seconds,
 * like a radar check, for a dot that states a record's live status beside its
 * word ("Active", "Error"). Never on a dot used as a bullet, a tag or a legend.
 */
export function Dot({
  className,
  pulse = false,
}: {
  className: string
  pulse?: boolean
}) {
  const ref = useRef<HTMLSpanElement>(null)
  // A CSS animation starts when its element mounts, so a row that arrives
  // later would ping out of phase with the rest. Starting each ring at the
  // timeline's origin puts every dot on the page on the same beat.
  useEffect(() => {
    if (!pulse) return
    for (const ring of ref.current?.getAnimations?.({ subtree: true }) ?? []) {
      ring.startTime = 0
    }
  }, [pulse])
  return (
    <span
      ref={ref}
      aria-hidden
      className={`h-1.5 w-1.5 shrink-0 ${pulse ? "otari-dot-pulse " : ""}${className}`}
    />
  )
}
