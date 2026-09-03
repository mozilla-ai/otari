import type { ReactNode } from "react"

/**
 * The line under a field, and the space it takes whether or not it speaks.
 *
 * A field's description and its error are the same role at the same size,
 * separated by color alone, so they occupy one line and share one reserve. The
 * reserve is what stops a form jumping when a message appears: one caption
 * line, which the parent's own 4px gap takes to the 23px a message actually
 * costs. It reads the role's line height rather than repeating 19px, so a
 * retune of the caption moves the reserve with it.
 *
 * The reserve attaches to the caption ROLE, not to supporting text in general.
 * A region's description never reserves a line, and that is not an opt-out: a
 * heading's paragraph has no error to make room for, so the question does not
 * arise for it.
 *
 * `reserve={false}` is for a field in a table row or a toolbar. Those never
 * speak, and holding a line open under each one would put a band of empty
 * space through every row of a table.
 */
export function FieldMessages({
  children,
  reserve = true,
}: {
  children: ReactNode
  reserve?: boolean
}) {
  // The role goes on the wrapper, and that is not a style preference. HeroUI
  // merges a component's className through tailwind-merge, which cannot tell a
  // custom `text-caption` from a text COLOR, so `text-caption text-muted` on a
  // `Description` loses the caption and keeps only the color: it compiles,
  // ships, and renders at HeroUI's own 12px. Setting the role here and leaving
  // the child nothing but its color keeps them out of the same merge group.
  return (
    <div
      className={`text-caption ${reserve ? "min-h-[var(--text-caption-step--line-height)]" : ""}`}
    >
      {children}
    </div>
  )
}
