import type { HTMLAttributes, ReactNode } from "react"

/**
 * A band of the page: rules that run the full width of the scroll area, with
 * the content inside them still in the centered column.
 *
 * It takes two elements, because one element cannot be both full-width and
 * centered. `.otari-bleed` escapes `<main>` (see globals.css for why it is
 * container units and not `100vw`); the inner element restores the column.
 *
 * `className` styles the band: its rules, its vertical padding, its own layout
 * if the content is a single row. `contentClassName` styles the column inside.
 *
 * `bleed={false}` for a band nested inside a column rather than sitting
 * directly in the scroll area. The escape is `100cqw` against `<main>`, so a
 * bleeding band nested in a narrow cell overflows the page instead of stopping
 * at its column. A nested band rules its own container and aligns to its edges.
 */
export function Section({
  className = "",
  contentClassName = "",
  bleed = true,
  children,
  ...rest
}: {
  className?: string
  contentClassName?: string
  bleed?: boolean
  children: ReactNode
} & Omit<HTMLAttributes<HTMLElement>, "className" | "children">) {
  return (
    <section
      className={bleed ? `otari-bleed ${className}` : className}
      {...rest}
    >
      <div
        className={
          bleed
            ? `mx-auto w-full max-w-[112.5rem] px-4 md:px-6 ${contentClassName}`
            : `w-full ${contentClassName}`
        }
      >
        {children}
      </div>
    </section>
  )
}
