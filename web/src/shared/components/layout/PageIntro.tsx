import type { ReactNode } from "react"

/**
 * A page's opening: its title, the paragraph under it, and the one action that
 * belongs beside rather than below them.
 *
 * Shared so the type of a page title is decided once rather than respelled on
 * every page. The title is `text-display`, the scale's 28/34 semibold step; the
 * weight axis has only 550 and 600, so an arbitrary utility asking for 600
 * lands on 550 instead. Do not quote an arbitrary size spelling anywhere in
 * this file, comments included: `foundation.test.ts` matches that spelling
 * against raw file contents without stripping comments first, so a quoted
 * example keeps the file on the offender list with nothing wrong to find.
 *
 * `pb-5` rather than a gap on the parent, because a page is a stack of bands
 * that set their own rules and spacing, and a column gap would add air above
 * the first rule as well.
 */
export function PageIntro({
  title,
  action,
  descriptionClassName = "",
  children,
}: {
  title: string
  action?: ReactNode
  /**
   * Overrides the description's measure. One caller uses it: the guide, whose
   * own prose is 560px, so the paragraph introducing it should not be the
   * widest line on a page about measure.
   */
  descriptionClassName?: string
  children?: ReactNode
}) {
  return (
    <header className="flex flex-col gap-4 pb-5 sm:flex-row sm:items-start sm:justify-between">
      <div className="max-w-[38.75rem]">
        <h1 className="text-display">{title}</h1>
        {children ? (
          <p className={`mt-1 text-sm text-muted ${descriptionClassName}`}>
            {children}
          </p>
        ) : null}
      </div>
      {action ? <div className="shrink-0">{action}</div> : null}
    </header>
  )
}
