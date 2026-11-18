import type { ReactNode } from "react"
import { Section } from "@/design-system/layout/Section"

/**
 * A titled band: the heading row a page section opens with, ruled off from the
 * next. The one section shape of the People, Rules and Home pages.
 */
export function Band({
  title,
  count,
  action,
  description,
  children,
  isLast = false,
}: {
  title: string
  count?: number
  action?: ReactNode
  description?: ReactNode
  children: ReactNode
  isLast?: boolean
}) {
  return (
    <Section
      shouldBleed
      className={isLast ? "" : "border-b border-border"}
      contentClassName="flex flex-col gap-3 pb-6"
    >
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="flex min-w-0 flex-col gap-1">
          <h2 className="text-title">
            {title}
            {count !== undefined ? (
              <span className="ml-2 font-normal text-muted tabular-nums">
                {count}
              </span>
            ) : null}
          </h2>
          {description ? <p className="text-caption">{description}</p> : null}
        </div>
        {action ? <div className="flex shrink-0 gap-2">{action}</div> : null}
      </div>
      {children}
    </Section>
  )
}
