import type { ReactNode } from "react"
import { Section } from "./Section"

/**
 * A settings list: a heading between rules, then its rows on the page ground
 * divided by row separators.
 *
 * Shared so the rank of a group is decided once: spelled by hand, groups of the
 * same rank on the same page drift to different heading sizes.
 *
 * Two bands rather than one, which is what puts the heading *between* rules
 * rather than above them: the first carries the rule over the heading, the
 * second the rule under it and the rule closing the last row.
 */
export function SettingsGroup({
  title,
  count,
  description,
  children,
}: {
  /**
   * Omitted where the page's own title already names the group, which happens
   * when a page is a filtered view of one service. The rows band keeps its
   * rules either way; what goes is the heading band above it.
   */
  title?: string
  /** Shown beside the title where a group's size is worth knowing up front. */
  count?: number
  /**
   * What the group is, under its heading and inside the same band. Capped to a
   * readable measure, because a band spans the page and a sentence should not.
   * A node rather than a string so a caller can put a link, or the group's own
   * error banner, in the same place.
   */
  description?: ReactNode
  children: ReactNode
}) {
  return (
    <>
      {title === undefined && description === undefined ? null : (
        <Section
          className="border-t border-border pt-6 pb-3"
          contentClassName="flex flex-col gap-2"
        >
          {title === undefined ? null : (
            <h2 className="text-title">
              {title}
              {count === undefined ? null : (
                <span className="font-normal text-subtle"> ({count})</span>
              )}
            </h2>
          )}
          {description ? (
            <div className="max-w-prose text-sm text-muted">{description}</div>
          ) : null}
        </Section>
      )}
      {/* `border-subtle` between the rows, `border` around the group. The two
          tiers are the structure: a section rule divides the page, a row
          separator divides repeated things inside one section, and using the
          section tier for both flattens the hierarchy into one weight. This is
          the third place that mis-assignment has been found, so it is fixed
          here rather than at a call site: every settings list in the app is
          this component now, and none of them names a tier. */}
      <Section
        className="border-y border-border"
        contentClassName="flex flex-col divide-y divide-border-subtle"
      >
        {children}
      </Section>
    </>
  )
}
