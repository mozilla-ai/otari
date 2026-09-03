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
  // custom `text-caption` from a text COLOR, so putting the role and a color
  // together on a `Description` loses the caption and keeps only the color: it
  // compiles, ships, and renders at HeroUI's own 12px. Setting the role here
  // and leaving the child nothing but its color keeps them out of the same
  // merge group.
  //
  // Described rather than spelled out, for the reason `PageIntro`'s docstring
  // gives about the arbitrary-size rule: `foundation.test.ts` matches a role
  // beside its own ink against raw file contents and strips block comments
  // only, so a line comment quoting that pairing keeps this file on the
  // offender list with nothing visibly wrong in it.
  return (
    <div
      className={`text-caption ${reserve ? "min-h-[var(--text-caption-step--line-height)]" : ""}`}
    >
      {children}
    </div>
  )
}

/**
 * The label and description rungs of a control that is not a HeroUI field.
 *
 * HeroUI's `Label` and `Description` cover a `TextField`, and the migrated
 * paragraphs cover text beside a heading, but a third construct kept being
 * hand-written for everything else: a segmented group, a multi-select, a
 * scope picker. Written out each time, it was a `text-sm` label over a
 * `text-xs` description, which is how the size drifted below the caption in
 * eight files at once while both of the other two constructs were being
 * corrected. This exists so the next one cannot be born at the wrong size.
 *
 * The label rung is measured against HeroUI's own: both render 14px/20px at
 * weight 400 with the same tracking, so this matches the field label rather
 * than approximating it. (The 400 is deliberate upstream in the type scale;
 * `font-medium` is written at both and resolves to 400 there.)
 *
 * The description rung goes through `FieldMessages`, so it takes the caption
 * role and the reserved line on the same terms a field's does. With no
 * description there is no reserve: an empty line under a control that will
 * never say anything is space held for nothing.
 */
export function ControlField({
  label,
  description,
  reserve,
  children,
}: {
  label: ReactNode
  description?: ReactNode
  reserve?: boolean
  children?: ReactNode
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-sm font-medium text-foreground">{label}</span>
      {description ? (
        <FieldMessages reserve={reserve}>
          <p className="text-muted">{description}</p>
        </FieldMessages>
      ) : null}
      {children}
    </div>
  )
}
