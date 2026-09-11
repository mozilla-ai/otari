import type { ReactNode } from "react"

/**
 * The lane every row's control sits in, from `md` up; below it the control
 * stacks full width under its label.
 *
 * Exported so a loading frame standing in for a row takes the same width, which
 * is what keeps settings arriving from moving the page.
 *
 * 17.5rem is what the widest cluster on a settings page needs, a field beside
 * its own button, so widening the narrow controls to the lane is what changed
 * rather than narrowing the wide ones. It steps down at `md`, where the shell
 * still draws its 16.5rem sidebar beside a full-bleed row: a lane that wide
 * there leaves a label like "Intercept provider web search" about 120px to wrap
 * in. Both steps are one width across the whole group, so the column's edges are
 * straight at either.
 */
export const CONTROL_LANE = "w-full shrink-0 md:w-[13.75rem] lg:w-[17.5rem]"

/**
 * What a setting is on the left, the control that changes it on the right.
 *
 * Shared because a settings page is almost entirely this shape, and spelled by
 * hand the label size, the key caption and the control lane drift between
 * groups on the same page. The row draws no rules of its own: `SettingsGroup`
 * divides its children, and a border here would give every seam two lines.
 *
 * **The lane is one width down the page and the control fills it**, which is
 * the fixed-width-slot rule `design/layout.md` states for any repeated row.
 * Sized per control it was not a lane at all: a URL field, a select and a
 * two-digit number came out 220px, 174px and 88px, so the column had a
 * different left edge on every row. A field takes `w-full` (or `min-w-0 flex-1`
 * beside a trailing button), a `FilterSelect` takes `fullWidth`, and a number
 * keeps `text-right tabular-nums` so its digits still read against the lane's
 * edge.
 *
 * Only an error may add a line to a row, which is what keeps the help text from
 * rewrapping under the cursor while a value is being changed.
 */
export function SettingRow({
  label,
  labelId,
  configKey,
  help,
  control,
  controlId,
  note,
  nested = false,
  error,
  errorId,
}: {
  label: ReactNode
  /**
   * Ids the row's own control points at with `aria-labelledby`: this one on the
   * label, and `<labelId>-key` on the config key beside it. "Backend URL" alone
   * is not unique on a page that configures three services; "Backend URL
   * web_search_url" is, and it is what the row actually reads.
   */
  labelId?: string
  /** The config key this row writes, as a mono caption beside the label. */
  configKey?: string
  /**
   * The control's own `id`, which makes the label a real `<label for>`: on a
   * page that is nothing but labelled rows, and on a phone where the label sits
   * directly above its stacked field, the label is a target people press.
   * `aria-labelledby` still decides the accessible name where both are set.
   */
  controlId?: string
  help?: ReactNode
  control: ReactNode
  /** An outcome the row reports back (a reachability result), under the help. */
  note?: ReactNode
  /**
   * Indent the row, which is how a row says it belongs to the one above it
   * rather than being its sibling. Used by a disclosure's panel, whose rows are
   * children of the row that opened them.
   */
  nested?: boolean
  /** A rejected save, kept beside the value that caused it. */
  error?: string
  /** Ties the message to the control through `aria-describedby`. */
  errorId?: string
}) {
  return (
    <div
      className={`flex min-h-11 flex-col gap-2.5 py-3 pr-4 md:flex-row md:items-center md:gap-6 ${
        nested ? "pl-8" : "pl-4"
      }`}
    >
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <div className="flex flex-wrap items-baseline gap-x-2">
          <label id={labelId} htmlFor={controlId} className="text-emphasis">
            {label}
          </label>
          {configKey ? (
            <code
              id={labelId && `${labelId}-key`}
              className="text-mono-micro text-subtle"
            >
              {configKey}
            </code>
          ) : null}
        </div>
        {help ? <p className="text-caption text-subtle">{help}</p> : null}
        {note}
        {error ? (
          <p id={errorId} className="text-caption text-danger">
            {error}
          </p>
        ) : null}
      </div>
      <div className={CONTROL_LANE}>{control}</div>
    </div>
  )
}
