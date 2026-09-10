import type { ReactNode } from "react"

/**
 * A row of filter controls above a table.
 *
 * It exists to name a *place*, which is what keeps field height from being a
 * per-site choice: a field is 36px everywhere, and the controls inside one of
 * these are 32px, because a filter sits on the table's own header row and a
 * 36px control outgrows it. A call site says "this row is a toolbar"; it never
 * says "this control is 32px". Below `md` the place raises everything to 44px,
 * so the dense size is a desktop size the phone layout takes back off.
 *
 * `.otari-toolbar` is that place, and what it does is **declare two custom
 * properties**, `--field-height` and `--field-padding-block`; the rule that
 * reads them is on `.input` and `.select__trigger` in globals.css, because it
 * has to reach inside HeroUI's own DOM to find the select trigger.
 *
 * The distinction is worth keeping. The place used to spell
 * `.otari-toolbar .input { height: 32px }`, which is a descendant selector at
 * (0,2,0): unconditional, and invisible from here. A variable inherits rather
 * than winning, so this row still sets the density for everything inside it,
 * and a subtree that legitimately wants the form height can reset the property
 * on itself. globals.css carries the longer version of that argument.
 *
 * It also drops a ghost button's edge, which is the place's other job: a row of
 * edged ghosts reads as a grid of boxes. See design/actions.md, "Places".
 */
export function Toolbar({
  className = "",
  children,
}: {
  className?: string
  children: ReactNode
}) {
  return (
    <div
      className={`otari-toolbar flex flex-wrap items-center gap-2 ${className}`}
    >
      {children}
    </div>
  )
}
