import type { ReactNode } from "react"

/**
 * A row of filter controls above a table.
 *
 * It exists to name a *place*, which is what keeps field height from being a
 * per-site choice: a field is 40px everywhere, and the controls inside one of
 * these are 38px, because a filter sits on the table's own header row and a
 * 40px control outgrows it. A call site says "this row is a toolbar"; it never
 * says "this control is 38px".
 *
 * The height itself is in globals.css, on `.otari-toolbar`, since it has to
 * reach inside HeroUI's own DOM to find the select trigger.
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
