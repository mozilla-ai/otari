import { Link } from "@tanstack/react-router"
import type { ReactNode } from "react"
import { PageIntro } from "@/design-system/layout/PageIntro"

export type ToolsView = "/tools" | "/tools/web-search" | "/tools/code-execution"

/**
 * The page a tool's list and each tool's settings share. The list is titled
 * Tools; a tool's settings open from it, are titled by the tool and take
 * `back`, a link to the list placed above the title as every detail page's
 * back link is.
 */
export function ToolsFrame({
  title = "Tools",
  back = false,
  children,
}: {
  title?: string
  back?: boolean
  children: ReactNode
}) {
  return (
    <div className="flex min-w-0 flex-col gap-6">
      {back ? (
        <nav aria-label="Back" className="text-caption">
          <Link to="/tools" className="text-link hover:text-link-hover">
            ← Tools
          </Link>
        </nav>
      ) : null}
      <PageIntro title={title} />
      {children}
    </div>
  )
}
