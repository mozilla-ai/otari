import { Link } from "@tanstack/react-router"
import type { ReactNode } from "react"
import { PageIntro } from "@/design-system/layout/PageIntro"

export type ToolsView = "/tools" | "/tools/web-search" | "/tools/code-execution"

/**
 * Tools as one page: one title over the list of tools, and over each tool's
 * settings, which open from the list. A settings view takes `back`, a link to
 * the list, since nothing else on the page leads there.
 */
export function ToolsFrame({
  back = false,
  children,
}: {
  back?: boolean
  children: ReactNode
}) {
  return (
    <div className="flex min-w-0 flex-col gap-6">
      <PageIntro
        title="Tools"
        detail={
          back ? (
            <Link to="/tools" className="text-link hover:text-link-hover">
              ← Tools
            </Link>
          ) : undefined
        }
      />
      {children}
    </div>
  )
}
