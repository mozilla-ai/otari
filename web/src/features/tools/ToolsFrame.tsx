import type { ReactNode } from "react"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { ViewTabs } from "@/shared/components/ViewTabs"

export type ToolsView =
  | "/tools"
  | "/tools/web-search"
  | "/tools/code-execution"
  | "/tools/mcp-servers"

const TABS: { to: ToolsView; label: string }[] = [
  { to: "/tools", label: "Overview" },
  { to: "/tools/web-search", label: "Web search" },
  { to: "/tools/code-execution", label: "Code execution" },
  { to: "/tools/mcp-servers", label: "MCP servers" },
]

/**
 * Tools as one page: one title, and every tool a view under it. The rail has
 * one Tools row, and this is where the tools are told apart, so a view does
 * not title itself: the tab above it names it.
 */
export function ToolsFrame({
  current,
  children,
}: {
  current: ToolsView
  children: ReactNode
}) {
  return (
    <div className="flex min-w-0 flex-col gap-6">
      <div className="flex flex-col">
        <PageIntro title="Tools" />
        <ViewTabs label="Tools views" tabs={TABS} current={current} />
      </div>
      {children}
    </div>
  )
}
