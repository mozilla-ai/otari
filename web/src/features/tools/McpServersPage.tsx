import { PageIntro } from "@/design-system/layout/PageIntro"
import { WorkspaceMcpServersCard } from "@/features/tools/WorkspaceMcpServersCard"

/**
 * The selected workspace's MCP servers: a list of servers rather than one
 * service's settings, so not a `ToolsGuardrailsPage` view.
 */
export function McpServersPage() {
  return (
    <div className="flex flex-col gap-6">
      <PageIntro title="MCP servers" />
      <WorkspaceMcpServersCard />
    </div>
  )
}
