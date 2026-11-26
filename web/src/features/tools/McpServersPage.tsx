import { WorkspaceMcpServersCard } from "@/features/tools/WorkspaceMcpServersCard"

/**
 * The selected workspace's MCP servers, a view of Tools: a list of servers
 * rather than one service's settings, so not a `ToolsGuardrailsPage` view.
 */
export function McpServersPage() {
  return (
    <div className="flex flex-col gap-6">
      <WorkspaceMcpServersCard />
    </div>
  )
}
