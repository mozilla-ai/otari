import { WorkspaceMcpServersCard } from "@/features/tools/WorkspaceMcpServersCard"
import { PageHeader } from "@/shared/components/ui"

/**
 * The selected workspace's MCP servers, as a destination of their own.
 *
 * A sibling of the Tools group's other children, and the one that is not a view
 * of `ToolsGuardrailsPage`: the other two narrow a deployment-wide backend this
 * gateway configures, while MCP has no deployment-wide server list, so these
 * rows are the configuration rather than a narrowing of one. The same card
 * renders at the foot of `/tools`, which is the combined page; the heading is
 * suppressed here because the title below already says it.
 */
export function McpServersPage() {
  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="MCP servers"
        description="MCP endpoints a workspace's requests can reach by naming their ids, without carrying a URL or a bearer token of their own. Each is checked for SSRF safety when it is stored and again when a request uses it, and its token is encrypted at rest."
      />
      <WorkspaceMcpServersCard showHeading={false} />
    </div>
  )
}
