import { WorkspaceMcpServersCard } from "@/features/tools/WorkspaceMcpServersCard"
import { PageHeader } from "@/shared/components/ui"

/**
 * The selected workspace's MCP servers, as a destination of their own.
 *
 * The one Tools child that is not a filtered view of `ToolsGuardrailsPage`;
 * `web/AGENTS.md` says why. The same card renders at the foot of `/tools`, with
 * its heading suppressed here because the title above already carries it.
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
