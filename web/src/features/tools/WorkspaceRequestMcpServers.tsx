import type { Workspace } from "@/client"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Toggle } from "@/design-system/forms/Toggle"
import { useSetWorkspaceRequestMcpServers } from "@/shared/api/workspaces"

const LABEL = "Let requests name their own MCP servers"

/**
 * Whether the workspace's keys may name an MCP server by URL in a request,
 * beside the stored servers above. Writes when flipped.
 *
 * The explanation stays on the page rather than in a tooltip, because it is
 * what makes the switch matter: while it is on, a key can reach any server it
 * knows the URL of, whatever the access lists say.
 */
export function WorkspaceRequestMcpServers({
  workspace,
}: {
  workspace: Workspace
}) {
  const set = useSetWorkspaceRequestMcpServers()
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="text-body">{LABEL}</p>
          <p className="mt-1 max-w-prose text-caption">
            On, a request may list servers by URL with its own token, which the
            access lists here do not cover. Turn it off to make the access lists
            airtight: requests can then reach only the stored servers.
          </p>
        </div>
        <Toggle
          isSelected={workspace.request_mcp_servers_allowed}
          onChange={(allowed) => set.mutate({ id: workspace.id, allowed })}
          label={LABEL}
          isDisabled={set.isPending}
        />
      </div>
      <ErrorBanner error={set.error} />
    </div>
  )
}
