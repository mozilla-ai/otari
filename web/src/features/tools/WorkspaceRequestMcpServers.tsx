import type { Workspace } from "@/client"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Toggle } from "@/design-system/forms/Toggle"
import { McpInfoHint } from "@/features/tools/McpInfoHint"
import { useSetWorkspaceRequestMcpServers } from "@/shared/api/workspaces"

const LABEL = "Allow servers outside this list"

/**
 * Whether a request made with one of this workspace's API keys may carry its
 * own MCP server (URL and token) in its body, beside the stored servers. Writes
 * when flipped.
 *
 * One line under the page's opening, because it is not a property of any server
 * in the list: it decides whether servers that are *not* listed can be reached
 * at all. The label says what the switch allows, and the icon beside it says
 * what each position does.
 */
export function WorkspaceRequestMcpServers({
  workspace,
}: {
  workspace: Workspace
}) {
  const set = useSetWorkspaceRequestMcpServers()
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <Toggle
          isSelected={workspace.request_mcp_servers_allowed}
          onChange={(allowed) => set.mutate({ id: workspace.id, allowed })}
          label={LABEL}
          isDisabled={set.isPending}
        />
        <span className="text-sm">{LABEL}</span>
        <McpInfoHint label="Explain this switch">
          <span>
            On: an API request may carry the URL and token of any MCP server in
            its body. The access rules set here do not apply to those servers.
          </span>
          <span>
            Off: requests can only use the servers in this list. Members connect
            the same servers in the dashboard either way.
          </span>
        </McpInfoHint>
      </div>
      <ErrorBanner error={set.error} />
    </div>
  )
}
