import { WorkspaceMcpServersCard } from "@/features/tools/WorkspaceMcpServersCard"

/**
 * The selected workspace's MCP servers, as a destination of their own.
 *
 * The one Tools child that is not a filtered view of `ToolsGuardrailsPage`;
 * `web/AGENTS.md` says why. The card is the whole page here, so it renders the
 * page's opening itself: the register control belongs in that heading row, and
 * a page cannot hand a control to a card mounted under it.
 */
export function McpServersPage() {
  return (
    <div className="flex flex-col gap-6">
      <WorkspaceMcpServersCard variant="page" />
    </div>
  )
}
