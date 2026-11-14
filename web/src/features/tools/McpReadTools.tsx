import { useState } from "react"

import type { WorkspaceMcpServer } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import {
  SettingsBlock,
  SettingsGroup,
} from "@/design-system/layout/SettingsGroup"
import { McpNotice } from "@/features/tools/McpNotice"
import { checkedTools } from "@/features/tools/mcpServerHealth"
import { ToolChecklist } from "@/features/tools/ToolChecklist"
import { useMcpServerCheck } from "@/features/tools/useMcpServerCheck"
import { useUpdateWorkspaceMcpServer } from "@/shared/api/tools"

/**
 * Which of a server's tools count as read, the set a "Read tools only" rule
 * allows. Until a manager edits them, the labels are the tools the server
 * itself marks read-only, so a read rule works as soon as the server's tools
 * are known. The first save makes them the manager's, and no later check
 * changes them.
 *
 * The caller keys this on the labels and where they come from, so a save
 * re-seeds it and closes the editor.
 */
export function McpReadTools({
  workspaceId,
  server,
  canManage,
}: {
  workspaceId: string
  server: WorkspaceMcpServer
  canManage: boolean
}) {
  const update = useUpdateWorkspaceMcpServer()
  const stored = server.read_tools
  const isFromServer = server.read_tools_from_server
  const picker = checkedTools(server, { withinAllowList: true })
  const check = useMcpServerCheck({
    workspaceId,
    returnTo: `/tools/mcp-servers/${server.id}`,
  })
  const [isEditing, setEditing] = useState(false)
  const [picked, setPicked] = useState<string[]>([...stored])
  const isChanged = picked.join() !== stored.join()

  const source = isFromServer
    ? stored.length > 0
      ? "These are the tools the server marks read-only. Edit them to set your own."
      : picker.isKnown
        ? "The server marks none of its tools read-only. Edit to pick them yourself."
        : "Filled in from the tools the server marks read-only, once its tools are known."
    : ""

  return (
    <SettingsGroup
      title="Read tools"
      count={stored.length > 0 ? stored.length : undefined}
      isBounded
      description="Tools that only read. A rule set to Read tools only allows just these."
      action={
        canManage && !isEditing ? (
          <Button onPress={() => setEditing(true)} aria-label="Edit read tools">
            Edit
          </Button>
        ) : null
      }
    >
      <SettingsBlock>
        {source ? <p className="text-caption">{source}</p> : null}
        {isEditing ? (
          <div className="flex flex-col gap-3">
            <ToolChecklist
              label="Read tools"
              tools={picker.tools}
              descriptions={picker.descriptions}
              isKnown={picker.isKnown}
              selected={picked}
              onChange={setPicked}
              allowPatterns={false}
              onCheck={() => check.run(server)}
              isChecking={check.isRunning(server.id)}
            />
            {check.error ? (
              <McpNotice message={check.error} onDismiss={check.dismissError} />
            ) : null}
            <ErrorBanner error={update.error} />
            <div className="flex flex-wrap gap-2">
              <Button
                variant="primary"
                onPress={() =>
                  update.mutate({
                    workspaceId,
                    serverId: server.id,
                    body: { read_tools: picked },
                  })
                }
                isPending={update.isPending}
                // Saving the server's own labels unchanged still makes them the manager's.
                isDisabled={!isChanged && !isFromServer}
              >
                Save read tools
              </Button>
              <Button
                onPress={() => {
                  setEditing(false)
                  setPicked([...stored])
                  update.reset()
                }}
              >
                Cancel
              </Button>
            </div>
          </div>
        ) : stored.length > 0 ? (
          <p className="font-mono text-sm break-words">{stored.join(", ")}</p>
        ) : isFromServer ? null : (
          <p className="text-sm text-muted">None.</p>
        )}
      </SettingsBlock>
    </SettingsGroup>
  )
}
