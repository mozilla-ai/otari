import { useState } from "react"

import type { WorkspaceMcpServer } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { isLiteralToolName } from "@/features/tools/mcpAccess"
import { ToolChecklist } from "@/features/tools/ToolChecklist"
import { useUpdateWorkspaceMcpServer } from "@/shared/api/tools"

/**
 * Which of a server's tools count as read, the set a "Read tools only" rule
 * allows. The labels are the manager's: the server's own `readOnlyHint` is
 * named as a suggestion and never ticked for them, since a server can call any
 * tool read-only and one click must not turn its word into a label.
 *
 * The caller keys this on the stored labels, so a save re-seeds it and closes
 * the editor.
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
  const hinted = server.health?.read_only_hinted_tools ?? []
  const tools = (server.allowed_tools ?? server.health?.tools ?? []).filter(
    isLiteralToolName,
  )
  const [isEditing, setEditing] = useState(false)
  const [picked, setPicked] = useState<string[]>([...stored])
  const isChanged = picked.join() !== stored.join()

  return (
    <SettingsGroup
      title="Read tools"
      isBounded
      description="The tools a rule set to Read tools only allows. A tool not labeled here counts as write."
      action={
        canManage && !isEditing ? (
          <Button onPress={() => setEditing(true)} aria-label="Edit read tools">
            Edit
          </Button>
        ) : null
      }
    >
      {isEditing ? (
        <div className="flex flex-col gap-3">
          {stored.length === 0 && hinted.length > 0 ? (
            <p className="text-caption break-words">
              The server marks {hinted.join(", ")} as read-only. Its word is not
              a label: tick each one you trust.
            </p>
          ) : null}
          <ToolChecklist
            label="Read tools"
            tools={tools}
            selected={picked}
            onChange={setPicked}
            inputLabel="Tool name"
          />
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
              isDisabled={!isChanged}
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
      ) : (
        <p className="font-mono text-sm break-words">
          {stored.length > 0 ? (
            stored.join(", ")
          ) : (
            <span className="font-sans text-muted">
              None labeled, so a Read tools only rule allows nothing here.
            </span>
          )}
        </p>
      )}
    </SettingsGroup>
  )
}
