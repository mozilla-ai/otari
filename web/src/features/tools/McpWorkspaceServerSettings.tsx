import { useState } from "react"

import type { WorkspaceMcpServer } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import {
  SettingsBlock,
  SettingsGroup,
} from "@/design-system/layout/SettingsGroup"
import { McpAvailabilitySelect } from "@/features/tools/McpAvailabilitySelect"
import { McpGrantDialog } from "@/features/tools/McpGrantDialog"
import { isLiteralToolName, ruleSummary } from "@/features/tools/mcpAccess"
import { checkedTools } from "@/features/tools/mcpServerHealth"
import {
  availabilityOverride,
  hasWorkspaceSettings,
  isSamePolicy,
  organizationLimitLabel,
  organizationOffReason,
  workspaceAvailability,
} from "@/features/tools/mcpWorkspaceView"
import {
  useClearWorkspaceMcpServerOverride,
  useSetWorkspaceMcpServerOverride,
  useWorkspaceMcpServer,
  useWorkspaceMcpServerOverridePending,
} from "@/shared/api/tools"

/**
 * A workspace's own settings on its organization's server, for the
 * workspace's owners and admins: whether it is on here and attached to every
 * request (Availability), and what every key may call (the default policy).
 * The organization's limit for the workspace sits above them, since it caps
 * whatever is set here. "Use organization defaults" takes the workspace's own
 * settings off, and the limit stays either way.
 *
 * `server` is the row as read, its top-level fields the organization's.
 */
export function McpWorkspaceServerSettings({
  workspaceId,
  server,
}: {
  workspaceId: string
  server: WorkspaceMcpServer
}) {
  const set = useSetWorkspaceMcpServerOverride()
  const clear = useClearWorkspaceMcpServerOverride()
  const [isEditingDefault, setEditingDefault] = useState(false)
  // Bumped on each open and used as the dialog's key, so a draft never
  // survives into the next one.
  const [openCount, setOpenCount] = useState(0)
  const own = server.in_workspace
  const ids = { workspaceId, serverId: server.id }
  const limit = organizationLimitLabel(own?.organization_tools)
  const offReason = organizationOffReason(server)
  const defaultTools = own?.everyone_tools ?? server.everyone_tools
  // No control starts while a write here or in the access rules below is in
  // flight, or while the server is read again after one. Resetting a write in
  // flight would also lose its refusal.
  const isWriting = useWorkspaceMcpServerOverridePending(workspaceId, server.id)
  // After a failed write the server is read again; if that read fails too,
  // what is shown may not be what is stored, so nothing is changed from it.
  const read = useWorkspaceMcpServer(workspaceId, server.id)
  const isUnconfirmed = read.isRefetchError
  const isHeld = isWriting || isUnconfirmed
  const picker = checkedTools(server, { withinAllowList: true })
  const tools = picker.isKnown
    ? picker.tools
    : (server.allowed_tools ?? []).filter(isLiteralToolName)

  const write = (body: Parameters<typeof set.mutate>[0]["body"]) => {
    if (isHeld) return
    clear.reset()
    set.mutate({ ...ids, body })
  }

  return (
    <SettingsGroup
      title="In this workspace"
      description="Your organization manages this server. You can change how this workspace uses it."
      isBounded
    >
      <SettingsBlock>
        <div className="flex flex-col gap-4">
          {limit ? (
            <p className="text-body">
              <span className="font-medium">Organization limit:</span> {limit}.{" "}
              <span className="text-muted">
                The most this workspace can reach. Its rules can narrow it,
                never widen it.
              </span>
            </p>
          ) : null}
          {offReason ? <InfoBanner>{offReason}</InfoBanner> : null}
          {isUnconfirmed ? (
            <InfoBanner>
              <span className="flex flex-wrap items-center gap-3">
                These settings could not be read again, so they may not be what
                is saved.
                <Button
                  onPress={() => void read.refetch()}
                  className="min-h-11 md:min-h-0"
                >
                  Refresh
                </Button>
              </span>
            </InfoBanner>
          ) : null}
          <McpAvailabilitySelect
            value={workspaceAvailability(server)}
            onChange={(next) => write(availabilityOverride(next))}
            isDisabled={isHeld}
          />
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-body">
              <span className="font-medium">Default policy:</span>{" "}
              {ruleSummary(defaultTools)}
              {own?.everyone_tools === null ? (
                <span className="text-muted">
                  , as your organization set it
                </span>
              ) : null}
            </p>
            <Button
              onPress={() => {
                if (isHeld) return
                set.reset()
                setOpenCount((count) => count + 1)
                setEditingDefault(true)
              }}
              isDisabled={isHeld}
              aria-label="Edit the default policy for this workspace"
              className="min-h-11 shrink-0 md:min-h-0"
            >
              Edit
            </Button>
          </div>
          {/* The dialog shows its own write's refusal while it is open. */}
          <ErrorBanner
            error={isEditingDefault ? clear.error : (set.error ?? clear.error)}
          />
          <div>
            <Button
              onPress={() => {
                if (isHeld) return
                set.reset()
                clear.mutate(ids)
              }}
              isPending={clear.isPending}
              isDisabled={!hasWorkspaceSettings(server) || isHeld}
              className="min-h-11 md:min-h-0"
            >
              Use organization defaults
            </Button>
          </div>
        </div>
      </SettingsBlock>
      {isEditingDefault ? (
        <McpGrantDialog
          key={openCount}
          isOpen
          onOpenChange={(open) => {
            if (!open) setEditingDefault(false)
          }}
          users={[]}
          editing={{ name: "the default policy", tools: defaultTools }}
          tools={tools}
          descriptions={picker.descriptions}
          isKnown={picker.isKnown}
          readTools={server.read_tools}
          isPending={set.isPending}
          error={set.error}
          onSubmit={(draft) => {
            if (set.isPending) return
            // Unchanged: saving it would store the inherited policy as the
            // workspace's own, which would stop following the organization's.
            if (isSamePolicy(draft.tools, defaultTools)) {
              setEditingDefault(false)
              return
            }
            set.mutate(
              { ...ids, body: { everyone_tools: draft.tools } },
              { onSuccess: () => setEditingDefault(false) },
            )
          }}
        />
      ) : null}
    </SettingsGroup>
  )
}
