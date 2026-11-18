import { useState } from "react"

import type { WorkspaceMcpServer } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Select } from "@/design-system/forms/Select"
import {
  SettingsBlock,
  SettingsGroup,
} from "@/design-system/layout/SettingsGroup"
import { McpGrantDialog } from "@/features/tools/McpGrantDialog"
import { McpInfoHint } from "@/features/tools/McpInfoHint"
import { isLiteralToolName, ruleSummary } from "@/features/tools/mcpAccess"
import {
  AVAILABILITY,
  AVAILABILITY_OPTIONS,
  AVAILABILITY_ORDER,
  type McpAvailability,
} from "@/features/tools/mcpAvailability"
import { checkedTools } from "@/features/tools/mcpServerHealth"
import {
  availabilityOverride,
  hasWorkspaceSettings,
  organizationLimitLabel,
  organizationOffReason,
  overrideBody,
  workspaceAvailability,
} from "@/features/tools/mcpWorkspaceView"
import {
  useClearWorkspaceMcpServerOverride,
  useSetWorkspaceMcpServerOverride,
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
  // Any write here, or the access rules' default policy below, sends every
  // setting from the row as read, so none starts while another is in flight.
  // Resetting a write in flight would also lose its refusal.
  const isWriting = useWorkspaceMcpServerOverridePending(workspaceId, server.id)
  const picker = checkedTools(server, { withinAllowList: true })
  const tools = picker.isKnown
    ? picker.tools
    : (server.allowed_tools ?? []).filter(isLiteralToolName)

  const write = (body: Parameters<typeof set.mutate>[0]["body"]) => {
    if (isWriting) return
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
          <Select
            label="Availability"
            value={workspaceAvailability(server)}
            onChange={(next) =>
              write(availabilityOverride(server, next as McpAvailability))
            }
            options={AVAILABILITY_OPTIONS}
            labelAction={
              <McpInfoHint label="Explain the three choices">
                {AVAILABILITY_ORDER.map((value) => (
                  <span key={value}>
                    <span className="font-medium">
                      {AVAILABILITY[value].label}:
                    </span>{" "}
                    {AVAILABILITY[value].help}
                  </span>
                ))}
              </McpInfoHint>
            }
            isDisabled={isWriting}
            shouldReserveMessage={false}
            className="w-44"
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
                if (isWriting) return
                set.reset()
                setOpenCount((count) => count + 1)
                setEditingDefault(true)
              }}
              isDisabled={isWriting}
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
                if (isWriting) return
                set.reset()
                clear.mutate(ids)
              }}
              isPending={clear.isPending}
              isDisabled={!hasWorkspaceSettings(server) || isWriting}
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
            if (isWriting) return
            set.mutate(
              {
                ...ids,
                body: overrideBody(server, { everyone_tools: draft.tools }),
              },
              { onSuccess: () => setEditingDefault(false) },
            )
          }}
        />
      ) : null}
    </SettingsGroup>
  )
}
