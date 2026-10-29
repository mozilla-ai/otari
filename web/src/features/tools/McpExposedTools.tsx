import { useState } from "react"

import type { WorkspaceMcpServer } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { RadioGroup } from "@/design-system/forms/RadioGroup"
import { Dot } from "@/design-system/indicators/Dot"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import {
  healthDetail,
  healthNeedsAttention,
} from "@/features/tools/mcpServerHealth"
import { ToolChecklist } from "@/features/tools/ToolChecklist"
import {
  useCheckWorkspaceMcpServer,
  useUpdateWorkspaceMcpServer,
} from "@/shared/api/tools"
import { formatDateTime } from "@/shared/helpers/format"

type Exposure = "all" | "some"

/** What the server exposes, in one line: "3 of 4: list_issues, get_issue, create_issue". */
function exposedSummary(
  stored: readonly string[] | null,
  offered: readonly string[],
): string {
  if (stored === null) {
    return offered.length > 0
      ? `All ${offered.length} tools the server offers`
      : "Every tool the server offers"
  }
  const prefix =
    offered.length > 0 ? `${stored.length} of ${offered.length}: ` : ""
  return `${prefix}${stored.join(", ")}`
}

/**
 * Which of a server's tools otari exposes at all: the filter a call meets
 * before any ACL rule. A one-line summary with the connection's state, and a
 * checklist behind Edit. A change checks the server again, as the gateway does
 * for any change to the allow-list.
 *
 * The caller keys this on the stored allow-list, so a save re-seeds it and
 * closes the editor.
 */
export function McpExposedTools({
  workspaceId,
  server,
  canManage,
}: {
  workspaceId: string
  server: WorkspaceMcpServer
  canManage: boolean
}) {
  const update = useUpdateWorkspaceMcpServer()
  const check = useCheckWorkspaceMcpServer()
  const stored = server.allowed_tools
  const [isEditing, setEditing] = useState(false)
  const [exposure, setExposure] = useState<Exposure>(
    stored === null ? "all" : "some",
  )
  const [picked, setPicked] = useState<string[]>(stored ?? [])
  const health = server.health
  const offered = health?.tools ?? []
  const next = exposure === "all" ? null : picked
  const isChanged = JSON.stringify(next) !== JSON.stringify(stored)
  // `[]` would expose nothing, which the gateway reads as a deny-all; the way
  // to stop a server answering is to disable it, so this never sends one.
  const isComplete = exposure === "all" || picked.length > 0
  const missing = health?.missing_allowed_tools ?? []

  return (
    <SettingsGroup
      title="Exposed tools"
      isBounded
      description="Applies before the ACL: a tool the server does not expose here is never reachable, whatever a rule says."
      action={
        canManage ? (
          <div className="flex flex-wrap gap-2">
            <Button
              onPress={() => check.mutate({ workspaceId, serverId: server.id })}
              isPending={check.isPending}
            >
              Check connection
            </Button>
            {isEditing ? null : (
              <Button
                onPress={() => setEditing(true)}
                aria-label="Edit exposed tools"
              >
                Edit
              </Button>
            )}
          </div>
        ) : null
      }
    >
      <div className="flex flex-col gap-3">
        <p className="flex flex-wrap items-center gap-2 text-body">
          {health === null ? (
            <span className="text-muted">
              Not checked yet. Check the connection to list its tools.
            </span>
          ) : (
            <>
              <Dot
                className={
                  health.status === "failing"
                    ? "bg-danger"
                    : healthNeedsAttention(health)
                      ? "bg-warning"
                      : "bg-success"
                }
              />
              <span className="break-words">{healthDetail(health)}</span>
              <span className="text-caption">
                Checked {formatDateTime(health.checked_at)}
              </span>
            </>
          )}
        </p>
        <ErrorBanner error={check.error} />
        {isEditing ? (
          <div className="flex flex-col gap-3">
            <RadioGroup
              label="Expose"
              hideLabel
              value={exposure}
              onChange={(value) => setExposure(value as Exposure)}
              options={[
                { value: "all", label: "Every tool the server offers" },
                { value: "some", label: "Only the tools ticked" },
              ]}
            />
            {exposure === "some" ? (
              <div className="pl-6">
                <ToolChecklist
                  label="Exposed tools"
                  tools={offered}
                  selected={picked}
                  onChange={setPicked}
                  inputLabel="Tool name"
                />
              </div>
            ) : null}
            <ErrorBanner error={update.error} />
            <div className="flex flex-wrap gap-2">
              <Button
                variant="primary"
                onPress={() =>
                  update.mutate({
                    workspaceId,
                    serverId: server.id,
                    body: { allowed_tools: next },
                  })
                }
                isPending={update.isPending}
                isDisabled={!isChanged || !isComplete}
              >
                Save exposed tools
              </Button>
              <Button
                onPress={() => {
                  setEditing(false)
                  setExposure(stored === null ? "all" : "some")
                  setPicked(stored ?? [])
                  update.reset()
                }}
              >
                Cancel
              </Button>
            </div>
          </div>
        ) : (
          <p className="font-mono text-sm break-words">
            {exposedSummary(stored, offered)}
          </p>
        )}
        {missing.length > 0 ? (
          <InfoBanner tone="warning">
            Not offered by the server: {missing.join(", ")}
          </InfoBanner>
        ) : null}
      </div>
    </SettingsGroup>
  )
}
