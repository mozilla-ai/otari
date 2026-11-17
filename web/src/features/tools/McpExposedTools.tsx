import { useState } from "react"
import { FiActivity } from "react-icons/fi"

import type { WorkspaceMcpServer } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { RadioGroup } from "@/design-system/forms/RadioGroup"
import { Dot } from "@/design-system/indicators/Dot"
import {
  SettingsBlock,
  SettingsGroup,
} from "@/design-system/layout/SettingsGroup"
import { McpNotice } from "@/features/tools/McpNotice"
import { isLiteralToolName } from "@/features/tools/mcpAccess"
import { asksForOAuthClient } from "@/features/tools/mcpErrors"
import {
  checkedTools,
  healthDetail,
  healthNeedsAttention,
} from "@/features/tools/mcpServerHealth"
import { ToolChecklist } from "@/features/tools/ToolChecklist"
import {
  PERSONAL_CHECK_HELP,
  useMcpServerCheck,
} from "@/features/tools/useMcpServerCheck"
import { useUpdateWorkspaceMcpServer } from "@/shared/api/tools"
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
 * for any change to the allow-list. The state is also refreshed by every
 * request that reaches the server, so Check is for a server nobody uses yet.
 *
 * The caller keys this on the stored allow-list, so a save re-seeds it and
 * closes the editor.
 */
export function McpExposedTools({
  workspaceId,
  server,
  canManage,
  onAddClient,
}: {
  workspaceId: string
  server: WorkspaceMcpServer
  canManage: boolean
  /** Opens Settings on the OAuth client ID, when a check's sign-in asked for one. */
  onAddClient?: () => void
}) {
  const update = useUpdateWorkspaceMcpServer()
  const check = useMcpServerCheck({
    workspaceId,
    returnTo: `/tools/mcp-servers/${server.id}`,
  })
  const isPersonal = server.credential_mode === "per_user"
  const stored = server.allowed_tools
  const [isEditing, setEditing] = useState(false)
  const [exposure, setExposure] = useState<Exposure>(
    stored === null ? "all" : "some",
  )
  const [picked, setPicked] = useState<string[]>(stored ?? [])
  const health = server.health
  const offered = (health?.tools ?? []).filter(isLiteralToolName)
  const picker = checkedTools(server)
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
      description="The tools Otari offers from this server. Hidden tools can't be used, whatever the rules say."
      action={
        canManage ? (
          <div className="flex flex-wrap gap-2">
            <Button
              onPress={() => check.run(server)}
              isPending={check.isRunning(server.id)}
            >
              <FiActivity aria-hidden className="size-4" />
              Check
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
      <SettingsBlock>
        <p className="flex flex-wrap items-center gap-2 text-body">
          {health === null ? (
            <span className="text-muted">
              Not checked yet. Check it to list its tools, or wait for the first
              request that uses it.
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
              <span className="break-words">
                {server.needs_setup
                  ? "The server asks for an account (HTTP 401). Set up the shared account above."
                  : healthDetail(health)}
              </span>
              <span className="text-caption">
                Last contact {formatDateTime(health.checked_at)}
              </span>
            </>
          )}
        </p>
        {check.error ? (
          <McpNotice
            message={check.error}
            action={
              onAddClient && asksForOAuthClient(check.error) ? (
                <Button size="sm" onPress={onAddClient}>
                  Open Settings
                </Button>
              ) : undefined
            }
            onDismiss={check.dismissError}
          />
        ) : null}
        {canManage && isPersonal ? (
          <p className="text-caption">{PERSONAL_CHECK_HELP}</p>
        ) : null}
        {isEditing ? (
          <div className="flex flex-col gap-3">
            <RadioGroup
              label="Expose"
              hideLabel
              value={exposure}
              onChange={(value) => setExposure(value as Exposure)}
              options={[
                { value: "all", label: "Every tool the server offers" },
                { value: "some", label: "Only the tools I pick" },
              ]}
            />
            {exposure === "some" ? (
              <div className="pl-6">
                <ToolChecklist
                  label="Exposed tools"
                  tools={picker.tools}
                  descriptions={picker.descriptions}
                  isKnown={picker.isKnown}
                  selected={picked}
                  onChange={setPicked}
                  allowPatterns={false}
                  onCheck={() => check.run(server)}
                  isChecking={check.isRunning(server.id)}
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
      </SettingsBlock>
    </SettingsGroup>
  )
}
