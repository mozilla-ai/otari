import type { ReactNode } from "react"

import { SettingRow } from "@/design-system/layout/SettingRow"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { usePlaygroundTools } from "@/shared/api/playground"

/**
 * Whether a member's workspace allows a gateway-run tool, read-only.
 *
 * The workspace's own setting is for owners and admins, so this reads the
 * playground's per-workspace answer, which any member may and which reads "no
 * setting" the way the request path does.
 */
export function WorkspaceToolStatus({
  tool,
  leading,
  trailing,
  workspace,
}: {
  tool: "web_search" | "code_execution"
  leading?: ReactNode
  /** Read-only rows after the setting, such as what the tool runs on. */
  trailing?: ReactNode
  workspace: { workspace_id: string; name: string }
}) {
  const status = usePlaygroundTools(workspace.workspace_id).data?.[tool]
  if (!leading && !trailing && !status?.configured) return null
  return (
    <SettingsGroup isBounded>
      {leading}
      {status?.configured ? (
        <SettingRow
          label={`Allowed in ${workspace.name}`}
          help={
            status.enabled
              ? "Set by an owner or admin."
              : "An owner or admin can turn it on for this workspace."
          }
          control={
            <span className="text-caption text-foreground">
              {status.enabled ? "Yes" : "No"}
            </span>
          }
        />
      ) : null}
      {trailing}
    </SettingsGroup>
  )
}
