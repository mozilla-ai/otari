import type { PlaygroundTools } from "@/client"
import { usePlaygroundTools } from "@/shared/api/playground"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

/** Whether the selected workspace's requests may declare one gateway-run tool. */
export interface WorkspaceToolState {
  workspace: string
  /** `null` until the read answers, or when it failed. */
  enabled: boolean | null
  /** Why the workspace's requests may not declare it. */
  reason: string | null
  isError: boolean
}

const KEYS = {
  otari_web_search: "web_search",
  otari_web_fetch: "web_fetch",
  otari_code_execution: "code_execution",
} as const satisfies Record<string, keyof PlaygroundTools>

/**
 * The selected workspace's answer for each gateway-run tool, read the way the
 * request path reads it: its switch, and for the web tools its domain rules and
 * its organization's keys. Any member may read it. `undefined` for a tool with
 * no such answer, or with no workspace selected.
 */
export function useWorkspaceToolState(): (
  toolId: string,
) => WorkspaceToolState | undefined {
  const { selected } = useSelectedWorkspace()
  const tools = usePlaygroundTools(selected?.workspace_id)
  return (toolId) => {
    if (!selected || !(toolId in KEYS)) return undefined
    const status = tools.data?.[KEYS[toolId as keyof typeof KEYS]]
    return {
      workspace: selected.name,
      enabled: status ? status.enabled : null,
      reason: status?.reason ?? null,
      isError: tools.isError,
    }
  }
}
