import type {
  SetWorkspaceMcpServerOverrideRequest,
  WorkspaceMcpServer,
} from "@/client"
import { ruleTools } from "@/features/tools/mcpAccess"
import {
  availabilityOf,
  type McpAvailability,
} from "@/features/tools/mcpAvailability"

// A workspace reads its organization's servers beside its own. On such a row
// the top-level fields are the organization's definition, and `in_workspace`
// carries the workspace's own settings on it (null fields take the server's)
// and the organization's limit there. These helpers are the one place that
// folds the two together.

/** Whether a row is its organization's server rather than the workspace's own. */
export function isOrganizationServer(server: WorkspaceMcpServer): boolean {
  return server.scope === "organization"
}

/**
 * The server as this workspace uses it: on an organization's server, whether
 * its requests reach it, whether it is attached and its default access are the
 * workspace's own where it set them, else the organization's. A workspace's
 * own server is returned as it is.
 */
export function asWorkspaceUses(
  server: WorkspaceMcpServer,
): WorkspaceMcpServer {
  const own = server.in_workspace
  if (!own) return server
  return {
    ...server,
    enabled: own.reached,
    attach_by_default: own.attach_by_default ?? server.attach_by_default,
    everyone_tools: own.everyone_tools ?? server.everyone_tools,
  }
}

/**
 * The workspace's own Availability choice on an organization's server, apart
 * from whether the organization lets it through: an admin who turned it on
 * sees it on, with a line saying the organization turned it off.
 */
export function workspaceAvailability(
  server: WorkspaceMcpServer,
): McpAvailability {
  const own = server.in_workspace
  if (!own) return availabilityOf(server)
  return availabilityOf({
    enabled: own.enabled,
    attach_by_default: own.attach_by_default ?? server.attach_by_default,
  })
}

/**
 * The body that sets one Availability choice, and nothing else: `PATCH
 * .../override` keeps every field left out, so a default policy another admin
 * saved meanwhile stays. Off keeps the stored attach setting.
 */
export function availabilityOverride(
  availability: McpAvailability,
): SetWorkspaceMcpServerOverrideRequest {
  if (availability === "off") return { enabled: false }
  return { enabled: true, attach_by_default: availability === "every_request" }
}

/**
 * Whether a default policy the editor returns is the one it opened on, in any
 * order. Saving it would store the inherited policy as the workspace's own,
 * which then stops following the organization's.
 */
export function isSamePolicy(
  draft: readonly string[],
  shown: readonly string[],
): boolean {
  return (
    draft.length === shown.length &&
    draft.every((pattern) => shown.includes(pattern))
  )
}

/** Whether the workspace set anything of its own, which "Use organization defaults" clears. */
export function hasWorkspaceSettings(server: WorkspaceMcpServer): boolean {
  const own = server.in_workspace
  if (!own) return false
  return (
    !own.enabled ||
    own.everyone_tools !== null ||
    own.attach_by_default !== null
  )
}

/** The organization's limit in a workspace, in words; empty where it set none. */
export function organizationLimitLabel(
  tools: readonly string[] | null | undefined,
): string {
  if (tools === null || tools === undefined) return ""
  if (tools.length === 0) return "No tools"
  return ruleTools(tools) === "Any" ? "Any tool" : ruleTools(tools)
}

/**
 * Why the workspace's requests do not reach an organization's server it left
 * on, when that is so: the organization turned the server off, or turned it
 * off for this workspace. Empty when they reach it, or when the workspace
 * turned it off itself.
 */
export function organizationOffReason(server: WorkspaceMcpServer): string {
  const own = server.in_workspace
  if (!own || own.reached || !own.enabled) return ""
  if (!server.enabled) return "Your organization turned this server off."
  return "Your organization turned this server off for this workspace."
}
