import type { QueryClient } from "@tanstack/react-query"

import {
  ORGANIZATION_MCP_SERVERS,
  PLAYGROUND,
  WORKSPACES,
  workspaceMeta,
} from "@/shared/api/queryKeys"

/**
 * Whose MCP servers a call is about: one workspace's own, or the
 * organization's, which every workspace of it reaches. The two are managed
 * through the same requests under different roots, so one hook serves both.
 */
export type McpServerScope =
  | { kind: "workspace"; workspaceId: string }
  | { kind: "organization" }

export const ORGANIZATION_MCP_SCOPE: McpServerScope = { kind: "organization" }

export function workspaceMcpScope(workspaceId: string): McpServerScope {
  return { kind: "workspace", workspaceId }
}

/** Where a scope's servers are managed. */
export function mcpServersPath(scope: McpServerScope): string {
  return scope.kind === "workspace"
    ? `/workspaces/${encodeURIComponent(scope.workspaceId)}/mcp-servers`
    : "/organizations/me/mcp-servers"
}

export function mcpServerPath(scope: McpServerScope, serverId: string): string {
  return `${mcpServersPath(scope)}/${encodeURIComponent(serverId)}`
}

/** The key every read of a scope's servers sits under, the list's own included. */
export function mcpServersKey(scope: McpServerScope): readonly unknown[] {
  return scope.kind === "workspace"
    ? [WORKSPACES, scope.workspaceId, "mcp-servers"]
    : [ORGANIZATION_MCP_SERVERS]
}

/** One server, under the list's key, so every write that refreshes the list refreshes it too. */
export function mcpServerKey(
  scope: McpServerScope,
  serverId: string,
): readonly unknown[] {
  return [...mcpServersKey(scope), serverId]
}

export function mcpServerConnectionsKey(
  scope: McpServerScope,
  serverId: string,
): readonly unknown[] {
  return [...mcpServerKey(scope, serverId), "connections"]
}

/** The workspace a scope's reads are keyed on, for `useSwitchOrganization`; none for the organization's. */
export function mcpScopeMeta(scope: McpServerScope) {
  return workspaceMeta(scope.kind === "workspace" ? scope.workspaceId : null)
}

/**
 * Every workspace's view of MCP servers: its servers and what a key or a
 * member reaches on them, its members' connections, and the Playground's tools
 * menu. An organization's server is reached through each of them, so a write
 * to one refreshes them all.
 */
export function invalidateEveryWorkspaceMcpView(queryClient: QueryClient) {
  void queryClient.invalidateQueries({
    predicate: ({ queryKey }) =>
      (queryKey[0] === WORKSPACES &&
        (queryKey[2] === "mcp-servers" || queryKey[2] === "mcp-connections")) ||
      (queryKey[0] === PLAYGROUND && queryKey[2] === "tools"),
  })
}

/**
 * What a write to a server moves: its scope's servers, and for an
 * organization's server every workspace that reaches it.
 */
export function invalidateMcpServers(
  queryClient: QueryClient,
  scope: McpServerScope,
) {
  void queryClient.invalidateQueries({ queryKey: mcpServersKey(scope) })
  if (scope.kind === "organization") {
    invalidateEveryWorkspaceMcpView(queryClient)
  }
}
