import type { McpServerScope } from "@/shared/api/mcpServerScope"

/** The dashboard page a scope's servers are listed on. */
export function mcpServersPagePath(scope: McpServerScope): string {
  return scope.kind === "workspace"
    ? "/tools/mcp-servers"
    : "/organization/mcp-servers"
}

/** A server's own page, which a sign-in started there comes back to. */
export function mcpServerPagePath(
  scope: McpServerScope,
  serverId: string,
): string {
  return `${mcpServersPagePath(scope)}/${serverId}`
}
