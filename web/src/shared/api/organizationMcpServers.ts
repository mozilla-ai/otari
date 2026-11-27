import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"

import type {
  CreateWorkspaceMcpServerRequest,
  OrganizationMcpServerLimitRequest,
  OrganizationMcpServerWorkspace,
  OrganizationMcpServerWorkspaces,
  WorkspaceMcpServer,
  WorkspaceMcpServers,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import {
  invalidateEveryWorkspaceMcpView,
  invalidateMcpServers,
  mcpServerKey,
  mcpServerPath,
  mcpServersKey,
  mcpServersPath,
  ORGANIZATION_MCP_SCOPE,
} from "@/shared/api/mcpServerScope"

// The organization's own MCP servers, under `/organizations/me/mcp-servers`,
// for its owners and admins. The writes they share with a workspace's servers
// (settings, checks, sign-ins, connections) are the scoped hooks in `tools.ts`;
// these are the ones only the organization has.

/**
 * The route's own ceiling. The service caps how many servers an organization
 * holds below it, so one page is the whole list.
 */
const ORGANIZATION_MCP_SERVERS_PAGE_SIZE = 100

const SCOPE = ORGANIZATION_MCP_SCOPE

export function useOrganizationMcpServers(enabled = true) {
  return useQuery({
    queryKey: mcpServersKey(SCOPE),
    queryFn: () =>
      apiFetch<WorkspaceMcpServers>(
        `${mcpServersPath(SCOPE)}?limit=${ORGANIZATION_MCP_SERVERS_PAGE_SIZE}`,
      ),
    enabled,
    staleTime: 60_000,
  })
}

export function useOrganizationMcpServer(serverId: string, enabled = true) {
  return useQuery({
    queryKey: mcpServerKey(SCOPE, serverId),
    queryFn: () => apiFetch<WorkspaceMcpServer>(mcpServerPath(SCOPE, serverId)),
    enabled,
    staleTime: 60_000,
  })
}

export function useCreateOrganizationMcpServer() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateWorkspaceMcpServerRequest) =>
      apiFetch<WorkspaceMcpServer>(mcpServersPath(SCOPE), {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => invalidateMcpServers(queryClient, SCOPE),
  })
}

export function useDeleteOrganizationMcpServer() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (serverId: string) =>
      apiFetch<void>(mcpServerPath(SCOPE, serverId), { method: "DELETE" }),
    onSuccess: (_data, serverId) => {
      // Dropped rather than refetched: the server's own page is still mounted
      // while it navigates away, and a refetch would only earn it a 404.
      queryClient.removeQueries({ queryKey: mcpServerKey(SCOPE, serverId) })
      void queryClient.invalidateQueries({
        queryKey: mcpServersKey(SCOPE),
        exact: true,
      })
      invalidateEveryWorkspaceMcpView(queryClient)
    },
  })
}

function workspacesKey(serverId: string) {
  return [...mcpServerKey(SCOPE, serverId), "workspaces"] as const
}

/** Each workspace of the organization, whether it reaches the server, and the limit set for it. */
export function useOrganizationMcpServerWorkspaces(
  serverId: string,
  enabled = true,
) {
  return useQuery({
    queryKey: workspacesKey(serverId),
    queryFn: () =>
      apiFetch<OrganizationMcpServerWorkspaces>(
        `${mcpServerPath(SCOPE, serverId)}/workspaces`,
      ),
    enabled,
    // A workspace's admins change their own side without this page hearing of it.
    staleTime: 30_000,
  })
}

function workspaceLimitPath(serverId: string, workspaceId: string) {
  return `${mcpServerPath(SCOPE, serverId)}/workspaces/${encodeURIComponent(workspaceId)}`
}

// A limit moves the one workspace's row here, and what that workspace's keys,
// members and Playground reach on the server. The write stays pending until
// the table is read again, so a row is never acted on from what it said before.
// The answered row goes into the table first, so a failed reread still shows
// what was saved.
function useLimitWrite<Variables extends { serverId: string }>(
  request: (variables: Variables) => Promise<OrganizationMcpServerWorkspace>,
) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: request,
    onSuccess: (row, { serverId }) => {
      queryClient.setQueryData<OrganizationMcpServerWorkspaces>(
        workspacesKey(serverId),
        (table) =>
          table && {
            ...table,
            data: table.data.map((each) =>
              each.workspace_id === row.workspace_id ? row : each,
            ),
          },
      )
      invalidateEveryWorkspaceMcpView(queryClient)
      return queryClient.invalidateQueries({
        queryKey: workspacesKey(serverId),
      })
    },
  })
}

/** Sets the most a workspace's callers reach on the server; `[]` turns it off there. */
export function useSetOrganizationMcpServerLimit() {
  return useLimitWrite(
    ({
      serverId,
      workspaceId,
      tools,
    }: {
      serverId: string
      workspaceId: string
      tools: string[]
    }) => {
      const body: OrganizationMcpServerLimitRequest = { tools }
      return apiFetch<OrganizationMcpServerWorkspace>(
        workspaceLimitPath(serverId, workspaceId),
        { method: "PUT", body: JSON.stringify(body) },
      )
    },
  )
}

/** Lifts the organization's limit in a workspace, leaving its own rules alone. */
export function useLiftOrganizationMcpServerLimit() {
  return useLimitWrite(
    ({ serverId, workspaceId }: { serverId: string; workspaceId: string }) =>
      apiFetch<OrganizationMcpServerWorkspace>(
        workspaceLimitPath(serverId, workspaceId),
        { method: "DELETE" },
      ),
  )
}
