import {
  type QueryClient,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query"

import type {
  CreateWorkspaceMcpServerRequest,
  McpCredentialMode,
  McpServerFoldCandidate,
  McpServerFoldCandidates,
  OrganizationMcpServerLimitRequest,
  OrganizationMcpServerWorkspace,
  OrganizationMcpServerWorkspaces,
  PromoteMcpServerRequest,
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
  workspaceMcpScope,
} from "@/shared/api/mcpServerScope"
import {
  MCP_CATALOG,
  MCP_SERVER_REQUESTS,
  ORGANIZATION_MCP_SERVERS,
  WORKSPACES,
} from "@/shared/api/queryKeys"

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

/**
 * What a new organization server moves beyond its own list: every workspace
 * reaches it, the copies it folded are gone from theirs, the gateway approves
 * every workspace's waiting requests for it, and each workspace's catalog
 * leaves it out.
 */
function invalidateAfterOrganizationServer(queryClient: QueryClient) {
  invalidateMcpServers(queryClient, SCOPE)
  void queryClient.invalidateQueries({
    predicate: ({ queryKey }) =>
      queryKey[0] === WORKSPACES && queryKey[2] === MCP_SERVER_REQUESTS,
  })
  void queryClient.invalidateQueries({
    predicate: ({ queryKey }) =>
      queryKey[0] === MCP_CATALOG && queryKey[1] !== "sync",
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
    onSuccess: () => invalidateAfterOrganizationServer(queryClient),
  })
}

/** Which copies to look for: those of a server about to be added, or of one about to be promoted. */
export type McpServerCopiesOf =
  | { url: string; credentialMode: McpCredentialMode }
  | { url: string; serverId: string }

function copiesQuery(of: McpServerCopiesOf) {
  const params = new URLSearchParams({ url: of.url })
  if ("serverId" in of) params.set("server_id", of.serverId)
  else params.set("credential_mode", of.credentialMode)
  return {
    queryKey: [ORGANIZATION_MCP_SERVERS, "same-url", params.toString()],
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      apiFetch<McpServerFoldCandidates>(
        `${mcpServersPath(SCOPE)}/same-url?${params}`,
        { signal },
      ).then((answer) => answer.data),
    // Read as the step opens: another admin's write changes it at any time.
    staleTime: 0,
  }
}

/**
 * The workspaces' own servers at a URL, which an organization server can take
 * the place of, and what folding each would do.
 */
export function useMcpServerCopies(of: McpServerCopiesOf, enabled: boolean) {
  return useQuery({ ...copiesQuery(of), enabled })
}

/** The same read, for a form that asks at the moment it is submitted. */
export function useReadMcpServerCopies(): (
  of: McpServerCopiesOf,
) => Promise<McpServerFoldCandidate[]> {
  const queryClient = useQueryClient()
  return (of) => queryClient.fetchQuery(copiesQuery(of))
}

/**
 * Makes a workspace's own server its organization's, in place: it keeps its
 * id, settings and sign-ins, and folds the copies named.
 */
export function usePromoteWorkspaceMcpServer() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      serverId,
      body,
    }: {
      workspaceId: string
      serverId: string
      body: PromoteMcpServerRequest
    }) =>
      apiFetch<WorkspaceMcpServer>(
        `${mcpServerPath(workspaceMcpScope(workspaceId), serverId)}/promote`,
        { method: "POST", body: JSON.stringify(body) },
      ),
    onSuccess: () => invalidateAfterOrganizationServer(queryClient),
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
// members and Playground reach on the server.
function useLimitWrite<Variables extends { serverId: string }>(
  request: (variables: Variables) => Promise<OrganizationMcpServerWorkspace>,
) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: request,
    onSuccess: (_row, { serverId }) => {
      void queryClient.invalidateQueries({ queryKey: workspacesKey(serverId) })
      invalidateEveryWorkspaceMcpView(queryClient)
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
