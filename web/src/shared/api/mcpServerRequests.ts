import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  CreateMcpServerRequest,
  DeclineMcpServerRequests,
  McpServerRequest,
  McpServerRequestGroups,
  McpServerRequests,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import {
  MCP_CATALOG,
  MCP_SERVER_REQUESTS,
  WORKSPACES,
  workspaceMeta,
} from "@/shared/api/queryKeys"

// A member asks a workspace's admins for a server of the MCP catalog; an admin
// answers by adding it or declining. Both reads sit under one key, so any
// write here refreshes the member's list and the admin's together.

function requestsPath(workspaceId: string): string {
  return `/workspaces/${encodeURIComponent(workspaceId)}/mcp-server-requests`
}

function requestsKey(workspaceId: string | null) {
  return [WORKSPACES, workspaceId, MCP_SERVER_REQUESTS] as const
}

// Someone else answers a request, so a list is worth reading again this often.
const REQUESTS_STALE_MS = 30_000

/** The caller's own requests in a workspace: the waiting ones, and the answers they have not dismissed. */
export function useMyMcpServerRequests(
  workspaceId: string | null,
  enabled = true,
) {
  return useQuery({
    queryKey: [...requestsKey(workspaceId), "mine"],
    meta: workspaceMeta(workspaceId),
    enabled: enabled && workspaceId !== null,
    staleTime: REQUESTS_STALE_MS,
    queryFn: ({ signal }) =>
      apiFetch<McpServerRequests>(
        `${requestsPath(workspaceId as string)}/mine`,
        { signal },
      ),
  })
}

/** Every waiting request of a workspace, one entry per server. Workspace managers only. */
export function useMcpServerRequests(
  workspaceId: string | null,
  enabled: boolean,
) {
  return useQuery({
    queryKey: [...requestsKey(workspaceId), "pending"],
    meta: workspaceMeta(workspaceId),
    enabled: enabled && workspaceId !== null,
    staleTime: REQUESTS_STALE_MS,
    queryFn: ({ signal }) =>
      apiFetch<McpServerRequestGroups>(requestsPath(workspaceId as string), {
        signal,
      }),
  })
}

export function useRequestMcpServer() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      body,
    }: {
      workspaceId: string
      body: CreateMcpServerRequest
    }) =>
      apiFetch<McpServerRequest>(requestsPath(workspaceId), {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: (_data, { workspaceId }) => {
      void queryClient.invalidateQueries({ queryKey: requestsKey(workspaceId) })
    },
    onError: (_error, { workspaceId }) => {
      // A refusal can mean the list on screen is behind: someone added the
      // server, or the request is already waiting.
      void queryClient.invalidateQueries({ queryKey: requestsKey(workspaceId) })
      void queryClient.invalidateQueries({ queryKey: [MCP_CATALOG] })
    },
  })
}

/** Withdraws one of the caller's own waiting requests. */
export function useWithdrawMcpServerRequest() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      requestId,
    }: {
      workspaceId: string
      requestId: string
    }) =>
      apiFetch<void>(
        `${requestsPath(workspaceId)}/${encodeURIComponent(requestId)}`,
        { method: "DELETE" },
      ),
    onSettled: (_data, _error, { workspaceId }) => {
      void queryClient.invalidateQueries({ queryKey: requestsKey(workspaceId) })
    },
  })
}

/**
 * Records that the caller has seen the answer to one of their own requests,
 * which takes it off their list for good. It leaves the list on screen at
 * once, and comes back if the gateway refuses.
 */
export function useAcknowledgeMcpServerRequest() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      requestId,
    }: {
      workspaceId: string
      requestId: string
    }) =>
      apiFetch<void>(
        `${requestsPath(workspaceId)}/${encodeURIComponent(requestId)}/acknowledge`,
        { method: "POST" },
      ),
    onMutate: ({ workspaceId, requestId }) => {
      queryClient.setQueryData<McpServerRequests>(
        [...requestsKey(workspaceId), "mine"],
        (previous) =>
          previous && {
            ...previous,
            data: previous.data.filter((entry) => entry.id !== requestId),
          },
      )
    },
    onSettled: (_data, _error, { workspaceId }) => {
      void queryClient.invalidateQueries({ queryKey: requestsKey(workspaceId) })
    },
  })
}

/** Declines everyone waiting for one catalog server. Workspace managers only. */
export function useDeclineMcpServerRequests() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      body,
    }: {
      workspaceId: string
      body: DeclineMcpServerRequests
    }) =>
      apiFetch<{ declined: number }>(`${requestsPath(workspaceId)}/decline`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSettled: (_data, _error, { workspaceId }) => {
      void queryClient.invalidateQueries({ queryKey: requestsKey(workspaceId) })
    },
  })
}
