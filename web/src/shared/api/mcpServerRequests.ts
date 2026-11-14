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

/** One page of a workspace's waiting requests, one entry per server. Workspace managers only. */
export function useMcpServerRequests(
  workspaceId: string | null,
  { page, pageSize }: { page: number; pageSize: number },
  enabled: boolean,
) {
  return useQuery({
    queryKey: [...requestsKey(workspaceId), "pending", page, pageSize],
    meta: workspaceMeta(workspaceId),
    enabled: enabled && workspaceId !== null,
    staleTime: REQUESTS_STALE_MS,
    // Kept across a page change, never across a workspace change.
    placeholderData: (
      previous: McpServerRequestGroups | undefined,
      previousQuery,
    ) => (previousQuery?.queryKey[1] === workspaceId ? previous : undefined),
    queryFn: ({ signal }) => {
      const params = new URLSearchParams({
        skip: String(page * pageSize),
        limit: String(pageSize),
      })
      return apiFetch<McpServerRequestGroups>(
        `${requestsPath(workspaceId as string)}?${params}`,
        { signal },
      )
    },
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
    mutationKey: [MCP_SERVER_REQUESTS, "acknowledge"],
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
    onMutate: async ({ workspaceId, requestId }) => {
      const key = [...requestsKey(workspaceId), "mine"]
      // A read already on its way would land with the answer still in it.
      await queryClient.cancelQueries({ queryKey: key })
      const previous = queryClient.getQueryData<McpServerRequests>(key)
      const index = previous?.data.findIndex((entry) => entry.id === requestId)
      if (!previous || index === undefined || index < 0) return undefined
      queryClient.setQueryData<McpServerRequests>(key, {
        ...previous,
        data: previous.data.filter((entry) => entry.id !== requestId),
      })
      return { removed: previous.data[index], index }
    },
    onError: (_error, { workspaceId }, context) => {
      if (!context) return
      // Only this answer comes back: another dismissed meanwhile stays gone.
      queryClient.setQueryData<McpServerRequests>(
        [...requestsKey(workspaceId), "mine"],
        (current) => {
          if (
            !current ||
            current.data.some((entry) => entry.id === context.removed.id)
          )
            return current
          const data = [...current.data]
          data.splice(Math.min(context.index, data.length), 0, context.removed)
          return { ...current, data }
        },
      )
    },
    onSettled: (_data, _error, { workspaceId }) => {
      // The last of several dismissals reads the list again; an earlier one
      // would bring back an answer whose own dismissal is still on its way.
      if (
        queryClient.isMutating({
          mutationKey: [MCP_SERVER_REQUESTS, "acknowledge"],
        }) > 1
      )
        return
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
