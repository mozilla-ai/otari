import {
  keepPreviousData,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query"
import type { McpCatalogServers, McpCatalogSyncStatus } from "@/client"
import { apiFetch } from "@/shared/api/client"
import { MCP_CATALOG, MCP_CATALOG_SYNC_POLL_MS } from "@/shared/api/queryKeys"

// A sync changes the catalog at most a few times a day, so a page read a
// minute ago is still the page.
const CATALOG_STALE_MS = 60_000

/** Whether a sync is under way, which is when the lists below are worth re-reading. */
export function isMcpCatalogSyncing(
  status: McpCatalogSyncStatus | undefined,
): boolean {
  return status?.running === true || status?.requested === true
}

/**
 * When to read the sync's status again: on a short poll while a sync runs or
 * waits for a worker, once just after `next_request_at` while a cooldown holds
 * the request control off, and otherwise not on a timer.
 */
export function mcpCatalogSyncRefetchMs(
  status: McpCatalogSyncStatus | undefined,
  now = Date.now(),
): number | false {
  if (isMcpCatalogSyncing(status)) return MCP_CATALOG_SYNC_POLL_MS
  const until = status?.next_request_at
    ? Date.parse(status.next_request_at) - now
    : Number.NaN
  return Number.isFinite(until) ? Math.max(until, 0) + 1_000 : false
}

/**
 * One page of the catalog, filtered by the gateway. The previous page stays on
 * screen while the next query is in flight, so typing does not blank the list.
 * Re-read while a sync stores pages, so the list fills in as it goes.
 */
export function useMcpCatalogServers({
  query,
  page,
  pageSize,
  workspaceId,
  enabled = true,
  isLoading = false,
}: {
  query: string
  /** 0-based. */
  page: number
  /** At most 100, the gateway's bound. */
  pageSize: number
  /** The servers this workspace already has are left out. */
  workspaceId: string
  enabled?: boolean
  /** A sync is running, so the catalog is growing under the reader. */
  isLoading?: boolean
}) {
  return useQuery({
    queryKey: [MCP_CATALOG, "servers", workspaceId, query, page, pageSize],
    enabled,
    placeholderData: keepPreviousData,
    staleTime: CATALOG_STALE_MS,
    refetchInterval: isLoading ? MCP_CATALOG_SYNC_POLL_MS : false,
    queryFn: () => {
      const params = new URLSearchParams({
        skip: String(page * pageSize),
        limit: String(pageSize),
        workspace_id: workspaceId,
      })
      if (query) params.set("q", query)
      return apiFetch<McpCatalogServers>(`/mcp-catalog/servers?${params}`)
    },
  })
}

/**
 * The well-known servers the catalog holds, from the list bundled with the
 * gateway, without the ones the workspace already has.
 */
export function useMcpCatalogPopular(workspaceId: string, isLoading = false) {
  return useQuery({
    queryKey: [MCP_CATALOG, "popular", workspaceId],
    staleTime: CATALOG_STALE_MS,
    refetchInterval: isLoading ? MCP_CATALOG_SYNC_POLL_MS : false,
    queryFn: () =>
      apiFetch<McpCatalogServers>(
        `/mcp-catalog/popular?workspace_id=${encodeURIComponent(workspaceId)}`,
      ),
  })
}

/**
 * The sync's status. Polled while a sync is running or waiting for a worker,
 * which is the only time it changes without this tab, and read again when a
 * cooldown ends so the request control comes back. When a sync ends, the
 * catalog's lists are read again, so they show what the sync stored.
 */
export function useMcpCatalogSync(enabled = true) {
  const queryClient = useQueryClient()
  return useQuery({
    queryKey: [MCP_CATALOG, "sync"],
    enabled,
    staleTime: 0,
    queryFn: async () => {
      const before = queryClient.getQueryData<McpCatalogSyncStatus>([
        MCP_CATALOG,
        "sync",
      ])
      const status = await apiFetch<McpCatalogSyncStatus>("/mcp-catalog/sync")
      if (isMcpCatalogSyncing(before) && !isMcpCatalogSyncing(status)) {
        void queryClient.invalidateQueries({
          predicate: (cached) =>
            cached.queryKey[0] === MCP_CATALOG && cached.queryKey[1] !== "sync",
        })
      }
      return status
    },
    refetchInterval: (query) => mcpCatalogSyncRefetchMs(query.state.data),
  })
}

/**
 * Ask for a sync now, on behalf of a workspace the caller may add servers to.
 * A worker starts it within a minute; the status poll shows it.
 */
export function useRequestMcpCatalogSync() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (workspaceId: string) =>
      apiFetch<McpCatalogSyncStatus>("/mcp-catalog/sync", {
        method: "POST",
        body: JSON.stringify({ workspace_id: workspaceId }),
      }),
    onSuccess: (status) =>
      queryClient.setQueryData([MCP_CATALOG, "sync"], status),
  })
}
