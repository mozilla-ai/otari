import {
  keepPreviousData,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query"
import type {
  McpCatalogFeatured,
  McpCatalogFeaturedList,
  McpCatalogServers,
  McpCatalogSyncStatus,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import { MCP_CATALOG, MCP_CATALOG_SYNC_POLL_MS } from "@/shared/api/queryKeys"

// A scheduled sync can start while the page sits open, so idle status is rechecked slowly.
const MCP_CATALOG_SYNC_IDLE_POLL_MS = 60_000

// A sync changes the catalog at most a few times a day, so a page read a
// minute ago is still the page.
const CATALOG_STALE_MS = 60_000

/**
 * One page of the catalog, filtered by the gateway. The previous page stays on
 * screen while the next query is in flight, so typing does not blank the list.
 */
export function useMcpCatalogServers({
  query,
  page,
  pageSize,
}: {
  query: string
  /** 0-based. */
  page: number
  /** At most 100, the gateway's bound. */
  pageSize: number
}) {
  return useQuery({
    queryKey: [MCP_CATALOG, "servers", query, page, pageSize],
    placeholderData: keepPreviousData,
    staleTime: CATALOG_STALE_MS,
    queryFn: () => {
      const params = new URLSearchParams({
        skip: String(page * pageSize),
        limit: String(pageSize),
      })
      if (query) params.set("q", query)
      return apiFetch<McpCatalogServers>(`/mcp-catalog/servers?${params}`)
    },
  })
}

/** The servers the caller's organization recommends. Any member reads it. */
export function useFeaturedMcpServers(enabled = true) {
  return useQuery({
    queryKey: [MCP_CATALOG, "featured"],
    enabled,
    // Changed only by this organization's admins, and every write here invalidates it.
    staleTime: CATALOG_STALE_MS,
    queryFn: () =>
      apiFetch<McpCatalogFeaturedList>(
        "/organizations/me/mcp-catalog/featured",
      ),
  })
}

/** Feature a catalog server, or take its copy again if it is already featured. */
export function useFeatureMcpServer() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (serverId: string) =>
      apiFetch<McpCatalogFeatured>("/organizations/me/mcp-catalog/featured", {
        method: "POST",
        body: JSON.stringify({ server_id: serverId }),
      }),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: [MCP_CATALOG, "featured"] }),
  })
}

export function useUnfeatureMcpServer() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (featuredId: string) =>
      apiFetch<void>(
        `/organizations/me/mcp-catalog/featured/${encodeURIComponent(featuredId)}`,
        { method: "DELETE" },
      ),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: [MCP_CATALOG, "featured"] }),
  })
}

/**
 * The sync's status, for an operator. Polled only while a sync is running or
 * waiting for a worker, which is the only time it changes without this tab.
 */
export function useMcpCatalogSync(enabled: boolean) {
  return useQuery({
    queryKey: [MCP_CATALOG, "sync"],
    enabled,
    queryFn: () => apiFetch<McpCatalogSyncStatus>("/mcp-catalog/sync"),
    // A worker moves it, not this tab, so what is cached is never trusted.
    staleTime: 0,
    refetchInterval: (query) =>
      query.state.data?.running || query.state.data?.requested
        ? MCP_CATALOG_SYNC_POLL_MS
        : MCP_CATALOG_SYNC_IDLE_POLL_MS,
  })
}

/** Ask for a sync now. A worker starts it within a minute; the status poll shows it. */
export function useRequestMcpCatalogSync() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () =>
      apiFetch<McpCatalogSyncStatus>("/mcp-catalog/sync", { method: "POST" }),
    onSuccess: (status) =>
      queryClient.setQueryData([MCP_CATALOG, "sync"], status),
  })
}
