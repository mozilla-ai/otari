import {
  type QueryClient,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query"

import type { OrgWebSearchKey, OrgWebSearchKeys } from "@/client"
import { apiFetch } from "@/shared/api/client"
import {
  ORGANIZATION_WEB_SEARCH_KEYS,
  WORKSPACES,
} from "@/shared/api/queryKeys"

/**
 * The API's maximum. An organization adds its keys by hand, so one page is the
 * whole list.
 */
export const KEY_PAGE_LIMIT = 1000

// The organization's own search and fetch keys, under
// `/organizations/me/web-search-keys`. Archived keys are read too and filtered
// on the page: the list is small, and "show archived" should not be a round
// trip.
export function useOrgWebSearchKeys(enabled = true) {
  return useQuery({
    queryKey: [ORGANIZATION_WEB_SEARCH_KEYS],
    queryFn: () =>
      apiFetch<OrgWebSearchKeys>(
        `/organizations/me/web-search-keys?include_archived=true&limit=${KEY_PAGE_LIMIT}`,
      ),
    staleTime: 60_000,
    enabled,
  })
}

// Every write can change which key a workspace uses, and so what its listing
// and its web search page report (`web_search_configured`), so each one reads
// those again with the keys.
function invalidateKeys(queryClient: QueryClient): void {
  void queryClient.invalidateQueries({
    queryKey: [ORGANIZATION_WEB_SEARCH_KEYS],
  })
  void queryClient.invalidateQueries({
    predicate: ({ queryKey }) =>
      queryKey[0] === WORKSPACES &&
      (queryKey[2] === "web-search-keys" || queryKey[2] === "web-search"),
  })
}

function keyPath(keyId: string, action = ""): string {
  return `/organizations/me/web-search-keys/${encodeURIComponent(keyId)}${action}`
}

/** A `POST` or `DELETE` on one key's action route, answered with the key. */
function useKeyAction(action: string, method: "POST" | "DELETE") {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (keyId: string) =>
      apiFetch<OrgWebSearchKey>(keyPath(keyId, action), { method }),
    onSuccess: () => invalidateKeys(queryClient),
  })
}

// Archiving also stops a key being the default, and making one the default
// stops the previous one, so each of these reads the whole list again.
export function useArchiveOrgWebSearchKey() {
  return useKeyAction("/archive", "POST")
}

export function useRestoreOrgWebSearchKey() {
  return useKeyAction("/restore", "POST")
}

export function useSetOrgWebSearchKeyDefault() {
  return useKeyAction("/default", "POST")
}

export function useUnsetOrgWebSearchKeyDefault() {
  return useKeyAction("/default", "DELETE")
}

export function useDeleteOrgWebSearchKey() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (keyId: string) =>
      apiFetch<void>(keyPath(keyId), { method: "DELETE" }),
    onSuccess: () => invalidateKeys(queryClient),
  })
}
