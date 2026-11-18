import {
  type QueryClient,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query"

import type {
  CreateOrgWebSearchKeyRequest,
  OrgWebSearchKey,
  OrgWebSearchKeys,
  UpdateOrgWebSearchKeyRequest,
  WorkspaceWebSearchKeys,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import {
  ORGANIZATION_WEB_SEARCH_KEYS,
  WORKSPACES,
  workspaceMeta,
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

export function useCreateOrgWebSearchKey() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateOrgWebSearchKeyRequest) =>
      apiFetch<OrgWebSearchKey>("/organizations/me/web-search-keys", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => invalidateKeys(queryClient),
  })
}

export function useUpdateOrgWebSearchKey() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      keyId,
      body,
    }: {
      keyId: string
      body: UpdateOrgWebSearchKeyRequest
    }) =>
      apiFetch<OrgWebSearchKey>(keyPath(keyId), {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: () => invalidateKeys(queryClient),
  })
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

function workspacePath(workspaceId: string, keyId = ""): string {
  const base = `/workspaces/${encodeURIComponent(workspaceId)}/web-search-keys`
  return keyId === "" ? base : `${base}/${encodeURIComponent(keyId)}`
}

/** The workspace's listing, every key of its organization on one page. */
function workspaceListPath(workspaceId: string): string {
  return `${workspacePath(workspaceId)}?limit=${KEY_PAGE_LIMIT}`
}

// One workspace's view of its organization's keys, which any member reads:
// which key of each kind it uses, and what its searches and fetches run on.
export function useWorkspaceWebSearchKeys(workspaceId: string | null) {
  return useQuery({
    queryKey: [WORKSPACES, workspaceId, "web-search-keys"],
    meta: workspaceMeta(workspaceId),
    queryFn: () =>
      apiFetch<WorkspaceWebSearchKeys>(
        workspaceListPath(workspaceId as string),
      ),
    enabled: workspaceId !== null,
    staleTime: 60_000,
  })
}

// Both writes answer with the workspace's listing, which replaces the cached
// one at once. Once a write settles, failed or not, the listing and the web
// search page (which reports whether a key serves the workspace) are read
// again: a reset that failed partway has already removed some marks, and two
// writes can answer out of order.
function useWorkspaceKeysWrite(
  write: (
    workspaceId: string,
    keyIds: string[],
  ) => Promise<WorkspaceWebSearchKeys>,
) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      keyIds,
    }: {
      workspaceId: string
      keyIds: string[]
    }) => write(workspaceId, keyIds),
    onSuccess: (listing, { workspaceId }) => {
      queryClient.setQueryData(
        [WORKSPACES, workspaceId, "web-search-keys"],
        listing,
      )
    },
    onSettled: (_listing, _error, { workspaceId }) => {
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "web-search-keys"],
      })
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "web-search"],
      })
    },
  })
}

/** Pin one key as the workspace's own for its kind; the gateway unpins the other. */
export function usePinWorkspaceWebSearchKey() {
  return useWorkspaceKeysWrite((workspaceId, [keyId]) =>
    apiFetch<WorkspaceWebSearchKeys>(workspacePath(workspaceId, keyId), {
      method: "PATCH",
      body: JSON.stringify({ is_default: true }),
    }),
  )
}

/**
 * Remove the workspace's pin and turned-off marks on the given keys, so it
 * follows its organization's choice again. One call per key, since the API
 * resets one key at a time; the last answer is the listing after all of them.
 */
export function useResetWorkspaceWebSearchKeys() {
  return useWorkspaceKeysWrite(async (workspaceId, keyIds) => {
    let listing: WorkspaceWebSearchKeys | undefined
    for (const keyId of keyIds) {
      listing = await apiFetch<WorkspaceWebSearchKeys>(
        workspacePath(workspaceId, keyId),
        { method: "DELETE" },
      )
    }
    return (
      listing ??
      (await apiFetch<WorkspaceWebSearchKeys>(workspaceListPath(workspaceId)))
    )
  })
}
