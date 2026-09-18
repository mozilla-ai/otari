import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  ApiKey,
  CreateKeyRequest,
  CreateKeyResponse,
  CreateOwnKeyRequest,
  UpdateKeyRequest,
  UpdateOwnKeyRequest,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import { useOrganizationContext } from "@/shared/api/organizations"
import { fetchAllRows } from "@/shared/api/paging"
import { KEYS } from "@/shared/api/queryKeys"

// Which of the two key surfaces this caller may act on.
//
// `/keys` is deployment-wide and refuses anyone who does not operate the
// deployment; `/organizations/me/keys` serves the caller's own keys, in
// workspaces they belong to (otari-ai#1941). Both answer identical shapes, so
// the key hooks below differ only in the prefix they ask. Same construction as
// `useUsageScope` above, for the same reasons: the organization context is what
// the shell already reads, an errored context falls through to the narrower
// surface, and the base is part of every query key it feeds.
export function useKeysScope(): {
  base: string
  isReady: boolean
  isDeploymentWide: boolean
} {
  const context = useOrganizationContext()
  const isDeploymentWide = context.data?.deployment_operator === true
  return {
    base: isDeploymentWide ? "/keys" : "/organizations/me/keys",
    isReady: context.isSuccess || context.isError,
    isDeploymentWide,
  }
}

function fetchAllKeys(base: string, workspaceId?: string): Promise<ApiKey[]> {
  return fetchAllRows<ApiKey>(
    base,
    workspaceId ? { workspace_id: workspaceId } : undefined,
  )
}

// The workspace is part of the key, not just the request: switching workspaces
// has to refetch rather than serve the previous one's keys from cache. Same for
// the two below. An unset id keeps the whole-scope view, which is what the
// organization context and a deployment with no workspace selected still want.
// `enabled` is for a caller that reads keys for something other than showing
// them, and so should not pay the walk until it is on screen.
export function useKeys(workspaceId?: string, enabled = true) {
  const scope = useKeysScope()
  return useQuery({
    queryKey: [KEYS, scope.base, workspaceId ?? null],
    queryFn: () => fetchAllKeys(scope.base, workspaceId),
    enabled: scope.isReady && enabled,
    staleTime: 60_000,
  })
}

// Create returns the plaintext key exactly once (in `key`); the caller reveals it
// and must never write the response into the query cache. The member surface's
// body is a subset of the operator's (no owner, no budget exemption), which is
// the page's branch to build; the union keeps a member body from being forced
// to carry fields its endpoint refuses to honor.
export function useCreateKey() {
  const queryClient = useQueryClient()
  const scope = useKeysScope()
  return useMutation({
    mutationFn: (body: CreateKeyRequest | CreateOwnKeyRequest) =>
      apiFetch<CreateKeyResponse>(scope.base, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: [KEYS] }),
  })
}

export function useUpdateKey() {
  const queryClient = useQueryClient()
  const scope = useKeysScope()
  return useMutation({
    mutationFn: ({
      id,
      body,
    }: {
      id: string
      body: UpdateKeyRequest | UpdateOwnKeyRequest
    }) =>
      apiFetch<ApiKey>(`${scope.base}/${encodeURIComponent(id)}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: [KEYS] }),
  })
}

// Regenerate: a new secret for the same key row. The old secret stops working
// immediately. Returns the new plaintext once, like create.
export function useRotateKey() {
  const queryClient = useQueryClient()
  const scope = useKeysScope()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<CreateKeyResponse>(
        `${scope.base}/${encodeURIComponent(id)}/rotate`,
        {
          method: "POST",
        },
      ),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: [KEYS] }),
  })
}

export function useDeleteKey() {
  const queryClient = useQueryClient()
  const scope = useKeysScope()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`${scope.base}/${encodeURIComponent(id)}`, {
        method: "DELETE",
      }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: [KEYS] }),
  })
}

// The budgets endpoint caps `limit` at 1000 server-side; page through it (capped
// like keys/pricing) so a gateway with many budgets can't have rows silently
// vanish, and a backend that ignores `skip` can't spin an unbounded loop.
