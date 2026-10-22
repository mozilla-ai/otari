import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  ApiKey,
  CreateFederationRuleRequest,
  CreateKeyRequest,
  CreateKeyResponse,
  CreateOwnKeyRequest,
  FederationRule,
  UpdateKeyRequest,
  UpdateOwnKeyRequest,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import { useDeploymentOperator } from "@/shared/api/organizations"
import { fetchAllRows } from "@/shared/api/paging"
import { FEDERATION_RULES, KEYS, workspaceMeta } from "@/shared/api/queryKeys"

// Which of the two key surfaces this caller may act on.
//
// `/keys` is deployment-wide and refuses anyone who does not operate the
// deployment; `/organizations/me/keys` serves the caller's own keys, in
// workspaces they belong to (otari-ai#1941). Both answer identical shapes, so
// the key hooks below differ only in the prefix they ask. Same construction as
// `useUsageScope` above, for the same reasons: the organization context is what
// the shell already reads, an errored context takes the wider surface, and the
// base is part of every query key it feeds.
export function useKeysScope(): {
  base: string
  isReady: boolean
  isDeploymentWide: boolean
} {
  const { isOperator: isDeploymentWide, isSettled } = useDeploymentOperator()
  return {
    base: isDeploymentWide ? "/keys" : "/organizations/me/keys",
    isReady: isSettled,
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
    meta: workspaceMeta(workspaceId),
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

// The identity tokens that may stand in for one key's secret. Under the same
// prefix as the key: a member's rule on `/organizations/me/keys`, an operator's
// on `/keys`, which also says which issuers the gateway accepts for it.
function federationRulesPath(base: string, keyId: string): string {
  return `${base}/${encodeURIComponent(keyId)}/federation-rules`
}

export function useFederationRules(keyId: string) {
  const scope = useKeysScope()
  return useQuery({
    queryKey: [FEDERATION_RULES, scope.base, keyId],
    queryFn: () =>
      apiFetch<FederationRule[]>(federationRulesPath(scope.base, keyId)),
    enabled: scope.isReady,
  })
}

export function useAddFederationRule(keyId: string) {
  const queryClient = useQueryClient()
  const scope = useKeysScope()
  return useMutation({
    mutationFn: (body: CreateFederationRuleRequest) =>
      apiFetch<FederationRule>(federationRulesPath(scope.base, keyId), {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({
        queryKey: [FEDERATION_RULES, scope.base, keyId],
      }),
  })
}

export function useDeleteFederationRule(keyId: string) {
  const queryClient = useQueryClient()
  const scope = useKeysScope()
  return useMutation({
    mutationFn: (ruleId: string) =>
      apiFetch<void>(
        `${federationRulesPath(scope.base, keyId)}/${encodeURIComponent(ruleId)}`,
        { method: "DELETE" },
      ),
    onSuccess: () =>
      void queryClient.invalidateQueries({
        queryKey: [FEDERATION_RULES, scope.base, keyId],
      }),
  })
}

// The budgets endpoint caps `limit` at 1000 server-side; page through it (capped
// like keys/pricing) so a gateway with many budgets can't have rows silently
// vanish, and a backend that ignores `skip` can't spin an unbounded loop.
