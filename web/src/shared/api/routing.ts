import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  AliasResponse,
  CreateAliasRequest,
  ExplainPolicyRequest,
  ExplainPolicyResponse,
  RankCandidatesRequest,
  RankCandidatesResponse,
  RouterStatus,
  RoutingPolicyResponse,
  SetRoutingPolicyRequest,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import {
  ALIASES,
  MODELS,
  ORGANIZATION_ALIASES,
  ORGANIZATION_ROUTING_POLICIES,
  ROUTER_STATUS,
  ROUTING_POLICIES,
} from "@/shared/api/queryKeys"

export function useAliases(enabled = true) {
  return useQuery({
    queryKey: [ALIASES],
    queryFn: () => apiFetch<AliasResponse[]>("/v1/aliases"),
    staleTime: 60_000,
    enabled,
  })
}

// Unscoped for the same reason as `useAliases` above, `enabled` too.
export function useRoutingPolicies(enabled = true) {
  return useQuery({
    queryKey: [ROUTING_POLICIES],
    queryFn: () => apiFetch<RoutingPolicyResponse[]>("/v1/routing/policies"),
    staleTime: 60_000,
    enabled,
  })
}

// The routing policies in force where the caller may see: stored rows from
// their visible workspaces plus the deployment-wide config ones, the same
// response shape as `useRoutingPolicies`. This is the read a signed-in member
// gets (otari-ai#1942); the deployment-wide list above stays operator-only, so
// the Routing page picks between the two off `isDeploymentOperator`.
export function useOrganizationRoutingPolicies(enabled = true) {
  return useQuery({
    queryKey: [ORGANIZATION_ROUTING_POLICIES],
    queryFn: () =>
      apiFetch<RoutingPolicyResponse[]>(
        "/v1/organizations/me/routing-policies",
      ),
    staleTime: 60_000,
    enabled,
  })
}

// The aliases in force where the caller may see, the policies list's sibling
// over `model_aliases` and scoped the same way. This is the read a signed-in
// member gets (otari-ai#1969); the deployment-wide `useAliases` above stays
// operator-only, so the Routing page picks between the two off
// `isDeploymentOperator`.
export function useOrganizationAliases(enabled = true) {
  return useQuery({
    queryKey: [ORGANIZATION_ALIASES],
    queryFn: () => apiFetch<AliasResponse[]>("/v1/organizations/me/aliases"),
    staleTime: 60_000,
    enabled,
  })
}

export function useSetRoutingPolicy() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: SetRoutingPolicyRequest) =>
      apiFetch<RoutingPolicyResponse>("/v1/routing/policies", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ROUTING_POLICIES] })
      void queryClient.invalidateQueries({
        queryKey: [ORGANIZATION_ROUTING_POLICIES],
      })
      // A policy is listed as a model, so the catalog changes too.
      void queryClient.invalidateQueries({ queryKey: [MODELS] })
    },
  })
}

export function useDeleteRoutingPolicy() {
  const queryClient = useQueryClient()
  return useMutation({
    // Scoped like an alias delete: the same name can exist globally and per user,
    // so a delete must say which. Only a null/absent userId means global, checked
    // explicitly because "" is a legal user id.
    mutationFn: ({
      name,
      userId,
    }: {
      name: string
      userId?: string | null
    }) => {
      const scope =
        userId == null ? "" : `?user_id=${encodeURIComponent(userId)}`
      return apiFetch<void>(
        `/v1/routing/policies/${encodeURIComponent(name)}${scope}`,
        { method: "DELETE" },
      )
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ROUTING_POLICIES] })
      void queryClient.invalidateQueries({
        queryKey: [ORGANIZATION_ROUTING_POLICIES],
      })
      void queryClient.invalidateQueries({ queryKey: [MODELS] })
    },
  })
}

/**
 * The four tenant-scoped writers behind the Build pages' Edit for admins.
 *
 * Each is its operator sibling above with one difference the server insists on:
 * the workspace is named rather than defaulted, because the deployment's default
 * workspace is not the caller's organization's to write into (otari-ai#1969).
 * `user_id` has no counterpart here at all: an organization's entries are
 * workspace-wide.
 *
 * Both list keys are invalidated by each of them, because which of the two a
 * page is reading is a function of the caller's role rather than of the write.
 */
function invalidateRoutingLists(
  queryClient: ReturnType<typeof useQueryClient>,
) {
  void queryClient.invalidateQueries({ queryKey: [ROUTING_POLICIES] })
  void queryClient.invalidateQueries({
    queryKey: [ORGANIZATION_ROUTING_POLICIES],
  })
  void queryClient.invalidateQueries({ queryKey: [ALIASES] })
  void queryClient.invalidateQueries({ queryKey: [ORGANIZATION_ALIASES] })
  // A policy and an alias are both listed as models, so the catalog changes too.
  void queryClient.invalidateQueries({ queryKey: [MODELS] })
}

export function useSetOrganizationRoutingPolicy() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: SetRoutingPolicyRequest & { workspace_id: string }) =>
      apiFetch<RoutingPolicyResponse>("/v1/organizations/me/routing-policies", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => invalidateRoutingLists(queryClient),
  })
}

export function useDeleteOrganizationRoutingPolicy() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      name,
      workspaceId,
    }: {
      name: string
      workspaceId: string
    }) =>
      apiFetch<void>(
        `/v1/organizations/me/routing-policies/${encodeURIComponent(name)}?workspace_id=${encodeURIComponent(workspaceId)}`,
        { method: "DELETE" },
      ),
    onSuccess: () => invalidateRoutingLists(queryClient),
  })
}

export function useCreateOrganizationAlias() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateAliasRequest & { workspace_id: string }) =>
      apiFetch<AliasResponse>("/v1/organizations/me/aliases", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => invalidateRoutingLists(queryClient),
  })
}

export function useDeleteOrganizationAlias() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      name,
      workspaceId,
    }: {
      name: string
      workspaceId: string
    }) =>
      apiFetch<void>(
        `/v1/organizations/me/aliases/${encodeURIComponent(name)}?workspace_id=${encodeURIComponent(workspaceId)}`,
        { method: "DELETE" },
      ),
    onSuccess: () => invalidateRoutingLists(queryClient),
  })
}

/** Compile a policy (saved or draft) without dispatching anything.
 *
 *  A mutation rather than a query: it is an explicit "check this now" action on
 *  inputs the operator is editing, not cacheable server state. */
export function useExplainPolicy() {
  return useMutation({
    mutationFn: (body: ExplainPolicyRequest) =>
      apiFetch<ExplainPolicyResponse>("/v1/routing/policies/explain", {
        method: "POST",
        body: JSON.stringify(body),
      }),
  })
}

// --- Learned routing ------------------------------------------------------

/** How warm a user's routing memory is.
 *
 *  Keyed by user because warmth is per user: the records hold that user's
 *  prompts, so a global learned policy warms once per caller. Disabled until a
 *  user is chosen rather than defaulting to one, because "whose memory" has no
 *  sensible default.
 */
export function useRouterStatus(userId: string | null) {
  return useQuery({
    queryKey: [ROUTER_STATUS, userId],
    queryFn: () =>
      apiFetch<RouterStatus>(
        `/v1/routing/status?user_id=${encodeURIComponent(userId ?? "")}`,
      ),
    enabled: userId !== null && userId !== "",
    staleTime: 30_000,
  })
}

/** Record how well each candidate did, which is what the router later votes over. */
export function useRankCandidates() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: RankCandidatesRequest) =>
      apiFetch<RankCandidatesResponse>("/v1/routing/preferences/rank", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      // One more example may have crossed the seed count, which changes whether
      // the policy routes at all.
      void queryClient.invalidateQueries({ queryKey: [ROUTER_STATUS] })
    },
  })
}

export function useCreateAlias() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateAliasRequest) =>
      apiFetch<AliasResponse>("/v1/aliases", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ALIASES] })
      // An alias is listed as a model, so the catalog changes too.
      void queryClient.invalidateQueries({ queryKey: [MODELS] })
    },
  })
}

export function useDeleteAlias() {
  const queryClient = useQueryClient()
  return useMutation({
    // Scoped: the same name can exist globally and per user, so deleting one
    // must name which. Only a null/absent userId means global; the check is
    // explicit rather than truthy because "" is a legal user id, and treating it
    // as global would delete the wrong row.
    mutationFn: ({
      name,
      userId,
    }: {
      name: string
      userId?: string | null
    }) => {
      const scope =
        userId == null ? "" : `?user_id=${encodeURIComponent(userId)}`
      return apiFetch<void>(`/v1/aliases/${encodeURIComponent(name)}${scope}`, {
        method: "DELETE",
      })
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ALIASES] })
      void queryClient.invalidateQueries({ queryKey: [MODELS] })
    },
  })
}

// `enabled` is the `useToolSettings` composition: the read is
// deployment-operator-only, so a tenant-facing page declines to ask.
