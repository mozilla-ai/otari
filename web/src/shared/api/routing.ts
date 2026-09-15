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
import { useOrganizationContext } from "@/shared/api/organizations"
import {
  ALIASES,
  CATALOG,
  MODELS,
  ROUTER_STATUS,
  ROUTING_POLICIES,
} from "@/shared/api/queryKeys"

// Which of the two routing surfaces this caller may read, the `useKeysScope`
// construction over the same pair of endpoints. `/routing/policies` and
// `/aliases` are deployment-wide and refuse anyone who does not operate the
// deployment; the `/organizations/me/*` two answer the same shapes for the
// caller's own organization (otari-ai#1942, otari-ai#1969). The bases are part
// of every query key they feed, so a demotion cannot serve the wider list from
// cache, and an errored context falls through to the narrower surface.
export function useRoutingScope(): {
  policies: string
  aliases: string
  isReady: boolean
} {
  const context = useOrganizationContext()
  const isDeploymentWide = context.data?.deployment_operator === true
  return {
    policies: isDeploymentWide
      ? "/routing/policies"
      : "/organizations/me/routing-policies",
    aliases: isDeploymentWide ? "/aliases" : "/organizations/me/aliases",
    isReady: context.isSuccess || context.isError,
  }
}

function inWorkspace(base: string, workspaceId?: string): string {
  return workspaceId
    ? `${base}?workspace_id=${encodeURIComponent(workspaceId)}`
    : base
}

// The scope query a deployment-wide delete carries. Only a null or absent
// `userId` means workspace-wide, checked explicitly because "" is a legal user
// id and treating it as workspace-wide would delete the wrong row.
function deploymentScope(
  userId?: string | null,
  workspaceId?: string | null,
): string {
  const parts: string[] = []
  if (userId != null) parts.push(`user_id=${encodeURIComponent(userId)}`)
  if (workspaceId != null)
    parts.push(`workspace_id=${encodeURIComponent(workspaceId)}`)
  return parts.length === 0 ? "" : `?${parts.join("&")}`
}

// The workspace is part of the key, not just the request, so switching
// workspaces refetches rather than serving the previous one's rows. An unset id
// keeps the whole-scope view, which is what a caller who belongs to no
// workspace, and every read that wants the catalog rather than one workspace's
// management list, still wants.
export function useAliases(workspaceId?: string) {
  const scope = useRoutingScope()
  return useQuery({
    queryKey: [ALIASES, scope.aliases, workspaceId ?? null],
    queryFn: () =>
      apiFetch<AliasResponse[]>(inWorkspace(scope.aliases, workspaceId)),
    staleTime: 60_000,
    enabled: scope.isReady,
  })
}

export function useRoutingPolicies(workspaceId?: string) {
  const scope = useRoutingScope()
  return useQuery({
    queryKey: [ROUTING_POLICIES, scope.policies, workspaceId ?? null],
    queryFn: () =>
      apiFetch<RoutingPolicyResponse[]>(
        inWorkspace(scope.policies, workspaceId),
      ),
    staleTime: 60_000,
    enabled: scope.isReady,
  })
}

export function useSetRoutingPolicy() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: SetRoutingPolicyRequest) =>
      apiFetch<RoutingPolicyResponse>("/routing/policies", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ROUTING_POLICIES] })
      // A policy is listed as a model, so the catalog changes too.
      void queryClient.invalidateQueries({ queryKey: [MODELS] })
      void queryClient.invalidateQueries({ queryKey: [CATALOG] })
    },
  })
}

export function useDeleteRoutingPolicy() {
  const queryClient = useQueryClient()
  return useMutation({
    // Scoped like an alias delete: the same name can exist globally and per user,
    // so a delete must say which. Only a null/absent userId means global, checked
    // explicitly because "" is a legal user id. The workspace travels too, or the
    // route falls back to the deployment's default one and deletes a row the page
    // is not showing.
    mutationFn: ({
      name,
      userId,
      workspaceId,
    }: {
      name: string
      userId?: string | null
      workspaceId?: string | null
    }) =>
      apiFetch<void>(
        `/routing/policies/${encodeURIComponent(name)}${deploymentScope(userId, workspaceId)}`,
        { method: "DELETE" },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ROUTING_POLICIES] })
      void queryClient.invalidateQueries({ queryKey: [MODELS] })
      void queryClient.invalidateQueries({ queryKey: [CATALOG] })
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
 * Both list keys carry their surface and their workspace as trailing segments,
 * so invalidating the head of each covers whichever one this caller is reading.
 */
function invalidateRoutingLists(
  queryClient: ReturnType<typeof useQueryClient>,
) {
  void queryClient.invalidateQueries({ queryKey: [ROUTING_POLICIES] })
  void queryClient.invalidateQueries({ queryKey: [ALIASES] })
  // A policy and an alias are both listed as models, so the catalog changes too.
  void queryClient.invalidateQueries({ queryKey: [MODELS] })
  void queryClient.invalidateQueries({ queryKey: [CATALOG] })
}

export function useSetOrganizationRoutingPolicy() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: SetRoutingPolicyRequest & { workspace_id: string }) =>
      apiFetch<RoutingPolicyResponse>("/organizations/me/routing-policies", {
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
        `/organizations/me/routing-policies/${encodeURIComponent(name)}?workspace_id=${encodeURIComponent(workspaceId)}`,
        { method: "DELETE" },
      ),
    onSuccess: () => invalidateRoutingLists(queryClient),
  })
}

export function useCreateOrganizationAlias() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateAliasRequest & { workspace_id: string }) =>
      apiFetch<AliasResponse>("/organizations/me/aliases", {
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
        `/organizations/me/aliases/${encodeURIComponent(name)}?workspace_id=${encodeURIComponent(workspaceId)}`,
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
      apiFetch<ExplainPolicyResponse>("/routing/policies/explain", {
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
        `/routing/status?user_id=${encodeURIComponent(userId ?? "")}`,
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
      apiFetch<RankCandidatesResponse>("/routing/preferences/rank", {
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
      apiFetch<AliasResponse>("/aliases", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ALIASES] })
      // An alias is listed as a model, so the catalog changes too.
      void queryClient.invalidateQueries({ queryKey: [MODELS] })
      void queryClient.invalidateQueries({ queryKey: [CATALOG] })
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
      workspaceId,
    }: {
      name: string
      userId?: string | null
      workspaceId?: string | null
    }) =>
      apiFetch<void>(
        `/aliases/${encodeURIComponent(name)}${deploymentScope(userId, workspaceId)}`,
        { method: "DELETE" },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ALIASES] })
      void queryClient.invalidateQueries({ queryKey: [MODELS] })
      void queryClient.invalidateQueries({ queryKey: [CATALOG] })
    },
  })
}

// `enabled` is the `useToolSettings` composition: the read is
// deployment-operator-only, so a tenant-facing page declines to ask.
