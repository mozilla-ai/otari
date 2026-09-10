import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  DashboardBuild,
  DeploymentAdminAccess,
  DeploymentUser,
  GatewayHealth,
  UpdateDeploymentUserRequest,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import { fetchAllPaged } from "@/shared/api/paging"
import {
  BUILD,
  BUILD_POLL_MS,
  DEPLOYMENT_ADMIN,
  HEALTH,
  HEALTH_POLL_MS,
  NO_RETRY,
  ORGANIZATION_MEMBERS,
} from "@/shared/api/queryKeys"

export function useDashboardBuild() {
  return useQuery({
    queryKey: [BUILD],
    queryFn: () => apiFetch<DashboardBuild>("/dashboard-build.json"),
    refetchInterval: BUILD_POLL_MS,
    // A tab left open in the background is the one most likely to be stale, so
    // check again the moment someone comes back to it.
    refetchOnWindowFocus: true,
    staleTime: 0,
    // A failed check is not worth reporting: the tab keeps working, and the next
    // poll retries anyway.
    retry: false,
  })
}

/**
 * Whether this gateway is answering, and whether it can reach its control plane.
 *
 * `/health` is public and served in both modes, which is what makes it the one
 * read a hybrid gateway's landing page can make: it hosts no management API, so
 * every other endpoint the dashboard knows is a 404 there. A failure is the
 * answer here rather than an error to retry past, so the page can say the
 * gateway is not responding on the first attempt instead of ~three requests
 * later.
 */
export function useGatewayHealth() {
  return useQuery({
    ...NO_RETRY,
    queryKey: [HEALTH],
    queryFn: () => apiFetch<GatewayHealth>("/health"),
    refetchInterval: HEALTH_POLL_MS,
    // A tab left open in the background holds the stalest answer of all, so ask
    // again the moment someone looks at it.
    refetchOnWindowFocus: true,
    staleTime: 0,
  })
}

// Every model the configured credentials can reach, per provider. Distinct from
// useModels: that is the catalog served to API callers (curated by
// model_discovery, aliases listed, targets withheld), while this is what an
// operator could pick from. A provider that failed is reported rather than
// dropped, so the picker can say why a list is empty.
//
// Live provider calls, cached gateway-side; kept fresh for the length of a
// session rather than refetched per open, since the set of models a key can
// reach does not move minute to minute.
//
// `enabled` is the `useToolSettings` composition: the read is
// deployment-operator-only, so a tenant-facing page declines to ask.

// ---------------------------------------------------------------------------
// Deployment administration
//
// The one management surface scoped to the deployment rather than to an
// organization. `useDeploymentAdminAccess` is its gate, which is why it answers
// 200 with a boolean where the rest of the prefix refuses a non-operator with
// 404: a gate cannot be built on a failed request, and hiding a destination
// grants nothing either way.
//
// Read by the deployment accounts page alone, to word its own refusal, and it
// holds that decision until this lands. Nothing else asks: the rail and the
// Overview index need the answer before their first paint, so they take
// `deployment_operator` off the organization context, which is the same
// server-side predicate on a read the shell already makes (otari#836).
// ---------------------------------------------------------------------------

export function useDeploymentAdminAccess() {
  return useQuery({
    queryKey: [DEPLOYMENT_ADMIN, "access"],
    queryFn: () =>
      apiFetch<DeploymentAdminAccess>("/admin/access").then(
        (body) => body.granted,
      ),
    staleTime: 60_000,
    // No `enabled` parameter: the one caller sits behind a route declaring
    // `surface: "admin"`, so a deployment that does not host `/admin` never
    // renders it and the 404 never becomes a second reading of `surfaces`.
    // Deliberately not inferred from a 404 here either, which is the scattered
    // mode check the surface axis replaced; the ordinary retry policy applies,
    // and a failure is a failure.
  })
}

export function useDeploymentUsers(enabled = true) {
  return useQuery({
    queryKey: [DEPLOYMENT_ADMIN, "users"],
    queryFn: () => fetchAllPaged<DeploymentUser>("/admin/users"),
    staleTime: 60_000,
    enabled,
  })
}

export function useUpdateDeploymentUser() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      id,
      body,
    }: {
      id: string
      body: UpdateDeploymentUserRequest
    }) =>
      apiFetch<DeploymentUser>(`/admin/users/${encodeURIComponent(id)}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [DEPLOYMENT_ADMIN] })
      // Deactivating an account ends its sessions and changes what the
      // organization roster may offer it, so the tenancy reads go stale with it.
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATION_MEMBERS] })
    },
  })
}
