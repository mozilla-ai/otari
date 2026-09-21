import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  Budget,
  BudgetResetLog,
  CreateBudgetRequest,
  CreateOrganizationBudget,
  CreateOrganizationSpendCeiling,
  CreateScopedBudgetRequest,
  OrganizationBudget,
  OrganizationContext,
  OrganizationSpendCeiling,
  OrganizationSpendCeilings,
  ScopedBudget,
  UpdateBudgetRequest,
  UpdateOrganizationBudget,
  UpdateOrganizationSpendCeiling,
  UpdateScopedBudgetRequest,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import { useOrganizationContext } from "@/shared/api/organizations"
import { fetchAllPaged, fetchAllRows } from "@/shared/api/paging"
import {
  BUDGETS,
  ORGANIZATION_BUDGETS,
  ORGANIZATION_CONTEXT,
  ORGANIZATION_SPEND_CEILINGS,
  SCOPED_BUDGETS,
} from "@/shared/api/queryKeys"

const fetchAllBudgets = () => fetchAllRows<Budget>("/budgets")

// `enabled` is for a page that composes this deployment-wide read into a
// tenant-scoped one: since #821 it answers 403 to anyone who does not operate
// the deployment, so a caller who knows they are a tenant declines to ask rather
// than surfacing the refusal (otari#838).
export function useBudgets(enabled = true) {
  return useQuery({
    queryKey: [BUDGETS],
    queryFn: fetchAllBudgets,
    staleTime: 60_000,
    enabled,
  })
}

// Per-user reset history for one budget. Enabled only once a budget id is set
// (the drill-down is opened), so the query does not fire for the whole list.
export function useBudgetResetLogs(budgetId: string | null) {
  return useQuery({
    queryKey: [BUDGETS, budgetId, "reset-logs"],
    queryFn: () =>
      apiFetch<BudgetResetLog[]>(
        `/budgets/${encodeURIComponent(budgetId as string)}/reset-logs`,
      ),
    enabled: budgetId !== null,
    staleTime: 60_000,
  })
}

export function useCreateBudget() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateBudgetRequest) =>
      apiFetch<Budget>("/budgets", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [BUDGETS] }),
  })
}

export function useUpdateBudget() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: UpdateBudgetRequest }) =>
      apiFetch<Budget>(`/budgets/${encodeURIComponent(id)}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [BUDGETS] }),
  })
}

export function useDeleteBudget() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/budgets/${encodeURIComponent(id)}`, {
        method: "DELETE",
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [BUDGETS] }),
  })
}

// The tenancy-scoped ceilings, which are a different mechanism from the budgets
// above rather than a view over them: each row carries its own counters, so one
// row is a pooled cap over whatever its scope names. See `client/index.ts`.
//
const fetchAllScopedBudgets = () =>
  fetchAllRows<ScopedBudget>("/scoped-budgets")

// Gated for the same reason as `useBudgets` above.
export function useScopedBudgets(enabled = true) {
  return useQuery({
    queryKey: [SCOPED_BUDGETS],
    queryFn: fetchAllScopedBudgets,
    staleTime: 60_000,
    enabled,
  })
}

export function useCreateScopedBudget() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateScopedBudgetRequest) =>
      apiFetch<ScopedBudget>("/scoped-budgets", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [SCOPED_BUDGETS] }),
  })
}

export function useUpdateScopedBudget() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      id,
      body,
    }: {
      id: string
      body: UpdateScopedBudgetRequest
    }) =>
      apiFetch<ScopedBudget>(`/scoped-budgets/${encodeURIComponent(id)}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [SCOPED_BUDGETS] }),
  })
}

export function useDeleteScopedBudget() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/scoped-budgets/${encodeURIComponent(id)}`, {
        method: "DELETE",
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [SCOPED_BUDGETS] }),
  })
}

// The users endpoint caps `limit` at 1000 server-side; page through it (capped
// like keys/budgets) so a gateway with many users can't have rows silently
// vanish, and a backend that ignores `skip` can't spin an unbounded loop.

// ---------------------------------------------------------------------------
// The organization's own budgets and spend ceilings
//
// The tenant-scoped counterpart to `useBudgets` / `useScopedBudgets` above,
// which read `/budgets` and `/scoped-budgets` and have answered 403 to
// anyone who does not operate the deployment since #821. These read the
// caller's own organization instead, and are owner-or-admin on both halves:
// unlike the rate overrides, a cap is a statement about what colleagues may
// spend, so the roles matrix has it Hidden for a member (otari-ai#1943).
//
// A ceiling naming a budget the organization does not own reports `manageable`
// false. Those are what the otari-ai cutover writes, and they are listed rather
// than hidden because they are enforcing today; the page offers to move one onto
// one of the organization's own budgets instead of pretending it can edit the
// figure.
// ---------------------------------------------------------------------------

export function useOrganizationBudgets(enabled = true) {
  return useQuery({
    queryKey: [ORGANIZATION_BUDGETS],
    // Paged through with the tenancy walker rather than read in one shot, for
    // the reason `useOrganizationPricing` gives: the endpoint caps `limit`
    // server-side, and the cap is what would silently truncate a long-lived
    // organization's list.
    queryFn: () =>
      fetchAllPaged<OrganizationBudget>("/organizations/me/budgets"),
    staleTime: 60_000,
    enabled,
  })
}

/**
 * One page of the organization's spend ceilings, with the total.
 *
 * Read whole until otari#1420: the Overview also read it, for a worst-case
 * aggregate, and a second reader wanting every row is what kept this a walk.
 * That reader moved to the summary endpoint in otari#1425, so the table is the
 * only one left and can ask for the page it shows.
 */
export function useOrganizationSpendCeilings(
  page: number,
  pageSize: number,
  enabled = true,
) {
  const queryClient = useQueryClient()
  // The context the caller's `enabled` was read from, whatever it read off it.
  const context = useOrganizationContext().data
  return useQuery({
    // The organization is part of the key, not only of the request, which
    // carries it implicitly: the server scopes this read by the session's
    // active organization, so a walk still in flight when the caller switches
    // is answered about the organization just left. Keyed per organization, it
    // lands under the one it asked about rather than under the one now on
    // screen. `invalidateOrganizationSpend` matches on the head, so the extra
    // segment costs it nothing, and a context that names no organization keys
    // as `null` rather than taking the page down over a cache entry.
    queryKey: [
      ORGANIZATION_SPEND_CEILINGS,
      context?.organization?.id ?? null,
      page,
      pageSize,
    ],
    queryFn: () =>
      apiFetch<OrganizationSpendCeilings>(
        `/organizations/me/spend-ceilings?skip=${page * pageSize}&limit=${pageSize}`,
      ),
    staleTime: 60_000,
    // Kept across a page change and dropped across an organization change.
    // `keepPreviousData` alone answers the new organization's key with the old
    // organization's rows until the fetch lands, which is one tenant's spend on
    // another tenant's screen, briefly and for no reason.
    placeholderData: (previous, previousQuery) =>
      previousQuery?.queryKey[1] === (context?.organization?.id ?? null)
        ? previous
        : undefined,
    // A callback, because this is the one read here that a *role* opens, and a
    // role moves under a mounted query. Switching organization invalidates
    // everything cached, and React Query resolves a plain `enabled` from the
    // render before, so an owner or admin here who is a member there refetched
    // this owners-and-admins-only read under the role just left and the page
    // reported the refusal (otari#1300). A callback is resolved when the
    // refetch is decided, by which point the switch has written the new
    // context, so the read is withheld rather than made and apologized for.
    // The gate reopens on the caller's next render, which is where `enabled`
    // is worked out again from the context now in its hands.
    enabled: () =>
      enabled &&
      queryClient.getQueryData<OrganizationContext>(ORGANIZATION_CONTEXT) ===
        context,
  })
}

// Both keys move together on every write. A budget's figure is read *through*
// the budget by every ceiling naming it, so changing one changes what those
// ceilings report; and creating a ceiling changes a budget's `ceiling_count`,
// which is what makes its delete refuse.
function invalidateOrganizationSpend(
  queryClient: ReturnType<typeof useQueryClient>,
) {
  void queryClient.invalidateQueries({ queryKey: [ORGANIZATION_BUDGETS] })
  void queryClient.invalidateQueries({
    queryKey: [ORGANIZATION_SPEND_CEILINGS],
  })
}

export function useCreateOrganizationBudget() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateOrganizationBudget) =>
      apiFetch<OrganizationBudget>("/organizations/me/budgets", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => invalidateOrganizationSpend(queryClient),
  })
}

// PATCH, not PUT: an omitted field is left alone and an explicit null clears it,
// which is what lets the dialog send `max_budget: null` to take a budget back to
// uncapped without deleting it.
export function useUpdateOrganizationBudget() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      id,
      body,
    }: {
      id: string
      body: UpdateOrganizationBudget
    }) =>
      apiFetch<OrganizationBudget>(
        `/organizations/me/budgets/${encodeURIComponent(id)}`,
        { method: "PATCH", body: JSON.stringify(body) },
      ),
    onSuccess: () => invalidateOrganizationSpend(queryClient),
  })
}

export function useDeleteOrganizationBudget() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<{ message: string }>(
        `/organizations/me/budgets/${encodeURIComponent(id)}`,
        { method: "DELETE" },
      ),
    onSuccess: () => invalidateOrganizationSpend(queryClient),
  })
}

export function useCreateOrganizationSpendCeiling() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateOrganizationSpendCeiling) =>
      apiFetch<OrganizationSpendCeiling>("/organizations/me/spend-ceilings", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => invalidateOrganizationSpend(queryClient),
  })
}

export function useUpdateOrganizationSpendCeiling() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      id,
      body,
    }: {
      id: string
      body: UpdateOrganizationSpendCeiling
    }) =>
      apiFetch<OrganizationSpendCeiling>(
        `/organizations/me/spend-ceilings/${encodeURIComponent(id)}`,
        { method: "PATCH", body: JSON.stringify(body) },
      ),
    onSuccess: () => invalidateOrganizationSpend(queryClient),
  })
}

export function useDeleteOrganizationSpendCeiling() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<{ message: string }>(
        `/organizations/me/spend-ceilings/${encodeURIComponent(id)}`,
        { method: "DELETE" },
      ),
    onSuccess: () => invalidateOrganizationSpend(queryClient),
  })
}
