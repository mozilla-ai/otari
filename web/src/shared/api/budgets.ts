import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  Budget,
  BudgetResetLog,
  CreateBudgetRequest,
  CreateOrganizationBudget,
  CreateScopedBudgetRequest,
  OrganizationBudget,
  OrganizationContext,
  OrganizationSpendCeilings,
  ScopedBudget,
  UpdateBudgetRequest,
  UpdateOrganizationBudget,
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
// The organization's own budgets
//
// The tenant-scoped counterpart to `useBudgets` / `useScopedBudgets` above,
// which read `/budgets` and `/scoped-budgets` and have answered 403 to
// anyone who does not operate the deployment since #821. These read the
// caller's own organization instead, and are owner-or-admin: unlike the rate
// overrides, a cap is a statement about what colleagues may spend, so the roles
// matrix has it Hidden for a member (otari-ai#1943). Each budget carries the
// entities it applies to, so one read draws the page and seeds the form.
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
 * The ceilings applying one of the organization's budgets, with their spend.
 *
 * One page at the endpoint's maximum, which a budget cannot outgrow: its create
 * and update bodies cap the entities it applies to at the same number. Keyed per
 * organization, because the server scopes this by the session's active one.
 */
export function useOrganizationBudgetCeilings(budgetId: string) {
  const queryClient = useQueryClient()
  const context = useOrganizationContext().data
  return useQuery({
    queryKey: [
      ORGANIZATION_SPEND_CEILINGS,
      context?.organization?.id ?? null,
      budgetId,
    ],
    queryFn: () =>
      apiFetch<OrganizationSpendCeilings>(
        `/organizations/me/spend-ceilings?budget_id=${encodeURIComponent(budgetId)}&limit=1000`,
      ),
    staleTime: 60_000,
    // A callback, because this owners-and-admins-only read is opened by a role,
    // and a role moves under a mounted query (otari#1300): resolved when the
    // refetch is decided, it sees the context the switch has already written.
    enabled: () =>
      queryClient.getQueryData<OrganizationContext>(ORGANIZATION_CONTEXT) ===
      context,
  })
}

// A write moves the ceilings too: the detail page reads them, and a budget's
// figure, cycle and entities are what they report.
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
