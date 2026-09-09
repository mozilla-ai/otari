import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  Budget,
  BudgetResetLog,
  CreateBudgetRequest,
  CreateOrganizationBudget,
  CreateOrganizationSpendCeiling,
  CreateScopedBudgetRequest,
  OrganizationBudget,
  OrganizationSpendCeiling,
  ScopedBudget,
  UpdateBudgetRequest,
  UpdateOrganizationBudget,
  UpdateOrganizationSpendCeiling,
  UpdateScopedBudgetRequest,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import { fetchAllPaged } from "@/shared/api/paging"
import {
  BUDGETS,
  ORGANIZATION_BUDGETS,
  ORGANIZATION_SPEND_CEILINGS,
  SCOPED_BUDGETS,
} from "@/shared/api/queryKeys"

const BUDGETS_PAGE_SIZE = 1000
const BUDGETS_MAX_PAGES = 100

async function fetchAllBudgets(): Promise<Budget[]> {
  const all: Budget[] = []
  for (let page = 0; page < BUDGETS_MAX_PAGES; page += 1) {
    const rows = await apiFetch<Budget[]>(
      `/v1/budgets?skip=${page * BUDGETS_PAGE_SIZE}&limit=${BUDGETS_PAGE_SIZE}`,
    )
    all.push(...rows)
    if (rows.length < BUDGETS_PAGE_SIZE) {
      break
    }
  }
  return all
}

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
        `/v1/budgets/${encodeURIComponent(budgetId as string)}/reset-logs`,
      ),
    enabled: budgetId !== null,
    staleTime: 60_000,
  })
}

export function useCreateBudget() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateBudgetRequest) =>
      apiFetch<Budget>("/v1/budgets", {
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
      apiFetch<Budget>(`/v1/budgets/${encodeURIComponent(id)}`, {
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
      apiFetch<void>(`/v1/budgets/${encodeURIComponent(id)}`, {
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
// The list route returns a bare array (not the `Paged` envelope the tenancy
// routes use) and caps `limit` at 1000 server-side, so it pages like budgets and
// keys do, with the same guard against a backend that ignores `skip`.
const SCOPED_BUDGETS_PAGE_SIZE = 1000
const SCOPED_BUDGETS_MAX_PAGES = 100

async function fetchAllScopedBudgets(): Promise<ScopedBudget[]> {
  const all: ScopedBudget[] = []
  for (let page = 0; page < SCOPED_BUDGETS_MAX_PAGES; page += 1) {
    const rows = await apiFetch<ScopedBudget[]>(
      `/v1/scoped-budgets?skip=${page * SCOPED_BUDGETS_PAGE_SIZE}&limit=${SCOPED_BUDGETS_PAGE_SIZE}`,
    )
    all.push(...rows)
    if (rows.length < SCOPED_BUDGETS_PAGE_SIZE) {
      break
    }
  }
  return all
}

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
      apiFetch<ScopedBudget>("/v1/scoped-budgets", {
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
      apiFetch<ScopedBudget>(`/v1/scoped-budgets/${encodeURIComponent(id)}`, {
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
      apiFetch<void>(`/v1/scoped-budgets/${encodeURIComponent(id)}`, {
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
// which read `/v1/budgets` and `/v1/scoped-budgets` and have answered 403 to
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
      fetchAllPaged<OrganizationBudget>("/v1/organizations/me/budgets"),
    staleTime: 60_000,
    enabled,
  })
}

export function useOrganizationSpendCeilings(enabled = true) {
  return useQuery({
    queryKey: [ORGANIZATION_SPEND_CEILINGS],
    queryFn: () =>
      fetchAllPaged<OrganizationSpendCeiling>(
        "/v1/organizations/me/spend-ceilings",
      ),
    staleTime: 60_000,
    enabled,
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
      apiFetch<OrganizationBudget>("/v1/organizations/me/budgets", {
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
        `/v1/organizations/me/budgets/${encodeURIComponent(id)}`,
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
        `/v1/organizations/me/budgets/${encodeURIComponent(id)}`,
        { method: "DELETE" },
      ),
    onSuccess: () => invalidateOrganizationSpend(queryClient),
  })
}

export function useCreateOrganizationSpendCeiling() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateOrganizationSpendCeiling) =>
      apiFetch<OrganizationSpendCeiling>(
        "/v1/organizations/me/spend-ceilings",
        { method: "POST", body: JSON.stringify(body) },
      ),
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
        `/v1/organizations/me/spend-ceilings/${encodeURIComponent(id)}`,
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
        `/v1/organizations/me/spend-ceilings/${encodeURIComponent(id)}`,
        { method: "DELETE" },
      ),
    onSuccess: () => invalidateOrganizationSpend(queryClient),
  })
}
