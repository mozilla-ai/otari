import {
  keepPreviousData,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query"
import type {
  AcceptedPricingSnapshot,
  CreateOrganizationPricingOverride,
  CurrentPricingPage,
  OrganizationPricingOverride,
  OrganizationPricingOverrides,
  PricingDriftRow,
  PricingRefreshPreview,
  PricingResponse,
  SetPricingRequest,
  UpdateOrganizationPricingOverride,
} from "@/client"
import { ApiError, apiFetch, longRequestSignal } from "@/shared/api/client"
import { fetchAllRows } from "@/shared/api/paging"
import {
  CATALOG,
  MODELS,
  NO_RETRY,
  ORGANIZATION_PRICING,
  PRICING,
  PRICING_DRIFT,
  PRICING_PENDING,
  PRICING_SNAPSHOTS,
  PROVIDERS,
} from "@/shared/api/queryKeys"

// The update the scheduled refresh has left for review, or null when there is
// none: the gateway answers 404 for the common case and that is not an error
// here. Operator-only, so `enabled` is the caller's gate.
export function usePendingPricingRefresh(enabled = true) {
  return useQuery({
    ...NO_RETRY,
    queryKey: PRICING_PENDING,
    queryFn: async (): Promise<PricingRefreshPreview | null> => {
      try {
        return await apiFetch<PricingRefreshPreview>("/pricing/refresh/pending")
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) return null
        throw error
      }
    },
    staleTime: 60_000,
    enabled,
  })
}

export function usePricingSnapshots(enabled = true) {
  return useQuery({
    ...NO_RETRY,
    queryKey: PRICING_SNAPSHOTS,
    queryFn: () => apiFetch<AcceptedPricingSnapshot[]>("/pricing/snapshots"),
    staleTime: 60_000,
    enabled,
  })
}

// Every stored rate against today's default. Resolves each key through
// genai-prices gateway-side, so it is kept warm like the other fan-out reads.
export function usePricingDrift(enabled = true) {
  return useQuery({
    ...NO_RETRY,
    queryKey: PRICING_DRIFT,
    queryFn: () => apiFetch<PricingDriftRow[]>("/pricing/drift"),
    staleTime: 60_000,
    enabled,
  })
}

const fetchAllPricing = () => fetchAllRows<PricingResponse>("/pricing")

/**
 * The rate one model is metered at, or null where it has none.
 *
 * The price editor is reachable from Models with a key the price table is not
 * showing, so it cannot resolve the row out of the page it happens to be on.
 * A 404 is the ordinary answer for an unpriced key, not an error.
 */
export function useModelPricing(modelKey: string | null) {
  return useQuery({
    ...NO_RETRY,
    queryKey: [PRICING, "one", modelKey],
    queryFn: async (): Promise<PricingResponse | null> => {
      try {
        return await apiFetch<PricingResponse>(
          `/pricing/${encodeURIComponent(modelKey ?? "")}`,
        )
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) return null
        throw error
      }
    },
    enabled: Boolean(modelKey),
  })
}

/**
 * One page of the rate each model is metered at now.
 *
 * `/pricing` answers the stored history (one row per `effective_at`), so a page
 * of it is a page of revisions rather than of models, and it carries no total to
 * put under a table. `/pricing/current` answers one row per key with a count,
 * which is what lets the table page instead of reading the collection.
 */
export function useCurrentPricing(page: number, pageSize: number) {
  return useQuery({
    queryKey: [PRICING, "current", page, pageSize],
    queryFn: () =>
      apiFetch<CurrentPricingPage>(
        `/pricing/current?skip=${page * pageSize}&limit=${pageSize}`,
      ),
    placeholderData: keepPreviousData,
  })
}

export function usePricing(enabled = true) {
  return useQuery({
    queryKey: [PRICING],
    queryFn: fetchAllPricing,
    enabled,
  })
}

export function useSetPricing() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: SetPricingRequest) =>
      apiFetch<PricingResponse>("/pricing", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [PRICING] })
      void queryClient.invalidateQueries({ queryKey: [MODELS] })
      void queryClient.invalidateQueries({ queryKey: [CATALOG] })
    },
  })
}

export function useDeletePricing() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (modelKey: string) =>
      apiFetch<void>(`/pricing/${encodeURIComponent(modelKey)}`, {
        method: "DELETE",
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [PRICING] })
      void queryClient.invalidateQueries({ queryKey: [MODELS] })
      void queryClient.invalidateQueries({ queryKey: [CATALOG] })
    },
  })
}

// Long deadline: this fetches the upstream snapshot and diffs it against every
// priced model, so it scales with the pricing table rather than with one hop.
export function usePreviewPricingRefresh() {
  return useMutation({
    mutationFn: () =>
      apiFetch<PricingRefreshPreview>("/pricing/refresh", {
        method: "POST",
        signal: longRequestSignal(),
      }),
  })
}

export function useConfirmPricingRefresh() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () => apiFetch("/pricing/refresh/confirm", { method: "POST" }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [PRICING] })
      void queryClient.invalidateQueries({ queryKey: [MODELS] })
      void queryClient.invalidateQueries({ queryKey: [CATALOG] })
      void queryClient.invalidateQueries({ queryKey: [PROVIDERS] })
    },
  })
}

// Rejecting also clears anything the scheduled check had left for review, so
// the pending read is refetched rather than left offering a review of nothing.
export function useRejectPricingRefresh() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () =>
      apiFetch<void>("/pricing/refresh/reject", { method: "POST" }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: PRICING_PENDING })
    },
  })
}

// The keys endpoint caps `limit` at 1000 server-side; page through it (capped like
// pricing) so a gateway with many keys can't have rows silently vanish from the
// table, and a backend that ignores `skip` can't spin an unbounded loop.

// ---------------------------------------------------------------------------
// Per-organization rate overrides
//
// A second, narrower price list above the deployment one (`usePricing` above).
// A model with no override here is priced by that list, so the two are read
// together on the page and never merged in the cache: an override is a row an
// operator manages, not a variant of a deployment price.
//
// Any member may read; only an owner or admin may write, which the server
// enforces and `canManage` mirrors so a refused control is disabled rather
// than offered.
// ---------------------------------------------------------------------------

/**
 * One page of the organization's rate overrides, with the total.
 *
 * The table grows a row per model per period, so reading it whole was a walk
 * that got longer for the life of the organization (otari#1420). The endpoint
 * answers the tenancy `{data, count}` envelope, so the page and the total both
 * come from the server.
 */
export function useOrganizationPricing(
  page: number,
  pageSize: number,
  enabled = true,
) {
  return useQuery({
    queryKey: [ORGANIZATION_PRICING, page, pageSize],
    queryFn: () =>
      apiFetch<OrganizationPricingOverrides>(
        `/organizations/me/pricing?skip=${page * pageSize}&limit=${pageSize}`,
      ),
    staleTime: 60_000,
    placeholderData: keepPreviousData,
    enabled,
  })
}

// MODELS is invalidated alongside, as the deployment pricing mutations do: the
// catalog carries each model's effective price, so a new override changes what
// that page shows.
function invalidateOrganizationPricing(
  queryClient: ReturnType<typeof useQueryClient>,
) {
  void queryClient.invalidateQueries({ queryKey: [ORGANIZATION_PRICING] })
  void queryClient.invalidateQueries({ queryKey: [MODELS] })
  void queryClient.invalidateQueries({ queryKey: [CATALOG] })
}

export function useCreateOrganizationPricing() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateOrganizationPricingOverride) =>
      apiFetch<OrganizationPricingOverride>("/organizations/me/pricing", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => invalidateOrganizationPricing(queryClient),
  })
}

// PUT, not PATCH: the endpoint replaces the row, so an omitted optional rate is
// cleared rather than inherited. The form therefore always sends every field.
export function useReplaceOrganizationPricing() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      id,
      body,
    }: {
      id: string
      body: UpdateOrganizationPricingOverride
    }) =>
      apiFetch<OrganizationPricingOverride>(
        `/organizations/me/pricing/${encodeURIComponent(id)}`,
        { method: "PUT", body: JSON.stringify(body) },
      ),
    onSuccess: () => invalidateOrganizationPricing(queryClient),
  })
}

export function useDeleteOrganizationPricing() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/organizations/me/pricing/${encodeURIComponent(id)}`, {
        method: "DELETE",
      }),
    onSuccess: () => invalidateOrganizationPricing(queryClient),
  })
}
