import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  AcceptedPricingSnapshot,
  CreateOrganizationPricingOverride,
  OrganizationPricingOverride,
  PricingDriftRow,
  PricingRefreshPreview,
  PricingResponse,
  SetPricingRequest,
  UpdateOrganizationPricingOverride,
} from "@/client"
import { ApiError, apiFetch, longRequestSignal } from "@/shared/api/client"
import { fetchAllPaged } from "@/shared/api/paging"
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

const PRICING_PAGE_SIZE = 1000

// Cap the walk so a backend or proxy that ignores `skip` (returning a full page
// every time) can't spin this into an unbounded request loop. 100 pages is 100k
// rows, far beyond any realistic price history.
const PRICING_MAX_PAGES = 100

async function fetchAllPricing(): Promise<PricingResponse[]> {
  const all: PricingResponse[] = []
  for (let page = 0; page < PRICING_MAX_PAGES; page += 1) {
    const rows = await apiFetch<PricingResponse[]>(
      `/pricing?skip=${page * PRICING_PAGE_SIZE}&limit=${PRICING_PAGE_SIZE}`,
    )
    all.push(...rows)
    if (rows.length < PRICING_PAGE_SIZE) {
      break
    }
  }
  return all
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

export function useOrganizationPricing(enabled = true) {
  return useQuery({
    queryKey: [ORGANIZATION_PRICING],
    // Paged through rather than read in one shot: the endpoint caps `limit`
    // server-side and the table grows a row per model per period, so a long-lived
    // organization would otherwise have its oldest overrides silently truncated.
    // `fetchAllPaged` carries the same hard page cap the rest of the tenancy
    // surface uses, so a backend that ignored `skip` cannot spin this.
    queryFn: () =>
      fetchAllPaged<OrganizationPricingOverride>("/organizations/me/pricing"),
    staleTime: 60_000,
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
