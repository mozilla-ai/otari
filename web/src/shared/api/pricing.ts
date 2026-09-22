import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  CreateOrganizationPricingOverride,
  OrganizationPricingOverride,
  OrganizationPricingOverrides,
  PricingResponse,
  SetPricingRequest,
  UpdateOrganizationPricingOverride,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import { useOrganizationContext } from "@/shared/api/organizations"
import { fetchAllRows } from "@/shared/api/paging"
import {
  CATALOG,
  MODELS,
  ORGANIZATION_PRICING,
  ORGANIZATION_PROVIDER_MODELS,
  PRICING,
} from "@/shared/api/queryKeys"

const fetchAllPricing = () => fetchAllRows<PricingResponse>("/pricing")

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
  /**
   * Narrow to one model, which is what the rate editor needs: every period
   * stored for it, so it opens on the one in force and can refuse a new one
   * that would overlap. Without it the editor reads the first page of the whole
   * table, and an organization with more overrides than that page silently
   * starts opening a create form over a rate that already exists.
   *
   * Empty means no filter, so a caller holding a URL value passes it as it is
   * rather than converting one absent spelling into another.
   */
  modelKey?: string,
) {
  const organization = useOrganizationContext()
  const context = organization.data
  return useQuery({
    // The organization is part of the key, not only of the request, which
    // carries it implicitly: the server scopes this read by the session's
    // active organization, so without it two organizations share one cache
    // entry and the second reads the first's rows until its own land.
    // `useOrganizationSpendCeilings` keys itself the same way and says more
    // about why. `invalidateOrganizationPricing` matches on the head, so the
    // extra segment costs it nothing.
    queryKey: [
      ORGANIZATION_PRICING,
      context?.organization?.id ?? null,
      page,
      pageSize,
      modelKey ?? null,
    ],
    queryFn: () =>
      apiFetch<OrganizationPricingOverrides>(
        `/organizations/me/pricing?skip=${page * pageSize}&limit=${pageSize}${
          modelKey ? `&model_key=${encodeURIComponent(modelKey)}` : ""
        }`,
      ),
    staleTime: 60_000,
    // Kept across a page change and dropped across an organization change, for
    // the reason `useOrganizationSpendCeilings` gives.
    placeholderData: (previous, previousQuery) =>
      previousQuery?.queryKey[1] === (context?.organization?.id ?? null)
        ? previous
        : undefined,
    // Withheld until the context has settled, because the organization is part
    // of the key: asking before it lands keys the read as `null` and then again
    // under the organization, which is two requests for one page.
    enabled: enabled && (organization.isSuccess || organization.isError),
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
  // The offered-models panel reads the same rates through a different route, so
  // a rate written here moves a row there: its price and the badge saying which
  // rung set it. Without this the panel keeps showing the number you just
  // replaced.
  void queryClient.invalidateQueries({
    queryKey: [ORGANIZATION_PROVIDER_MODELS],
  })
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
