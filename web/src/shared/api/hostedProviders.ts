import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"

import type {
  CreateHostedModelRequest,
  CreateHostedProviderRegionRequest,
  CreateHostedProviderRequest,
  DeploymentProviderMigration,
  DeploymentProviderMigrationReviewedRow,
  HostedAvailableModels,
  HostedCatalogUpdatePreview,
  HostedCatalogUpdateRequest,
  HostedCatalogUpdateResult,
  HostedModel,
  HostedModels,
  HostedModelsRefresh,
  HostedPriceReset,
  HostedProvider,
  HostedProviders,
  HostedProviderTestResult,
  HostedVendor,
  HostedVendorModelsSwitch,
  PlatformConnectStart,
  PlatformConnectStatus,
  ReencryptHostedProvidersResult,
  SwitchHostedVendorModelsRequest,
  TestHostedProviderRequest,
  UpdateHostedModelRequest,
  UpdateHostedProviderRegionRequest,
  UpdateHostedProviderRequest,
  UpdateHostedVendorRequest,
} from "@/client"
import { apiFetch, longRequestSignal } from "@/shared/api/client"
import {
  AUTOMATIC_FALLBACKS,
  CATALOG,
  HOSTED_CATALOG_UPDATE_PREVIEW,
  HOSTED_PRICE_RESET_PREVIEW,
  HOSTED_PROVIDER_AVAILABLE_MODELS,
  HOSTED_PROVIDERS,
  MODELS,
  NO_RETRY,
  PLATFORM_CONNECT,
  PRICING,
  PROVIDER_HEALTH,
  PROVIDERS,
} from "@/shared/api/queryKeys"

// The providers this deployment serves on its own keys, for a request that
// brings no credential of its own (`/hosted-providers`). Operator-only: the API
// answers a non-operator 404, which is why the page asks the deployment admin
// gate before it asks for this list.
//
// Everything is addressed by the provider's name (`name`, which defaults to its
// implementation `provider`), the row's identity. Each write that
// can change a price or what is served also invalidates the catalog reads,
// because a hosted model's rate is the deployment price list's.
//
// The create, both refreshes, the preview and the model picker dial the
// upstream (every provider, for the catalog update), so they carry `longRequestSignal`
// rather than the default request deadline.

/**
 * The API's maximum. An operator adds hosted providers by hand, so one page is
 * the whole list.
 */
const PROVIDER_PAGE_LIMIT = 1000

/** The API's maximum for one provider's models. */
const MODEL_LIST_LIMIT = 1000

const providerPath = (provider: string) =>
  `/hosted-providers/${encodeURIComponent(provider)}`

function useInvalidateHosted() {
  const queryClient = useQueryClient()
  return () => {
    void queryClient.invalidateQueries({ queryKey: [HOSTED_PROVIDERS] })
    void queryClient.invalidateQueries({ queryKey: [PRICING] })
    void queryClient.invalidateQueries({ queryKey: [MODELS] })
    void queryClient.invalidateQueries({ queryKey: [CATALOG] })
    void queryClient.invalidateQueries({ queryKey: [AUTOMATIC_FALLBACKS] })
    void queryClient.invalidateQueries({ queryKey: [PROVIDER_HEALTH] })
  }
}

/**
 * Every hosted provider. `isFresh` reads the list again rather than trusting a
 * cached one, for a page opened on one provider by name, which must not answer
 * "not found" from a list cached before that provider was added elsewhere.
 */
export function useHostedProviders(
  enabled = true,
  { isFresh = false }: { isFresh?: boolean } = {},
) {
  return useQuery({
    queryKey: [HOSTED_PROVIDERS, "list"],
    queryFn: () =>
      apiFetch<HostedProviders>(
        `/hosted-providers?limit=${PROVIDER_PAGE_LIMIT}`,
      ),
    // Zero rather than `refetchOnMount`, which never fires for a query that
    // is still waiting on the operator gate when the page mounts.
    staleTime: isFresh ? 0 : 30_000,
    enabled,
  })
}

export function useCreateHostedProvider() {
  const invalidate = useInvalidateHosted()
  return useMutation({
    mutationFn: (body: CreateHostedProviderRequest) =>
      apiFetch<HostedProvider>("/hosted-providers", {
        method: "POST",
        body: JSON.stringify(body),
        // The create asks the provider what it serves and offers all of it.
        signal: longRequestSignal(),
      }),
    onSuccess: invalidate,
  })
}

export function useUpdateHostedProvider() {
  const invalidate = useInvalidateHosted()
  return useMutation({
    mutationFn: ({
      provider,
      body,
    }: {
      provider: string
      body: UpdateHostedProviderRequest
    }) =>
      apiFetch<HostedProvider>(providerPath(provider), {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: invalidate,
  })
}

export function useDeleteHostedProvider() {
  const queryClient = useQueryClient()
  const invalidate = useInvalidateHosted()
  return useMutation({
    mutationFn: (provider: string) =>
      apiFetch<{ message: string }>(providerPath(provider), {
        method: "DELETE",
      }),
    onSuccess: () => {
      invalidate()
      // Removing the connected provider is how a deployment disconnects.
      void queryClient.invalidateQueries({ queryKey: [PLATFORM_CONNECT] })
    },
  })
}

// A provider's regions (`/hosted-providers/{provider}/regions`). Each write
// answers the whole provider, whose list row is what every page reads, so the
// hosted reads are invalidated rather than patched.

const regionPath = (provider: string, slug: string) =>
  `${providerPath(provider)}/regions/${encodeURIComponent(slug)}`

/**
 * Every region write answers the whole provider, which is put into the cached
 * list at once (keeping its list-only model counts) before the list is read
 * again: a second move pressed before the reread lands must see the new order.
 */
function useApplyRegionWrite() {
  const queryClient = useQueryClient()
  const invalidate = useInvalidateHosted()
  return (provider: HostedProvider) => {
    queryClient.setQueryData<HostedProviders>(
      [HOSTED_PROVIDERS, "list"],
      (list) =>
        list && {
          ...list,
          data: list.data.map((row) =>
            row.name === provider.name
              ? { ...provider, models: row.models }
              : row,
          ),
        },
    )
    invalidate()
  }
}

/**
 * Add a region after the provider's others. The first one also turns the
 * provider's own connection into a region ahead of it, named by
 * `primary_slug` and `primary_label`.
 */
export function useAddHostedProviderRegion(provider: string) {
  const applyRegionWrite = useApplyRegionWrite()
  return useMutation({
    mutationFn: (body: CreateHostedProviderRegionRequest) =>
      apiFetch<HostedProvider>(`${providerPath(provider)}/regions`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: applyRegionWrite,
  })
}

/** Change, switch or move one region; `position` is its new place, from 0. */
export function useUpdateHostedProviderRegion(provider: string) {
  const applyRegionWrite = useApplyRegionWrite()
  return useMutation({
    mutationFn: ({
      slug,
      body,
    }: {
      slug: string
      body: UpdateHostedProviderRegionRequest
    }) =>
      apiFetch<HostedProvider>(regionPath(provider, slug), {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: applyRegionWrite,
  })
}

/** Remove one region. Removing the last leaves the provider on its own connection. */
export function useRemoveHostedProviderRegion(provider: string) {
  const applyRegionWrite = useApplyRegionWrite()
  return useMutation({
    mutationFn: (slug: string) =>
      apiFetch<HostedProvider>(regionPath(provider, slug), {
        method: "DELETE",
      }),
    onSuccess: applyRegionWrite,
  })
}

// Connecting the deployment to otari.ai (`/platform/connect`): the gateway asks
// otari.ai for a code, the operator approves it there, and the gateway stores
// the key as the `otari-ai` hosted provider. Offered only where the bootstrap
// says so (`platform_connect`), which is also what keeps the status read from
// being sent to a gateway that mounts no such route.

/** How often an open page asks whether a pending code was approved. */
const PLATFORM_CONNECT_POLL_MS = 3_000

/**
 * Where the connection stands, polled only while a code is waiting to be
 * approved: every other status is one that does not change on its own.
 *
 * The gateway asks otari.ai at most once per interval however often this is
 * read. The read that first finds the attempt connected also refreshes the
 * provider and model lists, which the gateway changed while answering it.
 */
export function usePlatformConnectStatus(enabled: boolean) {
  const queryClient = useQueryClient()
  const invalidate = useInvalidateHosted()
  return useQuery({
    queryKey: [PLATFORM_CONNECT],
    queryFn: async () => {
      const before = queryClient.getQueryData<PlatformConnectStatus>([
        PLATFORM_CONNECT,
      ])
      const status = await apiFetch<PlatformConnectStatus>("/platform/connect")
      // Also on a first read: the gateway may have stored the provider while
      // answering it, beside a provider list fetched at the same moment.
      if (before?.status !== "connected" && status.status === "connected") {
        if (before?.status === "pending") {
          // The catalog's short model ids are re-indexed once a minute, and
          // the setup guide hands one out right after this. Best effort: a
          // failure only means the id resolves at the next tick.
          await apiFetch("/catalog/selectors/refresh", {
            method: "POST",
          }).catch(() => undefined)
        }
        invalidate()
      }
      return status
    },
    enabled,
    refetchInterval: (query) =>
      query.state.data?.status === "pending" ? PLATFORM_CONNECT_POLL_MS : false,
    // A failed read is shown where the code is, and the next poll asks again.
    ...NO_RETRY,
  })
}

/** Ask otari.ai for a code. The answer seeds the status, so the page shows the code at once. */
export function useStartPlatformConnect() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () =>
      apiFetch<PlatformConnectStart>("/platform/connect", { method: "POST" }),
    onSuccess: (started) => {
      queryClient.setQueryData<PlatformConnectStatus>([PLATFORM_CONNECT], {
        status: "pending",
        ...started,
      })
    },
  })
}

export function useCancelPlatformConnect() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () => apiFetch<void>("/platform/connect", { method: "DELETE" }),
    onSuccess: () => {
      queryClient.setQueryData<PlatformConnectStatus>([PLATFORM_CONNECT], {
        status: "idle",
        user_code: null,
        verification_uri_complete: null,
        expires_in: null,
      })
    },
  })
}

/**
 * A provider's offered models, all of them, with each vendor's switch and counts.
 *
 * Not paged: the models are shown under their vendors, and a vendor's models
 * split across pages would hide what its switch covers. The API's maximum is
 * the bound, which no provider here comes near.
 *
 * The provider is part of the key, so the placeholder is kept only while the
 * same provider is shown: a bare `keepPreviousData` would paint one provider's
 * models under another's name for a frame.
 */
export function useHostedModels(
  provider: string,
  /** One kind of model (`ModelType`, or `unknown`); empty for every kind. */
  type = "",
  /** Only models whose name or vendor contains this text; empty for every model. */
  search = "",
  /** Only served models, or only withheld ones; undefined for both. */
  served?: boolean,
) {
  return useQuery({
    queryKey: [HOSTED_PROVIDERS, "models", provider, type, search, served],
    queryFn: () =>
      apiFetch<HostedModels>(
        `${providerPath(provider)}/models?limit=${MODEL_LIST_LIMIT}${type ? `&type=${encodeURIComponent(type)}` : ""}${search ? `&search=${encodeURIComponent(search)}` : ""}${served === undefined ? "" : `&enabled=${served}`}`,
      ),
    staleTime: 30_000,
    placeholderData: (previous, previousQuery) =>
      previousQuery?.queryKey[2] === provider ? previous : undefined,
  })
}

/** Switch one vendor on or off on a provider; its models keep their own switches. */
export function useSetHostedVendor(provider: string) {
  const invalidate = useInvalidateHosted()
  return useMutation({
    mutationFn: (body: UpdateHostedVendorRequest) =>
      apiFetch<HostedVendor>(`${providerPath(provider)}/vendors`, {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: invalidate,
  })
}

/** Set every model of one vendor's switch at once. */
export function useSwitchHostedVendorModels(provider: string) {
  const invalidate = useInvalidateHosted()
  return useMutation({
    mutationFn: (body: SwitchHostedVendorModelsRequest) =>
      apiFetch<HostedVendorModelsSwitch>(
        `${providerPath(provider)}/models/switch`,
        { method: "POST", body: JSON.stringify(body) },
      ),
    onSuccess: invalidate,
  })
}

/**
 * What putting a provider's operator-set prices back on the default would
 * change, read once per opening of the sync dialog.
 *
 * A dry-run POST read as a query, keyed on the opening for the reason
 * `useHostedCatalogUpdatePreview` gives: a reopen shows the pending state
 * rather than the last opening's answer.
 */
export function useHostedPriceResetPreview(
  provider: string,
  opening: number,
  /** One vendor's models; empty for every vendor. */
  vendor = "",
) {
  return useQuery({
    ...NO_RETRY,
    queryKey: [HOSTED_PRICE_RESET_PREVIEW, provider, opening, vendor],
    queryFn: () =>
      apiFetch<HostedPriceReset>(
        `${providerPath(provider)}/models/reset-prices`,
        {
          method: "POST",
          body: JSON.stringify({ dry_run: true, vendor: vendor || null }),
        },
      ),
    enabled: opening > 0,
    staleTime: Number.POSITIVE_INFINITY,
    gcTime: 0,
  })
}

/** Put the named models back on the default price. */
export function useResetHostedModelPrices(provider: string) {
  const invalidate = useInvalidateHosted()
  return useMutation({
    mutationFn: (models: string[]) =>
      apiFetch<HostedPriceReset>(
        `${providerPath(provider)}/models/reset-prices`,
        { method: "POST", body: JSON.stringify({ models, dry_run: false }) },
      ),
    onSuccess: invalidate,
  })
}

/**
 * What the provider says it serves on the stored key. Fetched only while the
 * add-model form is open, because answering dials the upstream, and held for a
 * minute so one form visit is one dial.
 */
export function useHostedAvailableModels(provider: string, enabled: boolean) {
  return useQuery({
    ...NO_RETRY,
    queryKey: [HOSTED_PROVIDER_AVAILABLE_MODELS, provider],
    queryFn: () =>
      apiFetch<HostedAvailableModels>(
        `${providerPath(provider)}/available-models`,
        { signal: longRequestSignal() },
      ),
    staleTime: 60_000,
    enabled,
  })
}

export function useAddHostedModel(provider: string) {
  const invalidate = useInvalidateHosted()
  return useMutation({
    mutationFn: (body: CreateHostedModelRequest) =>
      apiFetch<HostedModel>(`${providerPath(provider)}/models`, {
        method: "POST",
        body: JSON.stringify({ ...body, model: body.model.trim() }),
      }),
    onSuccess: invalidate,
  })
}

export function useUpdateHostedModel(provider: string) {
  const invalidate = useInvalidateHosted()
  return useMutation({
    mutationFn: ({
      modelId,
      body,
    }: {
      modelId: string
      body: UpdateHostedModelRequest
    }) =>
      apiFetch<HostedModel>(
        `${providerPath(provider)}/models/${encodeURIComponent(modelId)}`,
        { method: "PATCH", body: JSON.stringify(body) },
      ),
    onSuccess: invalidate,
  })
}

export function useRefreshHostedModels(provider: string) {
  const queryClient = useQueryClient()
  const invalidate = useInvalidateHosted()
  return useMutation({
    mutationFn: () =>
      apiFetch<HostedModelsRefresh>(
        `${providerPath(provider)}/models/refresh`,
        {
          method: "POST",
          signal: longRequestSignal(),
        },
      ),
    onSuccess: () => {
      invalidate()
      // The dial's answer is fresher than whatever the picker last cached.
      void queryClient.invalidateQueries({
        queryKey: [HOSTED_PROVIDER_AVAILABLE_MODELS, provider],
      })
    },
  })
}

/**
 * What updating the catalog would do, read once per opening of its confirm
 * dialog, and again when the cleanup is ticked or cleared.
 *
 * A POST that the dialog reads as a query: it fetches the latest models.dev
 * prices and holds them as pending for the update to name by digest, and it
 * dials every provider; nothing else is written. `opening` counts the openings
 * and is 0 while the dialog is closed. Keying on it gives every opening a query
 * with no data, so a reopen shows the pending state rather than the last
 * opening's answer, and `gcTime: 0` drops that answer once nothing observes it.
 * Never stale within an opening, so a window focus cannot swap the preview under
 * a confirm that was about to fire on it.
 */
export function useHostedCatalogUpdatePreview(
  opening: number,
  cleanup: boolean,
) {
  return useQuery({
    ...NO_RETRY,
    queryKey: [HOSTED_CATALOG_UPDATE_PREVIEW, opening, cleanup],
    queryFn: () =>
      apiFetch<HostedCatalogUpdatePreview>(
        "/hosted-providers/catalog-update/preview",
        {
          method: "POST",
          body: JSON.stringify({ cleanup }),
          signal: longRequestSignal(),
        },
      ),
    enabled: opening > 0,
    staleTime: Number.POSITIVE_INFINITY,
    gcTime: 0,
  })
}

/**
 * The catalog update: the reviewed models.dev prices, then every provider's
 * models, then the cleanup if it was asked for.
 */
export function useHostedCatalogUpdate() {
  const invalidate = useInvalidateHosted()
  return useMutation({
    mutationFn: (body: HostedCatalogUpdateRequest) =>
      apiFetch<HostedCatalogUpdateResult>("/hosted-providers/catalog-update", {
        method: "POST",
        body: JSON.stringify(body),
        signal: longRequestSignal(),
      }),
    onSuccess: invalidate,
  })
}

/**
 * Credentials tried against the provider without storing them. With `name`
 * naming a stored provider, a field left out is read from it, so an edit can be
 * tested without typing the key again. Dials the upstream.
 */
export function useTestHostedProvider() {
  return useMutation({
    mutationFn: (body: TestHostedProviderRequest) =>
      apiFetch<HostedProviderTestResult>("/hosted-providers/test", {
        method: "POST",
        body: JSON.stringify(body),
        signal: longRequestSignal(),
      }),
  })
}

/** Re-encrypts every stored hosted provider key under the primary `OTARI_SECRET_KEY`. */
export function useReencryptHostedProviders() {
  const invalidate = useInvalidateHosted()
  return useMutation({
    mutationFn: () =>
      apiFetch<ReencryptHostedProvidersResult>("/hosted-providers/reencrypt", {
        method: "POST",
      }),
    onSuccess: invalidate,
  })
}

/**
 * What moving the providers startup could not move onto hosted providers
 * would do. Reads the stored rows only and dials nothing, so
 * it is cheap enough to ask whenever the page is open.
 */
export function useDeploymentProviderMigrationPreview(enabled: boolean) {
  return useQuery({
    queryKey: [HOSTED_PROVIDERS, "migration"],
    queryFn: () =>
      apiFetch<DeploymentProviderMigration>("/hosted-providers/migration"),
    staleTime: 30_000,
    enabled,
  })
}

/**
 * Moves the stored providers named, the rows of the preview the operator
 * reviewed at the versions shown; a row the preview did not show, or one
 * changed since, is left where it is. Each move asks
 * the provider what it serves, as a create does.
 */
export function useApplyDeploymentProviderMigration() {
  const queryClient = useQueryClient()
  const invalidate = useInvalidateHosted()
  return useMutation({
    mutationFn: (rows: DeploymentProviderMigrationReviewedRow[]) =>
      apiFetch<DeploymentProviderMigration>("/hosted-providers/migration", {
        method: "POST",
        body: JSON.stringify({ rows }),
        signal: longRequestSignal(),
      }),
    onSuccess: () => {
      invalidate()
      void queryClient.invalidateQueries({ queryKey: [PROVIDERS] })
    },
  })
}

/**
 * Removes a stored provider the move could not take. It serves nothing, so
 * only the migration preview changes.
 */
export function useDiscardDeploymentProvider() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (instance: string) =>
      apiFetch<void>(
        `/hosted-providers/migration/${encodeURIComponent(instance)}`,
        { method: "DELETE" },
      ),
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: [HOSTED_PROVIDERS, "migration"],
      }),
  })
}
