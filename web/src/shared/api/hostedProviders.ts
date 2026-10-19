import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"

import type {
  CreateHostedModelRequest,
  CreateHostedProviderRequest,
  DeploymentProviderMigration,
  DeploymentProviderMigrationReviewedRow,
  HostedAvailableModels,
  HostedCatalogRefresh,
  HostedModel,
  HostedModels,
  HostedModelsRefresh,
  HostedProvider,
  HostedProviders,
  HostedProviderTestResult,
  ReencryptHostedProvidersResult,
  TestHostedProviderRequest,
  UpdateHostedModelRequest,
  UpdateHostedProviderRequest,
} from "@/client"
import { apiFetch, longRequestSignal } from "@/shared/api/client"
import {
  CATALOG,
  HOSTED_PROVIDER_AVAILABLE_MODELS,
  HOSTED_PROVIDERS,
  HOSTED_PROVIDERS_REFRESH_PREVIEW,
  MODELS,
  NO_RETRY,
  PRICING,
  PROVIDER_HEALTH,
  PROVIDERS,
  STORED_PROVIDERS,
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
// upstream (every provider, for the sweep), so they carry `longRequestSignal`
// rather than the default request deadline.

/**
 * The API's maximum. An operator adds hosted providers by hand, so one page is
 * the whole list.
 */
const PROVIDER_PAGE_LIMIT = 1000

const providerPath = (provider: string) =>
  `/hosted-providers/${encodeURIComponent(provider)}`

function useInvalidateHosted() {
  const queryClient = useQueryClient()
  return () => {
    void queryClient.invalidateQueries({ queryKey: [HOSTED_PROVIDERS] })
    void queryClient.invalidateQueries({ queryKey: [PRICING] })
    void queryClient.invalidateQueries({ queryKey: [MODELS] })
    void queryClient.invalidateQueries({ queryKey: [CATALOG] })
    void queryClient.invalidateQueries({ queryKey: [PROVIDER_HEALTH] })
  }
}

export function useHostedProviders(enabled = true) {
  return useQuery({
    queryKey: [HOSTED_PROVIDERS, "list"],
    queryFn: () =>
      apiFetch<HostedProviders>(
        `/hosted-providers?limit=${PROVIDER_PAGE_LIMIT}`,
      ),
    staleTime: 30_000,
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
  const invalidate = useInvalidateHosted()
  return useMutation({
    mutationFn: (provider: string) =>
      apiFetch<{ message: string }>(providerPath(provider), {
        method: "DELETE",
      }),
    onSuccess: invalidate,
  })
}

/**
 * One page of a provider's offered models.
 *
 * The provider is part of the key, so the placeholder is kept only while the
 * same provider pages: a bare `keepPreviousData` would paint one provider's
 * models under another's name for a frame when a different row is opened.
 */
export function useHostedModels(
  provider: string,
  page: number,
  pageSize: number,
) {
  return useQuery({
    queryKey: [HOSTED_PROVIDERS, "models", provider, page, pageSize],
    queryFn: () =>
      apiFetch<HostedModels>(
        `${providerPath(provider)}/models?skip=${page * pageSize}&limit=${pageSize}`,
      ),
    staleTime: 30_000,
    placeholderData: (previous, previousQuery) =>
      previousQuery?.queryKey[2] === provider ? previous : undefined,
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
 * What a catalog sweep would remove, read once per opening of its confirm
 * dialog.
 *
 * `opening` counts the openings and is 0 while the dialog is closed. Keying on
 * it gives every opening a query with no data, so a reopen shows the pending
 * state rather than the last opening's list, and `gcTime: 0` drops that list
 * once nothing observes it. Never stale within an opening, so a window focus
 * cannot swap the list under a confirm that was about to fire on it.
 */
export function useHostedCatalogRefreshPreview(opening: number) {
  return useQuery({
    ...NO_RETRY,
    queryKey: [HOSTED_PROVIDERS_REFRESH_PREVIEW, opening],
    queryFn: () =>
      apiFetch<HostedCatalogRefresh>("/hosted-providers/refresh/preview", {
        signal: longRequestSignal(),
      }),
    enabled: opening > 0,
    staleTime: Number.POSITIVE_INFINITY,
    gcTime: 0,
  })
}

/**
 * The deployment-wide sweep: every enabled provider is asked again what it
 * lists, then every priced model nothing here serves leaves the catalog.
 */
export function useHostedCatalogRefresh() {
  const invalidate = useInvalidateHosted()
  return useMutation({
    mutationFn: () =>
      apiFetch<HostedCatalogRefresh>("/hosted-providers/refresh", {
        method: "POST",
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
 * What moving the deployment providers stored through the dashboard onto
 * hosted providers would do. Reads the stored rows only and dials nothing, so
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
 * Moves the deployment providers named, the rows of the preview the operator
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
      void queryClient.invalidateQueries({ queryKey: [STORED_PROVIDERS] })
      void queryClient.invalidateQueries({ queryKey: [PROVIDERS] })
    },
  })
}
