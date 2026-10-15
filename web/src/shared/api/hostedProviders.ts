import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"

import type {
  CreateHostedModelRequest,
  CreateHostedProviderRequest,
  HostedAvailableModels,
  HostedCatalogRefresh,
  HostedModel,
  HostedModels,
  HostedModelsRefresh,
  HostedProvider,
  HostedProviders,
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
} from "@/shared/api/queryKeys"

// The providers this deployment serves on its own keys, for a request that
// brings no credential of its own (`/hosted-providers`). Operator-only: the API
// answers a non-operator 404, which is why the page asks the deployment admin
// gate before it asks for this list.
//
// Everything is addressed by provider name, the row's identity. Each write that
// can change a price or what is served also invalidates the catalog reads,
// because a hosted model's rate is the deployment price list's.
//
// The create, both refreshes, the preview and the model picker dial the
// upstream (every provider, for the sweep), so they carry `longRequestSignal`
// rather than the default request deadline.

/** The hosted-provider list is small: one row per provider implementation. */
const PROVIDER_PAGE_LIMIT = 100

const providerPath = (provider: string) =>
  `/hosted-providers/${encodeURIComponent(provider)}`

function useInvalidateHosted() {
  const queryClient = useQueryClient()
  return () => {
    void queryClient.invalidateQueries({ queryKey: [HOSTED_PROVIDERS] })
    void queryClient.invalidateQueries({ queryKey: [PRICING] })
    void queryClient.invalidateQueries({ queryKey: [MODELS] })
    void queryClient.invalidateQueries({ queryKey: [CATALOG] })
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
 * What a catalog sweep would remove, read while its confirm dialog is open.
 *
 * Not cached past the dialog (`gcTime: 0`): the next press must ask again
 * rather than confirm against a list from five minutes ago.
 */
export function useHostedCatalogRefreshPreview(enabled: boolean) {
  return useQuery({
    ...NO_RETRY,
    queryKey: [HOSTED_PROVIDERS_REFRESH_PREVIEW],
    queryFn: () =>
      apiFetch<HostedCatalogRefresh>("/hosted-providers/refresh/preview", {
        signal: longRequestSignal(),
      }),
    enabled,
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
