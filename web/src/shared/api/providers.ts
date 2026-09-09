import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  CreateStoredProviderRequest,
  KnownProvider,
  KnownProviderSummary,
  ProviderHealthResponse,
  ProvidersResponse,
  ReencryptProviderCredentialsResult,
  StoredProvider,
  TestProviderResult,
  UpdateStoredProviderRequest,
} from "@/client"
import { apiFetch, longRequestSignal } from "@/shared/api/client"
import {
  DISCOVERABLE,
  MODELS,
  NO_RETRY,
  PROVIDER_HEALTH,
  PROVIDERS,
  STORED_PROVIDERS,
} from "@/shared/api/queryKeys"

export function useProviders(enabled = true) {
  return useQuery({
    queryKey: [PROVIDERS],
    queryFn: () => apiFetch<ProvidersResponse>("/v1/providers"),
    staleTime: 5 * 60_000,
    enabled,
  })
}

// Every known provider the add-provider picker can offer: id + display name
// only. Built gateway-side without importing any provider SDK, so it is cheap
// and never moves within a session (the old full-catalog fetch used to import
// every provider SDK, which lagged the picker; issue #365).
export function useProviderCatalog() {
  return useQuery({
    queryKey: ["provider-catalog"],
    queryFn: () => apiFetch<KnownProviderSummary[]>("/v1/providers/catalog"),
    staleTime: Infinity,
  })
}

// Autofill hints (credential env var, default endpoint, whether a key is
// required) for the one provider the add-provider form has selected. Resolved
// lazily so only the chosen provider's SDK is imported gateway-side; disabled
// until a provider is picked. env_key_present is process-static, so cache it for
// the session like the catalog.
export function useProviderDetail(providerId: string) {
  return useQuery({
    queryKey: ["provider-catalog", providerId],
    queryFn: () =>
      apiFetch<KnownProvider>(
        `/v1/providers/catalog/${encodeURIComponent(providerId)}`,
      ),
    enabled: providerId !== "",
    staleTime: Infinity,
  })
}

// Every configured provider's reachability, for the health monitor. Backed by
// the same model-discovery test path as the per-provider "test connection", so a
// provider is healthy when its credentials can list models. This fans out to
// every configured provider, so automatic checks run at most hourly. The
// response's healthy/total counts are reused by the overview summary tile
// (issue #302).
// Checking provider health lists models for every configured provider. Keep the
// automatic probe infrequent; operators can still force an immediate re-check.
export const PROVIDER_HEALTH_REFRESH_MS = 60 * 60_000

export function useProviderHealth() {
  return useQuery({
    ...NO_RETRY,
    queryKey: [PROVIDER_HEALTH],
    queryFn: () => apiFetch<ProviderHealthResponse>("/v1/providers/health"),
    staleTime: PROVIDER_HEALTH_REFRESH_MS,
    refetchInterval: PROVIDER_HEALTH_REFRESH_MS,
  })
}

// Force a live re-check of every provider (clears the gateway's discovery cache),
// for an explicit "Refresh" action. Writes the fresh result straight into the
// health query so the table and any summary tile update together.
export function useRecheckProviderHealth() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () =>
      apiFetch<ProviderHealthResponse>("/v1/providers/health?refresh=true"),
    onSuccess: (data) => queryClient.setQueryData([PROVIDER_HEALTH], data),
  })
}

// Providers configured at runtime through the dashboard. Distinct from
// useProviders (static metadata for every configured provider, config + stored
// merged): this is the editable set, with the last 4 of each stored key.
export function useStoredProviders() {
  return useQuery({
    queryKey: [STORED_PROVIDERS],
    queryFn: () => apiFetch<StoredProvider[]>("/v1/provider-credentials"),
    staleTime: 60_000,
  })
}

// A new or changed provider can change which models the catalog and picker
// report, so a credential write invalidates those too.
function invalidateProviderViews(
  queryClient: ReturnType<typeof useQueryClient>,
): void {
  void queryClient.invalidateQueries({ queryKey: [STORED_PROVIDERS] })
  void queryClient.invalidateQueries({ queryKey: [PROVIDERS] })
  void queryClient.invalidateQueries({ queryKey: [MODELS] })
  void queryClient.invalidateQueries({ queryKey: [DISCOVERABLE] })
  // A credential change can flip a provider's reachability, so the health view
  // must re-check rather than show a verdict from the old key.
  void queryClient.invalidateQueries({ queryKey: [PROVIDER_HEALTH] })
}

export function useCreateStoredProvider() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateStoredProviderRequest) =>
      apiFetch<StoredProvider>("/v1/provider-credentials", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => invalidateProviderViews(queryClient),
  })
}

export function useUpdateStoredProvider() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      instance,
      body,
    }: {
      instance: string
      body: UpdateStoredProviderRequest
    }) =>
      apiFetch<StoredProvider>(
        `/v1/provider-credentials/${encodeURIComponent(instance)}`,
        {
          method: "PATCH",
          body: JSON.stringify(body),
        },
      ),
    onSuccess: () => invalidateProviderViews(queryClient),
  })
}

export function useDeleteStoredProvider() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (instance: string) =>
      apiFetch<void>(
        `/v1/provider-credentials/${encodeURIComponent(instance)}`,
        { method: "DELETE" },
      ),
    onSuccess: () => invalidateProviderViews(queryClient),
  })
}

export function useReencryptProviderCredentials() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () =>
      apiFetch<ReencryptProviderCredentialsResult>(
        "/v1/provider-credentials/reencrypt",
        {
          method: "POST",
          signal: longRequestSignal(),
        },
      ),
    onSuccess: () => invalidateProviderViews(queryClient),
  })
}

// Tests a stored provider's key by listing its models. Read-only on the server,
// so it invalidates nothing.
export function useTestStoredProvider() {
  return useMutation({
    mutationFn: (instance: string) =>
      apiFetch<TestProviderResult>(
        `/v1/provider-credentials/${encodeURIComponent(instance)}/test`,
        {
          method: "POST",
        },
      ),
  })
}

// Tests credentials from the add/edit form before they are saved. Nothing is
// persisted server-side, so it invalidates nothing.
export function useTestProviderCredentials() {
  return useMutation({
    mutationFn: (body: CreateStoredProviderRequest) =>
      apiFetch<TestProviderResult>("/v1/provider-credentials/test", {
        method: "POST",
        body: JSON.stringify(body),
      }),
  })
}

// Per-model metadata (modalities, capabilities, knowledge cutoff) from the
// models.dev catalog, keyed by `provider:model`. The gateway fetches and caches
// it, so this is cheap; kept fresh for a session since the catalog barely moves.
// `enabled` is the `useToolSettings` composition: the read is
// deployment-operator-only, so a tenant-facing page declines to ask.
