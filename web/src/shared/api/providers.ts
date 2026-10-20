import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  KnownProvider,
  KnownProviderSummary,
  ProviderHealthResponse,
  ProvidersResponse,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import { useHostedProviders } from "@/shared/api/hostedProviders"
import { NO_RETRY, PROVIDER_HEALTH, PROVIDERS } from "@/shared/api/queryKeys"

export function useProviders(enabled = true) {
  return useQuery({
    queryKey: [PROVIDERS],
    queryFn: () => apiFetch<ProvidersResponse>("/providers"),
    staleTime: 5 * 60_000,
    enabled,
  })
}

/**
 * The provider names the deployment serves on its own credentials: the
 * config.yml instances and the hosted providers, sorted. Operator-only, so a
 * caller passes whether it is one.
 */
export function useDeploymentProviderNames(enabled: boolean): string[] {
  const configured = useProviders(enabled)
  const hosted = useHostedProviders(enabled)
  return [
    ...new Set([
      ...(configured.data?.providers ?? []).map((p) => p.instance),
      ...(hosted.data?.data ?? []).map((p) => p.name),
    ]),
  ].sort((a, b) => a.localeCompare(b))
}

// Every known provider the add-provider picker can offer: id + display name
// only. Built gateway-side without importing any provider SDK, so it is cheap
// and never moves within a session (the old full-catalog fetch used to import
// every provider SDK, which lagged the picker; issue #365).
export function useProviderCatalog() {
  return useQuery({
    queryKey: ["provider-catalog"],
    queryFn: () => apiFetch<KnownProviderSummary[]>("/providers/catalog"),
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
        `/providers/catalog/${encodeURIComponent(providerId)}`,
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

export function useProviderHealth(enabled = true) {
  return useQuery({
    ...NO_RETRY,
    queryKey: [PROVIDER_HEALTH],
    queryFn: () => apiFetch<ProviderHealthResponse>("/providers/health"),
    staleTime: PROVIDER_HEALTH_REFRESH_MS,
    refetchInterval: PROVIDER_HEALTH_REFRESH_MS,
    enabled,
  })
}

// Force a live re-check of every provider (clears the gateway's discovery cache),
// for an explicit "Refresh" action. Writes the fresh result straight into the
// health query so the table and any summary tile update together.
export function useRecheckProviderHealth() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () =>
      apiFetch<ProviderHealthResponse>("/providers/health?refresh=true"),
    onSuccess: (data) => queryClient.setQueryData([PROVIDER_HEALTH], data),
  })
}

// Per-model metadata (modalities, capabilities, knowledge cutoff) from the
// models.dev catalog, keyed by `provider:model`. The gateway fetches and caches
// it, so this is cheap; kept fresh for a session since the catalog barely moves.
// `enabled` is the `useToolSettings` composition: the read is
// deployment-operator-only, so a tenant-facing page declines to ask.
