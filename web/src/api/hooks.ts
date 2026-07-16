import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";

import { ApiError, apiFetch } from "@/api/client";
import type {
  AliasResponse,
  CreateAliasRequest,
  DashboardBuild,
  DiscoverableModelsResponse,
  GatewaySettings,
  HealthResponse,
  ModelListResponse,
  ModelMetadataResponse,
  ModelObject,
  PricingResponse,
  ProvidersResponse,
  SetPricingRequest,
  UpdateSettingsRequest,
  UsageEntry,
  UsageSummary,
} from "@/api/types";

const USAGE = "usage";
const HEALTH = "health";
const MODELS = "models";
const PRICING = "pricing";
const SETTINGS = "settings";
const ALIASES = "aliases";
// Deliberately not nested under MODELS: pricing mutations invalidate that key,
// and a price change cannot alter which models a provider serves. Sharing the
// key would fire a live provider call on every save.
const DISCOVERABLE = "discoverable";
const PROVIDERS = "providers";
const METADATA = "model-metadata";
const BUILD = "build";

// How often an open tab asks whether the app it is running is still the one the
// gateway serves. Cheap (a hash of one small file) and only while the tab is
// open, so a minute keeps a deploy from going unnoticed for long.
const BUILD_POLL_MS = 60_000;

export function useHealth() {
  return useQuery({
    queryKey: [HEALTH],
    queryFn: () => apiFetch<HealthResponse>("/health"),
    staleTime: 60_000,
  });
}

// A page of raw log rows, for the per-request log view. Anything reporting a
// count or a total wants useUsageSummary instead: this is capped server-side, so
// summing it under-reports as soon as the log outgrows `limit`.
export function useUsage(limit = 500) {
  return useQuery({
    queryKey: [USAGE, limit],
    queryFn: () => apiFetch<UsageEntry[]>(`/v1/usage?limit=${limit}`),
  });
}

// Totals and a per-model breakdown over the whole usage log, aggregated in the
// database rather than from a page of rows.
export function useUsageSummary() {
  return useQuery({
    queryKey: [USAGE, "summary"],
    queryFn: () => apiFetch<UsageSummary>("/v1/usage/summary"),
  });
}

export function useModels() {
  return useQuery({
    queryKey: [MODELS],
    queryFn: () => apiFetch<ModelListResponse>("/v1/models"),
    staleTime: 60_000,
  });
}

// Prices models the catalog does not list. GET /v1/models omits a model that is
// neither priced nor discoverable, and omits an alias's target on purpose, yet
// either can still have traffic and be billed at the default rate. Asking about
// each one by name is the only way to show what it actually costs.
//
// Bounded by "models with traffic the catalog does not describe", which is empty
// unless model_discovery is off. A model the gateway truly cannot price 404s,
// which is not an error worth retrying or surfacing: the row just stays unpriced.
export function useUnlistedModels(keys: string[]) {
  return useQueries({
    queries: keys.map((key) => ({
      queryKey: [MODELS, key],
      queryFn: () => apiFetch<ModelObject>(`/v1/models/${encodeURIComponent(key)}`),
      retry: (_count: number, error: Error) => !(error instanceof ApiError && error.status === 404),
      staleTime: 60_000,
    })),
    // Keyed by the requested key rather than the response id, so a row that came
    // back under a different id is dropped instead of shown against the wrong
    // model. `combine` keeps the map reference stable between renders.
    combine: (results) => {
      const byKey = new Map<string, ModelObject>();
      results.forEach((result, index) => {
        if (result.data) {
          byKey.set(keys[index], result.data);
        }
      });
      return byKey;
    },
  });
}

export function useDashboardBuild() {
  return useQuery({
    queryKey: [BUILD],
    queryFn: () => apiFetch<DashboardBuild>("/dashboard-build.json"),
    refetchInterval: BUILD_POLL_MS,
    // A tab left open in the background is the one most likely to be stale, so
    // check again the moment someone comes back to it.
    refetchOnWindowFocus: true,
    staleTime: 0,
    // A failed check is not worth reporting: the tab keeps working, and the next
    // poll retries anyway.
    retry: false,
  });
}

// Every model the configured credentials can reach, per provider. Distinct from
// useModels: that is the catalog served to API callers (curated by
// model_discovery, aliases listed, targets withheld), while this is what an
// operator could pick from. A provider that failed is reported rather than
// dropped, so the picker can say why a list is empty.
//
// Live provider calls, cached gateway-side; kept fresh for the length of a
// session rather than refetched per open, since the set of models a key can
// reach does not move minute to minute.
export function useDiscoverableModels() {
  return useQuery({
    queryKey: [DISCOVERABLE],
    queryFn: () => apiFetch<DiscoverableModelsResponse>("/v1/models/discoverable"),
    staleTime: 5 * 60_000,
  });
}

// Static metadata for every configured provider: capabilities, doc and pricing
// links, display name. Network-free gateway-side (bundled datasets), so it does
// not move within a session; kept fresh for a few minutes like discovery.
export function useProviders() {
  return useQuery({
    queryKey: [PROVIDERS],
    queryFn: () => apiFetch<ProvidersResponse>("/v1/providers"),
    staleTime: 5 * 60_000,
  });
}

// Per-model metadata (modalities, capabilities, knowledge cutoff) from the
// models.dev catalog, keyed by `provider:model`. The gateway fetches and caches
// it, so this is cheap; kept fresh for a session since the catalog barely moves.
export function useModelMetadata() {
  return useQuery({
    queryKey: [METADATA],
    queryFn: () => apiFetch<ModelMetadataResponse>("/v1/models/metadata"),
    staleTime: 10 * 60_000,
  });
}

export function useAliases() {
  return useQuery({
    queryKey: [ALIASES],
    queryFn: () => apiFetch<AliasResponse[]>("/v1/aliases"),
    staleTime: 60_000,
  });
}

export function useCreateAlias() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateAliasRequest) =>
      apiFetch<AliasResponse>("/v1/aliases", { method: "POST", body: JSON.stringify(body) }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ALIASES] });
      // An alias is listed as a model, so the catalog changes too.
      void queryClient.invalidateQueries({ queryKey: [MODELS] });
    },
  });
}

export function useDeleteAlias() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (name: string) => apiFetch<void>(`/v1/aliases/${encodeURIComponent(name)}`, { method: "DELETE" }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ALIASES] });
      void queryClient.invalidateQueries({ queryKey: [MODELS] });
    },
  });
}

export function useSettings() {
  return useQuery({
    queryKey: [SETTINGS],
    queryFn: () => apiFetch<GatewaySettings>("/v1/settings"),
    staleTime: 60_000,
  });
}

export function useUpdateSettings() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: UpdateSettingsRequest) =>
      apiFetch<GatewaySettings>("/v1/settings", { method: "PATCH", body: JSON.stringify(body) }),
    onSuccess: (data) => {
      queryClient.setQueryData([SETTINGS], data);
      // Toggling discovery changes which models the catalog and picker report.
      void queryClient.invalidateQueries({ queryKey: [MODELS] });
      void queryClient.invalidateQueries({ queryKey: [DISCOVERABLE] });
    },
  });
}

export function usePricing() {
  return useQuery({
    queryKey: [PRICING],
    queryFn: () => apiFetch<PricingResponse[]>("/v1/pricing?limit=1000"),
  });
}

export function useSetPricing() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: SetPricingRequest) =>
      apiFetch<PricingResponse>("/v1/pricing", { method: "POST", body: JSON.stringify(body) }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [PRICING] });
      void queryClient.invalidateQueries({ queryKey: [MODELS] });
    },
  });
}

export function useDeletePricing() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (modelKey: string) =>
      apiFetch<void>(`/v1/pricing/${encodeURIComponent(modelKey)}`, { method: "DELETE" }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [PRICING] });
      void queryClient.invalidateQueries({ queryKey: [MODELS] });
    },
  });
}
