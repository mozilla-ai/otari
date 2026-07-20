import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { apiFetch } from "@/api/client";
import type {
  AliasResponse,
  CreateAliasRequest,
  CreateStoredProviderRequest,
  DashboardBuild,
  DiscoverableModelsResponse,
  GatewaySettings,
  KnownProvider,
  ModelListResponse,
  ModelMetadataResponse,
  PricingResponse,
  ProvidersResponse,
  SetPricingRequest,
  StoredProvider,
  TestProviderResult,
  UpdateSettingsRequest,
  UpdateStoredProviderRequest,
} from "@/api/types";

const MODELS = "models";
const PRICING = "pricing";
const SETTINGS = "settings";
const ALIASES = "aliases";
// Deliberately not nested under MODELS: pricing mutations invalidate that key,
// and a price change cannot alter which models a provider serves. Sharing the
// key would fire a live provider call on every save.
const DISCOVERABLE = "discoverable";
const PROVIDERS = "providers";
const STORED_PROVIDERS = "stored-providers";
const METADATA = "model-metadata";
const BUILD = "build";

// How often an open tab asks whether the app it is running is still the one the
// gateway serves. Cheap (a hash of one small file) and only while the tab is
// open, so a minute keeps a deploy from going unnoticed for long.
const BUILD_POLL_MS = 60_000;

export function useModels() {
  return useQuery({
    queryKey: [MODELS],
    queryFn: () => apiFetch<ModelListResponse>("/v1/models"),
    staleTime: 60_000,
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

// Every known provider the add-provider picker can offer, with autofill hints.
// Bundled/static gateway-side, so it never moves within a session.
export function useProviderCatalog() {
  return useQuery({
    queryKey: ["provider-catalog"],
    queryFn: () => apiFetch<KnownProvider[]>("/v1/providers/catalog"),
    staleTime: Infinity,
  });
}

// Providers configured at runtime through the dashboard. Distinct from
// useProviders (static metadata for every configured provider, config + stored
// merged): this is the editable set, with the last 4 of each stored key.
export function useStoredProviders() {
  return useQuery({
    queryKey: [STORED_PROVIDERS],
    queryFn: () => apiFetch<StoredProvider[]>("/v1/provider-credentials"),
    staleTime: 60_000,
  });
}

// A new or changed provider can change which models the catalog and picker
// report, so a credential write invalidates those too.
function invalidateProviderViews(queryClient: ReturnType<typeof useQueryClient>): void {
  void queryClient.invalidateQueries({ queryKey: [STORED_PROVIDERS] });
  void queryClient.invalidateQueries({ queryKey: [PROVIDERS] });
  void queryClient.invalidateQueries({ queryKey: [MODELS] });
  void queryClient.invalidateQueries({ queryKey: [DISCOVERABLE] });
}

export function useCreateStoredProvider() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateStoredProviderRequest) =>
      apiFetch<StoredProvider>("/v1/provider-credentials", { method: "POST", body: JSON.stringify(body) }),
    onSuccess: () => invalidateProviderViews(queryClient),
  });
}

export function useUpdateStoredProvider() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ instance, body }: { instance: string; body: UpdateStoredProviderRequest }) =>
      apiFetch<StoredProvider>(`/v1/provider-credentials/${encodeURIComponent(instance)}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: () => invalidateProviderViews(queryClient),
  });
}

export function useDeleteStoredProvider() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (instance: string) =>
      apiFetch<void>(`/v1/provider-credentials/${encodeURIComponent(instance)}`, { method: "DELETE" }),
    onSuccess: () => invalidateProviderViews(queryClient),
  });
}

// Tests a stored provider's key by listing its models. Read-only on the server,
// so it invalidates nothing.
export function useTestStoredProvider() {
  return useMutation({
    mutationFn: (instance: string) =>
      apiFetch<TestProviderResult>(`/v1/provider-credentials/${encodeURIComponent(instance)}/test`, {
        method: "POST",
      }),
  });
}

// Tests credentials from the add/edit form before they are saved. Nothing is
// persisted server-side, so it invalidates nothing.
export function useTestProviderCredentials() {
  return useMutation({
    mutationFn: (body: CreateStoredProviderRequest) =>
      apiFetch<TestProviderResult>("/v1/provider-credentials/test", { method: "POST", body: JSON.stringify(body) }),
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

// The pricing endpoint caps `limit` at 1000 server-side, so page through it
// rather than truncating: a gateway with a long price history could otherwise
// have older rows silently vanish from the models table.
const PRICING_PAGE_SIZE = 1000;

// Cap the walk so a backend or proxy that ignores `skip` (returning a full page
// every time) can't spin this into an unbounded request loop. 100 pages is 100k
// rows, far beyond any realistic price history.
const PRICING_MAX_PAGES = 100;

async function fetchAllPricing(): Promise<PricingResponse[]> {
  const all: PricingResponse[] = [];
  for (let page = 0; page < PRICING_MAX_PAGES; page += 1) {
    const rows = await apiFetch<PricingResponse[]>(
      `/v1/pricing?skip=${page * PRICING_PAGE_SIZE}&limit=${PRICING_PAGE_SIZE}`,
    );
    all.push(...rows);
    if (rows.length < PRICING_PAGE_SIZE) {
      break;
    }
  }
  return all;
}

export function usePricing() {
  return useQuery({
    queryKey: [PRICING],
    queryFn: fetchAllPricing,
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
