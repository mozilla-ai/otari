import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { ApiError, apiFetch, longRequestSignal } from "@/api/client";
import { isoAgo } from "@/lib/timeRange";
import type {
  AliasResponse,
  ExplainPolicyRequest,
  ExplainPolicyResponse,
  RoutingPolicyResponse,
  SetRoutingPolicyRequest,
  ApiKey,
  Budget,
  BudgetResetLog,
  CreateAliasRequest,
  CreateBudgetRequest,
  CreateKeyRequest,
  CreateKeyResponse,
  CreateStoredProviderRequest,
  DashboardBuild,
  DiscoverableModelsResponse,
  GatewaySettings,
  KnownProvider,
  KnownProviderSummary,
  ModelListResponse,
  ModelMetadataResponse,
  PricingRefreshPreview,
  PricingResponse,
  ProviderHealthResponse,
  ProvidersResponse,
  ReencryptProviderCredentialsResult,
  RotateMasterKeyResponse,
  SetPricingRequest,
  StoredProvider,
  SummaryDimension,
  TestProviderResult,
  TestServiceResponse,
  ToolSettingsResponse,
  ToolsResponse,
  UpdateBudgetRequest,
  UpdateKeyRequest,
  UpdateSettingsRequest,
  UpdateToolSettingsRequest,
  UpdateStoredProviderRequest,
  UpdateUserRequest,
  UsageBucket,
  UsageCount,
  UsageDeleteResult,
  UsageEntry,
  UsageFilters,
  UsageGroupBy,
  UsageGroupedSeries,
  UsageMutationSelection,
  UsageSetPriceRequest,
  UsageSetPriceResult,
  UsageSummary,
  User,
  CreateUserRequest,
} from "@/api/types";

const MODELS = "models";
const PRICING = "pricing";
const SETTINGS = "settings";
const TOOL_SETTINGS = "tool-settings";
const TOOLS = "tools";
const ALIASES = "aliases";
const ROUTING_POLICIES = "routing-policies";
// Deliberately not nested under MODELS: pricing mutations invalidate that key,
// and a price change cannot alter which models a provider serves. Sharing the
// key would fire a live provider call on every save.
const DISCOVERABLE = "discoverable";
const PROVIDERS = "providers";
const PROVIDER_HEALTH = "provider-health";
const STORED_PROVIDERS = "stored-providers";
const METADATA = "model-metadata";
const BUILD = "build";
const KEYS = "keys";
const BUDGETS = "budgets";
const USERS = "users";
const USAGE = "usage";

// How often an open tab asks whether the app it is running is still the one the
// gateway serves. Cheap (a hash of one small file) and only while the tab is
// open, so a minute keeps a deploy from going unnoticed for long.
const BUILD_POLL_MS = 60_000;
// Checking provider health lists models for every configured provider. Keep the
// automatic probe infrequent; operators can still force an immediate re-check.
export const PROVIDER_HEALTH_REFRESH_MS = 60 * 60_000;

// The four queries below are backed by provider or models.dev fan-out
// gateway-side. That is cached and refreshed in the background now, so they are
// normally fast, but they are the ones that go slow when a provider does. The
// global default retries a failed query twice (see provider.tsx), which would
// turn one slow failure into three sequential ones and hold a browser
// connection slot for the whole time; on HTTP/1.1 (6 sockets per origin) enough
// of those queue every other request behind them, including the POST an
// operator just clicked. Failing once and showing the error is the honest
// behavior, and it frees the socket.
const NO_RETRY = { retry: false } as const;

export function useModels() {
  return useQuery({
    ...NO_RETRY,
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
    ...NO_RETRY,
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

// Every known provider the add-provider picker can offer: id + display name
// only. Built gateway-side without importing any provider SDK, so it is cheap
// and never moves within a session (the old full-catalog fetch used to import
// every provider SDK, which lagged the picker; issue #365).
export function useProviderCatalog() {
  return useQuery({
    queryKey: ["provider-catalog"],
    queryFn: () => apiFetch<KnownProviderSummary[]>("/v1/providers/catalog"),
    staleTime: Infinity,
  });
}

// Autofill hints (credential env var, default endpoint, whether a key is
// required) for the one provider the add-provider form has selected. Resolved
// lazily so only the chosen provider's SDK is imported gateway-side; disabled
// until a provider is picked. env_key_present is process-static, so cache it for
// the session like the catalog.
export function useProviderDetail(providerId: string) {
  return useQuery({
    queryKey: ["provider-catalog", providerId],
    queryFn: () => apiFetch<KnownProvider>(`/v1/providers/catalog/${encodeURIComponent(providerId)}`),
    enabled: providerId !== "",
    staleTime: Infinity,
  });
}

// Every configured provider's reachability, for the health monitor. Backed by
// the same model-discovery test path as the per-provider "test connection", so a
// provider is healthy when its credentials can list models. This fans out to
// every configured provider, so automatic checks run at most hourly. The
// response's healthy/total counts are reused by the overview summary tile
// (issue #302).
export function useProviderHealth() {
  return useQuery({
    ...NO_RETRY,
    queryKey: [PROVIDER_HEALTH],
    queryFn: () => apiFetch<ProviderHealthResponse>("/v1/providers/health"),
    staleTime: PROVIDER_HEALTH_REFRESH_MS,
    refetchInterval: PROVIDER_HEALTH_REFRESH_MS,
  });
}

// Force a live re-check of every provider (clears the gateway's discovery cache),
// for an explicit "Refresh" action. Writes the fresh result straight into the
// health query so the table and any summary tile update together.
export function useRecheckProviderHealth() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => apiFetch<ProviderHealthResponse>("/v1/providers/health?refresh=true"),
    onSuccess: (data) => queryClient.setQueryData([PROVIDER_HEALTH], data),
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
  // A credential change can flip a provider's reachability, so the health view
  // must re-check rather than show a verdict from the old key.
  void queryClient.invalidateQueries({ queryKey: [PROVIDER_HEALTH] });
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

export function useReencryptProviderCredentials() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () =>
      apiFetch<ReencryptProviderCredentialsResult>("/v1/provider-credentials/reencrypt", {
        method: "POST",
        signal: longRequestSignal(),
      }),
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
    ...NO_RETRY,
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

export function useRoutingPolicies() {
  return useQuery({
    queryKey: [ROUTING_POLICIES],
    queryFn: () => apiFetch<RoutingPolicyResponse[]>("/v1/routing/policies"),
    staleTime: 60_000,
  });
}

export function useSetRoutingPolicy() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: SetRoutingPolicyRequest) =>
      apiFetch<RoutingPolicyResponse>("/v1/routing/policies", { method: "POST", body: JSON.stringify(body) }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ROUTING_POLICIES] });
      // A policy is listed as a model, so the catalog changes too.
      void queryClient.invalidateQueries({ queryKey: [MODELS] });
    },
  });
}

export function useDeleteRoutingPolicy() {
  const queryClient = useQueryClient();
  return useMutation({
    // Scoped like an alias delete: the same name can exist globally and per user,
    // so a delete must say which. Only a null/absent userId means global, checked
    // explicitly because "" is a legal user id.
    mutationFn: ({ name, userId }: { name: string; userId?: string | null }) => {
      const scope = userId == null ? "" : `?user_id=${encodeURIComponent(userId)}`;
      return apiFetch<void>(`/v1/routing/policies/${encodeURIComponent(name)}${scope}`, { method: "DELETE" });
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ROUTING_POLICIES] });
      void queryClient.invalidateQueries({ queryKey: [MODELS] });
    },
  });
}

/** Compile a policy (saved or draft) without dispatching anything.
 *
 *  A mutation rather than a query: it is an explicit "check this now" action on
 *  inputs the operator is editing, not cacheable server state. */
export function useExplainPolicy() {
  return useMutation({
    mutationFn: (body: ExplainPolicyRequest) =>
      apiFetch<ExplainPolicyResponse>("/v1/routing/policies/explain", {
        method: "POST",
        body: JSON.stringify(body),
      }),
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
    // Scoped: the same name can exist globally and per user, so deleting one
    // must name which. Only a null/absent userId means global; the check is
    // explicit rather than truthy because "" is a legal user id, and treating it
    // as global would delete the wrong row.
    mutationFn: ({ name, userId }: { name: string; userId?: string | null }) => {
      const scope = userId == null ? "" : `?user_id=${encodeURIComponent(userId)}`;
      return apiFetch<void>(`/v1/aliases/${encodeURIComponent(name)}${scope}`, { method: "DELETE" });
    },
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

export function useRotateMasterKey() {
  return useMutation({
    mutationFn: () => apiFetch<RotateMasterKeyResponse>("/v1/settings/master-key/rotate", { method: "POST" }),
  });
}

export function useToolSettings() {
  return useQuery({
    queryKey: [TOOL_SETTINGS],
    queryFn: () => apiFetch<ToolSettingsResponse>("/v1/tool-settings"),
    staleTime: 60_000,
  });
}

// The declaration forms this deployment honours. Depends on tool settings
// (interception, the backend URLs), so a settings save invalidates it.
export function useTools() {
  return useQuery({
    queryKey: [TOOLS],
    queryFn: () => apiFetch<ToolsResponse>("/v1/tools"),
    staleTime: 60_000,
  });
}

export function useUpdateToolSettings() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: UpdateToolSettingsRequest) =>
      apiFetch<ToolSettingsResponse>("/v1/tool-settings", { method: "PATCH", body: JSON.stringify(body) }),
    onSuccess: (data) => {
      queryClient.setQueryData([TOOL_SETTINGS], data);
      // Toggling interception or clearing a backend URL changes which
      // declarations the gateway accepts, so the "how to call" card must refetch.
      void queryClient.invalidateQueries({ queryKey: [TOOLS] });
    },
  });
}

// Probe a (typically unsaved) service URL for reachability. Read-only, so it
// invalidates nothing.
export function useTestService() {
  return useMutation({
    mutationFn: ({ service, url }: { service: string; url: string }) =>
      apiFetch<TestServiceResponse>(`/v1/tool-settings/${encodeURIComponent(service)}/test`, {
        method: "POST",
        body: JSON.stringify({ url }),
      }),
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

// Long deadline: this fetches the upstream snapshot and diffs it against every
// priced model, so it scales with the pricing table rather than with one hop.
export function usePreviewPricingRefresh() {
  return useMutation({
    mutationFn: () =>
      apiFetch<PricingRefreshPreview>("/v1/pricing/refresh", { method: "POST", signal: longRequestSignal() }),
  });
}

export function useConfirmPricingRefresh() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => apiFetch("/v1/pricing/refresh/confirm", { method: "POST" }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [PRICING] });
      void queryClient.invalidateQueries({ queryKey: [MODELS] });
      void queryClient.invalidateQueries({ queryKey: [PROVIDERS] });
    },
  });
}

export function useRejectPricingRefresh() {
  return useMutation({
    mutationFn: () => apiFetch<void>("/v1/pricing/refresh/reject", { method: "POST" }),
  });
}

// The keys endpoint caps `limit` at 1000 server-side; page through it (capped like
// pricing) so a gateway with many keys can't have rows silently vanish from the
// table, and a backend that ignores `skip` can't spin an unbounded loop.
const KEYS_PAGE_SIZE = 1000;
const KEYS_MAX_PAGES = 100;

async function fetchAllKeys(): Promise<ApiKey[]> {
  const all: ApiKey[] = [];
  for (let page = 0; page < KEYS_MAX_PAGES; page += 1) {
    const rows = await apiFetch<ApiKey[]>(`/v1/keys?skip=${page * KEYS_PAGE_SIZE}&limit=${KEYS_PAGE_SIZE}`);
    all.push(...rows);
    if (rows.length < KEYS_PAGE_SIZE) {
      break;
    }
  }
  return all;
}

export function useKeys() {
  return useQuery({
    queryKey: [KEYS],
    queryFn: fetchAllKeys,
    staleTime: 60_000,
  });
}

// Create returns the plaintext key exactly once (in `key`); the caller reveals it
// and must never write the response into the query cache.
export function useCreateKey() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateKeyRequest) =>
      apiFetch<CreateKeyResponse>("/v1/keys", { method: "POST", body: JSON.stringify(body) }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: [KEYS] }),
  });
}

export function useUpdateKey() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: UpdateKeyRequest }) =>
      apiFetch<ApiKey>(`/v1/keys/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(body) }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: [KEYS] }),
  });
}

// Regenerate: a new secret for the same key row. The old secret stops working
// immediately. Returns the new plaintext once, like create.
export function useRotateKey() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<CreateKeyResponse>(`/v1/keys/${encodeURIComponent(id)}/rotate`, { method: "POST" }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: [KEYS] }),
  });
}

export function useDeleteKey() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiFetch<void>(`/v1/keys/${encodeURIComponent(id)}`, { method: "DELETE" }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: [KEYS] }),
  });
}

// The budgets endpoint caps `limit` at 1000 server-side; page through it (capped
// like keys/pricing) so a gateway with many budgets can't have rows silently
// vanish, and a backend that ignores `skip` can't spin an unbounded loop.
const BUDGETS_PAGE_SIZE = 1000;
const BUDGETS_MAX_PAGES = 100;

async function fetchAllBudgets(): Promise<Budget[]> {
  const all: Budget[] = [];
  for (let page = 0; page < BUDGETS_MAX_PAGES; page += 1) {
    const rows = await apiFetch<Budget[]>(`/v1/budgets?skip=${page * BUDGETS_PAGE_SIZE}&limit=${BUDGETS_PAGE_SIZE}`);
    all.push(...rows);
    if (rows.length < BUDGETS_PAGE_SIZE) {
      break;
    }
  }
  return all;
}

export function useBudgets() {
  return useQuery({
    queryKey: [BUDGETS],
    queryFn: fetchAllBudgets,
    staleTime: 60_000,
  });
}

// Per-user reset history for one budget. Enabled only once a budget id is set
// (the drill-down is opened), so the query does not fire for the whole list.
export function useBudgetResetLogs(budgetId: string | null) {
  return useQuery({
    queryKey: [BUDGETS, budgetId, "reset-logs"],
    queryFn: () => apiFetch<BudgetResetLog[]>(`/v1/budgets/${encodeURIComponent(budgetId as string)}/reset-logs`),
    enabled: budgetId !== null,
    staleTime: 60_000,
  });
}

export function useCreateBudget() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateBudgetRequest) =>
      apiFetch<Budget>("/v1/budgets", { method: "POST", body: JSON.stringify(body) }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: [BUDGETS] }),
  });
}

export function useUpdateBudget() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: UpdateBudgetRequest }) =>
      apiFetch<Budget>(`/v1/budgets/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(body) }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: [BUDGETS] }),
  });
}

export function useDeleteBudget() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiFetch<void>(`/v1/budgets/${encodeURIComponent(id)}`, { method: "DELETE" }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: [BUDGETS] }),
  });
}

// The users endpoint caps `limit` at 1000 server-side; page through it (capped
// like keys/budgets) so a gateway with many users can't have rows silently
// vanish, and a backend that ignores `skip` can't spin an unbounded loop.
const USERS_PAGE_SIZE = 1000;
const USERS_MAX_PAGES = 100;

async function fetchAllUsers(): Promise<User[]> {
  const all: User[] = [];
  for (let page = 0; page < USERS_MAX_PAGES; page += 1) {
    const rows = await apiFetch<User[]>(`/v1/users?skip=${page * USERS_PAGE_SIZE}&limit=${USERS_PAGE_SIZE}`);
    all.push(...rows);
    if (rows.length < USERS_PAGE_SIZE) {
      break;
    }
  }
  return all;
}

export function useUsers() {
  return useQuery({
    queryKey: [USERS],
    queryFn: fetchAllUsers,
    staleTime: 60_000,
  });
}

// Assigning a budget to a user changes that budget's usage rollup, so a user
// write invalidates the budgets list too.
function invalidateUserViews(queryClient: ReturnType<typeof useQueryClient>): void {
  void queryClient.invalidateQueries({ queryKey: [USERS] });
  void queryClient.invalidateQueries({ queryKey: [BUDGETS] });
}

export function useCreateUser() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateUserRequest) =>
      apiFetch<User>("/v1/users", { method: "POST", body: JSON.stringify(body) }),
    onSuccess: () => invalidateUserViews(queryClient),
  });
}

export function useUpdateUser() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: UpdateUserRequest }) =>
      apiFetch<User>(`/v1/users/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(body) }),
    onSuccess: () => invalidateUserViews(queryClient),
  });
}

export function useDeleteUser() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiFetch<void>(`/v1/users/${encodeURIComponent(id)}`, { method: "DELETE" }),
    onSuccess: () => {
      invalidateUserViews(queryClient);
      // Deleting a user deactivates its keys server-side.
      void queryClient.invalidateQueries({ queryKey: [KEYS] });
    },
  });
}

// ---------- activity / request log ----------

// Serialize the activity-log filters into query params, dropping empty values so
// the query key and the request URL stay stable across renders.
//
// Every field of UsageFilters has to appear here. The list, the row count, and the
// summary all go through this one function, so a field left out does not fail
// loudly: the page still shows its filter chip and the URL still carries the value,
// while the request goes out unfiltered and the table quietly shows everything.
function usageParams(filters: UsageFilters): URLSearchParams {
  const params = new URLSearchParams();
  // A multi-value filter goes on the wire as a repeated param (the analytics
  // endpoints match any of them); an empty array is no filter at all, not a
  // filter matching nothing.
  const appendAll = (key: string, value: string | string[] | undefined) => {
    for (const one of typeof value === "string" ? [value] : (value ?? [])) {
      if (one) params.append(key, one);
    }
  };
  if (filters.start_date) params.set("start_date", filters.start_date);
  if (filters.end_date) params.set("end_date", filters.end_date);
  if (filters.status) params.set("status", filters.status);
  appendAll("model", filters.model);
  if (filters.endpoint) params.set("endpoint", filters.endpoint);
  if (filters.provider) params.set("provider", filters.provider);
  appendAll("user_id", filters.user_id);
  appendAll("api_key_id", filters.api_key_id);
  if (filters.source) params.set("source", filters.source);
  if (filters.source_label) params.set("source_label", filters.source_label);
  if (filters.tool) params.set("tool", filters.tool);
  if (filters.priced !== undefined) params.set("priced", String(filters.priced));
  if (filters.counts_toward_budget !== undefined) {
    params.set("counts_toward_budget", String(filters.counts_toward_budget));
  }
  return params;
}

// One page of usage-log rows for the Activity viewer, newest first.
// `placeholderData: keepPreviousData` keeps the current page on screen while the
// next loads, so paging does not flash empty.
export function useUsageLogs(filters: UsageFilters, page: number, pageSize: number) {
  return useQuery({
    queryKey: [USAGE, "list", filters, page, pageSize],
    queryFn: () => {
      const params = usageParams(filters);
      params.set("skip", String(page * pageSize));
      params.set("limit", String(pageSize));
      return apiFetch<UsageEntry[]>(`/v1/usage?${params.toString()}`);
    },
    placeholderData: keepPreviousData,
    // A request log moves constantly; keep it fresh but don't refetch on every focus.
    staleTime: 10_000,
  });
}

// Total rows matching the same filters, for the paginator's "N of M". A separate
// request so /v1/usage stays a bare array; run alongside the list.
export function useUsageCount(filters: UsageFilters, enabled = true) {
  return useQuery({
    queryKey: [USAGE, "count", filters],
    queryFn: () => apiFetch<UsageCount>(`/v1/usage/count?${usageParams(filters).toString()}`),
    enabled,
    placeholderData: keepPreviousData,
    staleTime: 10_000,
  });
}

// How often the rolling failure count re-reads. A dropped-traffic signal is only
// useful if it moves while the operator watches it.
const FAILURE_COUNT_POLL_MS = 60_000;

// Requests that failed within the last `windowSeconds`, as a live count. The
// window is resolved inside the query function, not in the key, for two reasons:
// the key stays stable (a "now"-derived key would mint a new cache entry on every
// render), and every refetch re-anchors, so a tab left open keeps reporting the
// last hour rather than quietly widening to the last hour and a half.
//
// Scoped to `source: "gateway"`: imported usage can carry status=error too (the
// external-events API accepts it), and an imported session's failures are not this
// gateway dropping traffic. Counting them would make the signal cry wolf.
export function useFailureCount(windowSeconds: number, enabled = true) {
  return useQuery({
    queryKey: [USAGE, "count", "failures", windowSeconds],
    queryFn: () => {
      const filters: UsageFilters = {
        status: "error",
        source: "gateway",
        start_date: isoAgo(windowSeconds),
      };
      return apiFetch<UsageCount>(`/v1/usage/count?${usageParams(filters).toString()}`);
    },
    enabled,
    refetchInterval: FAILURE_COUNT_POLL_MS,
    refetchOnWindowFocus: true,
    staleTime: 0,
    // A failed count is not worth surfacing: it sits beside its own alarm, and
    // the next poll retries anyway.
    retry: false,
  });
}

// The rows of one or more request groups: every attempt a routed request made,
// which is what turns "attempt 1 of 2, failed" into "and here is what served it".
// A plan is capped at a handful of candidates and the activity table pages at a
// hundred rows, so this leaves an order of magnitude of headroom over the largest
// batch either caller can ask for. It is deliberately not a tight bound: nothing
// downstream detects truncation, so the limit has to be one no real page reaches.
const REQUEST_GROUP_PAGE_LIMIT = 1000;

// Fetched as a batch (the endpoint takes a repeatable `request_group_id`) so a
// page of the activity log costs one lookup rather than one per row. The key
// sorts its ids so two callers asking for the same set share a cache entry.
export function useRequestGroups(groupIds: readonly string[]) {
  const ids = [...new Set(groupIds)].sort();
  return useQuery({
    queryKey: [USAGE, "groups", ids],
    queryFn: () => {
      const params = new URLSearchParams();
      for (const id of ids) params.append("request_group_id", id);
      params.set("limit", String(REQUEST_GROUP_PAGE_LIMIT));
      return apiFetch<UsageEntry[]>(`/v1/usage?${params.toString()}`);
    },
    enabled: ids.length > 0,
    placeholderData: keepPreviousData,
    // A group is immutable once its request finished, so the only reason to
    // refetch is a group that was still in flight when it was first read.
    staleTime: 30_000,
  });
}

// Delete imported usage rows by selection (ids or by_filter). Only rows the
// server treats as imported (counts_toward_budget = false) are removed; every
// usage view is invalidated so the list, count, and analytics refresh.
//
// Given the long deadline, not apiFetch's default: a by_filter delete is one
// unbounded DELETE server-side, so its duration tracks the number of matched
// rows. Timing out here would report failure for a delete that committed anyway.
export function useDeleteUsage() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: UsageMutationSelection) =>
      apiFetch<UsageDeleteResult>("/v1/usage", {
        method: "DELETE",
        body: JSON.stringify(body),
        signal: longRequestSignal(),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [USAGE] });
    },
  });
}

// Set the cost of imported usage rows from manual per-1M rates. Long deadline
// for the same reason as the delete: the server reprices every matched row.
export function useSetUsagePrice() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: UsageSetPriceRequest) =>
      apiFetch<UsageSetPriceResult>("/v1/usage/set-price", {
        method: "POST",
        body: JSON.stringify(body),
        signal: longRequestSignal(),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [USAGE] });
    },
  });
}

// ---------- usage analytics summary ----------

// Pass this as `dimensions` to read only `totals` / `series`. Named rather than a
// bare `[]` at each call site so it is clear the omission is deliberate.
export const NO_BREAKDOWNS: SummaryDimension[] = [];

// Aggregated spend/tokens/requests for the Usage page. Shares the activity
// filter serialization and adds the time-series bucket. `enabled` lets a caller
// skip the request (e.g. the previous-period query when the range is unbounded,
// so there is nothing to compare against). staleTime is longer than the live
// Activity log's: an aggregate over days moves slowly and need not refetch on
// every focus.
//
// `dimensions` names the breakdowns to compute. Each one costs the server a
// separate GROUP BY over the window, and several callers here read only `totals`
// or `series` (tiles, timeline context, the previous-period comparison), so they
// pass `[]` and skip all of them. Omitting the argument keeps the server default
// (every breakdown).
export function useUsageSummary(
  filters: UsageFilters,
  bucket: UsageBucket,
  dimensions?: SummaryDimension[],
  enabled = true,
) {
  return useQuery({
    queryKey: [USAGE, "summary", filters, bucket, dimensions ?? "all"],
    queryFn: () => {
      const params = usageParams(filters);
      params.set("bucket", bucket);
      // A repeated query param has no empty-list form, so an empty selection goes
      // on the wire as the server's `none` sentinel.
      if (dimensions) {
        for (const dimension of dimensions.length > 0 ? dimensions : ["none"]) {
          params.append("dimensions", dimension);
        }
      }
      return apiFetch<UsageSummary>(`/v1/usage/summary?${params.toString()}`);
    },
    enabled,
    placeholderData: keepPreviousData,
    staleTime: 30_000,
  });
}

// A per-group time series for the stacked analytics chart (top groups by spend
// plus an "other" fold). Only fetched while a group-by dimension is active, so
// the ungrouped view costs nothing extra. Caching mirrors useUsageSummary.
export function useUsageGroupedSeries(
  filters: UsageFilters,
  bucket: UsageBucket,
  groupBy: UsageGroupBy | null,
  enabled = true,
) {
  return useQuery({
    queryKey: [USAGE, "series", filters, bucket, groupBy],
    queryFn: () => {
      const params = usageParams(filters);
      params.set("bucket", bucket);
      params.set("group_by", groupBy as string);
      return apiFetch<UsageGroupedSeries>(`/v1/usage/series?${params.toString()}`);
    },
    enabled: enabled && groupBy !== null,
    placeholderData: keepPreviousData,
    staleTime: 30_000,
    // A 404 is version skew (a gateway older than this dashboard, e.g. not yet
    // restarted onto the build that ships it); retrying cannot fix that, and
    // the page falls back to the ungrouped view with a notice instead.
    retry: (failureCount, error) => !(error instanceof ApiError && error.status === 404) && failureCount < 3,
  });
}
