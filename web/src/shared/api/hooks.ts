import {
  keepPreviousData,
  useMutation,
  useQueries,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query"
import type {
  AcceptInvitationResult,
  ActivationApiKey,
  AliasResponse,
  ApiKey,
  Budget,
  BudgetResetLog,
  CallerOrganizationMembership,
  CreateAliasRequest,
  CreateBudgetRequest,
  CreateKeyRequest,
  CreateKeyResponse,
  CreateOrganizationGuardrailRequest,
  CreateOrganizationMemberRequest,
  CreateOrganizationMemberResult,
  CreateOrganizationPricingOverride,
  CreateOrganizationRequest,
  CreateScopedBudgetRequest,
  CreateSearchToolRequest,
  CreateStoredProviderRequest,
  CreateUserRequest,
  CreateWorkspaceBudgetDefaultRequest,
  CreateWorkspaceMcpServerRequest,
  CreateWorkspaceRequest,
  DashboardBuild,
  DiscoverableModelsResponse,
  ExplainPolicyRequest,
  ExplainPolicyResponse,
  GatewayHealth,
  GatewaySettings,
  InFlightResponse,
  InvitationPreview,
  InviteOrganizationMemberRequest,
  InviteOrganizationMemberResult,
  KnownProvider,
  KnownProviderSummary,
  MailSettings,
  MaintenanceMode,
  ModelListResponse,
  ModelMetadataResponse,
  Organization,
  OrganizationContext,
  OrganizationGuardrail,
  OrganizationMember,
  OrganizationPricingOverride,
  Passkey,
  PasskeysResponse,
  PasswordResponse,
  PricingRefreshPreview,
  PricingResponse,
  ProviderHealthResponse,
  ProvidersResponse,
  RankCandidatesRequest,
  RankCandidatesResponse,
  ReencryptProviderCredentialsResult,
  RenamePasskeyRequest,
  RequestPasswordResetResponse,
  ResendVerificationResponse,
  ResetPasswordRequest,
  RotateMasterKeyResponse,
  RouterStatus,
  RoutingPolicyResponse,
  ScopedBudget,
  SearchProviderInfo,
  SearchToolsResponse,
  SendTestMailRequest,
  SendTestMailResponse,
  SetPasswordRequest,
  SetPricingRequest,
  SetRoutingPolicyRequest,
  SignupRequest,
  SignupResponse,
  StoredProvider,
  StoredSearchTool,
  SummaryDimension,
  SwitchOrganizationRequest,
  TestProviderResult,
  TestServiceResponse,
  ToolSettingsResponse,
  ToolsResponse,
  UpdateBudgetRequest,
  UpdateKeyRequest,
  UpdateOrganizationGuardrailRequest,
  UpdateOrganizationMemberRequest,
  UpdateOrganizationPricingOverride,
  UpdateOrganizationRequest,
  UpdateScopedBudgetRequest,
  UpdateSearchToolRequest,
  UpdateSettingsRequest,
  UpdateStoredProviderRequest,
  UpdateToolSettingsRequest,
  UpdateUserRequest,
  UpdateWorkspaceBudgetDefaultRequest,
  UpdateWorkspaceCodeExecutionPolicyRequest,
  UpdateWorkspaceMcpServerRequest,
  UpdateWorkspaceRequest,
  UpdateWorkspaceWebSearchConfigRequest,
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
  VerifyEmailResponse,
  Workspace,
  WorkspaceActivation,
  WorkspaceBudgetDefault,
  WorkspaceCodeExecutionPolicy,
  WorkspaceMcpServer,
  WorkspaceMcpServers,
  WorkspaceMember,
  WorkspaceMemberRole,
  WorkspaceWebSearchConfig,
} from "@/client"
import { ApiError, apiFetch, longRequestSignal } from "@/shared/api/client"
import { isoAgo } from "@/shared/helpers/timeRange"
import { createPasskey } from "@/shared/helpers/webauthn"

const MODELS = "models"
const PRICING = "pricing"
const SETTINGS = "settings"
const MAIL_SETTINGS = "mail-settings"
const MAINTENANCE_MODE = "maintenance-mode"
// One indexed single-row read, and only while the settings page is mounted.
const MAINTENANCE_MODE_POLL_MS = 30_000
const TOOL_SETTINGS = "tool-settings"
const TOOLS = "tools"
const SEARCH_TOOLS = "search-tools"
const SEARCH_PROVIDERS = "search-providers"
const ALIASES = "aliases"
const ROUTING_POLICIES = "routing-policies"
const ROUTER_STATUS = "router-status"
// Deliberately not nested under MODELS: pricing mutations invalidate that key,
// and a price change cannot alter which models a provider serves. Sharing the
// key would fire a live provider call on every save.
const DISCOVERABLE = "discoverable"
const PROVIDERS = "providers"
const PROVIDER_HEALTH = "provider-health"
const STORED_PROVIDERS = "stored-providers"
const METADATA = "model-metadata"
const BUILD = "build"
const HEALTH = "health"
const KEYS = "keys"
const BUDGETS = "budgets"
const SCOPED_BUDGETS = "scoped-budgets"
const USERS = "users"
const USAGE = "usage"
const ORGANIZATIONS = "organizations"
// Deliberately its own key rather than a child of ORGANIZATIONS: switching
// organizations invalidates both, but a role change invalidates only the roster,
// and nesting would re-read the context (and every page gated on it) as well.
const ORGANIZATION_MEMBERS = "organization-members"
// Its own key rather than a child of ORGANIZATIONS, for the reason the members
// key is: the organization context is read on nearly every page, and a rate
// edit should not make all of them refetch.
const ORGANIZATION_PRICING = "organization-pricing"
const ORGANIZATION_GUARDRAILS = "organization-guardrails"
const WORKSPACES = "workspaces"
// The first-request setup guide's state. Its own key rather than a child of
// WORKSPACES: the guide polls while it is on screen, and nesting it would make
// every one of those ticks invalidate (or be invalidated by) the workspace list
// and its rosters.
const ACTIVATION = "workspace-activation"
// The signed-in identity's own passkeys. Its own key and not a child of any
// organization key: a passkey belongs to a person, not to the organization they
// happen to be acting in, and switching organizations does not change the list.
const PASSKEYS = "passkeys"

// How often an open tab asks whether the app it is running is still the one the
// gateway serves. Cheap (a hash of one small file) and only while the tab is
// open, so a minute keeps a deploy from going unnoticed for long.
const BUILD_POLL_MS = 60_000
// How often the hybrid landing page re-asks whether this gateway is up and can
// still reach its control plane. That pair is the only thing on that page which
// changes, and it is the reason to leave the page open, so it ticks faster than
// the build check. The gateway bounds its own upstream probe (`resolve_timeout_ms`),
// so a stalled control plane answers "no" rather than piling up requests.
const HEALTH_POLL_MS = 15_000
// Checking provider health lists models for every configured provider. Keep the
// automatic probe infrequent; operators can still force an immediate re-check.
export const PROVIDER_HEALTH_REFRESH_MS = 60 * 60_000

// The four queries below are backed by provider or models.dev fan-out
// gateway-side. That is cached and refreshed in the background now, so they are
// normally fast, but they are the ones that go slow when a provider does. The
// global default retries a failed query twice (see provider.tsx), which would
// turn one slow failure into three sequential ones and hold a browser
// connection slot for the whole time; on HTTP/1.1 (6 sockets per origin) enough
// of those queue every other request behind them, including the POST an
// operator just clicked. Failing once and showing the error is the honest
// behavior, and it frees the socket.
const NO_RETRY = { retry: false } as const

export function useModels() {
  return useQuery({
    ...NO_RETRY,
    queryKey: [MODELS],
    queryFn: () => apiFetch<ModelListResponse>("/v1/models"),
    staleTime: 60_000,
  })
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
  })
}

/**
 * Whether this gateway is answering, and whether it can reach its control plane.
 *
 * `/health` is public and served in both modes, which is what makes it the one
 * read a hybrid gateway's landing page can make: it hosts no management API, so
 * every other endpoint the dashboard knows is a 404 there. A failure is the
 * answer here rather than an error to retry past, so the page can say the
 * gateway is not responding on the first attempt instead of ~three requests
 * later.
 */
export function useGatewayHealth() {
  return useQuery({
    ...NO_RETRY,
    queryKey: [HEALTH],
    queryFn: () => apiFetch<GatewayHealth>("/health"),
    refetchInterval: HEALTH_POLL_MS,
    // A tab left open in the background holds the stalest answer of all, so ask
    // again the moment someone looks at it.
    refetchOnWindowFocus: true,
    staleTime: 0,
  })
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
    queryFn: () =>
      apiFetch<DiscoverableModelsResponse>("/v1/models/discoverable"),
    staleTime: 5 * 60_000,
  })
}

// Static metadata for every configured provider: capabilities, doc and pricing
// links, display name. Network-free gateway-side (bundled datasets), so it does
// not move within a session; kept fresh for a few minutes like discovery.
export function useProviders() {
  return useQuery({
    queryKey: [PROVIDERS],
    queryFn: () => apiFetch<ProvidersResponse>("/v1/providers"),
    staleTime: 5 * 60_000,
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
export function useModelMetadata() {
  return useQuery({
    ...NO_RETRY,
    queryKey: [METADATA],
    queryFn: () => apiFetch<ModelMetadataResponse>("/v1/models/metadata"),
    staleTime: 10 * 60_000,
  })
}

// Deliberately unscoped, and deliberately without the parameter the endpoint
// would accept: the gateway stores every alias in the default workspace because
// resolution reads a process-wide name-keyed cache, so a filtered list would
// hide live aliases. Leaving the argument here would be a loaded gun beside the
// comment explaining why it must not be fired.
export function useAliases() {
  return useQuery({
    queryKey: [ALIASES],
    queryFn: () => apiFetch<AliasResponse[]>("/v1/aliases"),
    staleTime: 60_000,
  })
}

// Unscoped for the same reason as `useAliases` above.
export function useRoutingPolicies() {
  return useQuery({
    queryKey: [ROUTING_POLICIES],
    queryFn: () => apiFetch<RoutingPolicyResponse[]>("/v1/routing/policies"),
    staleTime: 60_000,
  })
}

export function useSetRoutingPolicy() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: SetRoutingPolicyRequest) =>
      apiFetch<RoutingPolicyResponse>("/v1/routing/policies", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ROUTING_POLICIES] })
      // A policy is listed as a model, so the catalog changes too.
      void queryClient.invalidateQueries({ queryKey: [MODELS] })
    },
  })
}

export function useDeleteRoutingPolicy() {
  const queryClient = useQueryClient()
  return useMutation({
    // Scoped like an alias delete: the same name can exist globally and per user,
    // so a delete must say which. Only a null/absent userId means global, checked
    // explicitly because "" is a legal user id.
    mutationFn: ({
      name,
      userId,
    }: {
      name: string
      userId?: string | null
    }) => {
      const scope =
        userId == null ? "" : `?user_id=${encodeURIComponent(userId)}`
      return apiFetch<void>(
        `/v1/routing/policies/${encodeURIComponent(name)}${scope}`,
        { method: "DELETE" },
      )
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ROUTING_POLICIES] })
      void queryClient.invalidateQueries({ queryKey: [MODELS] })
    },
  })
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
  })
}

// --- Learned routing ------------------------------------------------------

/** How warm a user's routing memory is.
 *
 *  Keyed by user because warmth is per user: the records hold that user's
 *  prompts, so a global learned policy warms once per caller. Disabled until a
 *  user is chosen rather than defaulting to one, because "whose memory" has no
 *  sensible default.
 */
export function useRouterStatus(userId: string | null) {
  return useQuery({
    queryKey: [ROUTER_STATUS, userId],
    queryFn: () =>
      apiFetch<RouterStatus>(
        `/v1/routing/status?user_id=${encodeURIComponent(userId ?? "")}`,
      ),
    enabled: userId !== null && userId !== "",
    staleTime: 30_000,
  })
}

/** Record how well each candidate did, which is what the router later votes over. */
export function useRankCandidates() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: RankCandidatesRequest) =>
      apiFetch<RankCandidatesResponse>("/v1/routing/preferences/rank", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      // One more example may have crossed the seed count, which changes whether
      // the policy routes at all.
      void queryClient.invalidateQueries({ queryKey: [ROUTER_STATUS] })
    },
  })
}

export function useCreateAlias() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateAliasRequest) =>
      apiFetch<AliasResponse>("/v1/aliases", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ALIASES] })
      // An alias is listed as a model, so the catalog changes too.
      void queryClient.invalidateQueries({ queryKey: [MODELS] })
    },
  })
}

export function useDeleteAlias() {
  const queryClient = useQueryClient()
  return useMutation({
    // Scoped: the same name can exist globally and per user, so deleting one
    // must name which. Only a null/absent userId means global; the check is
    // explicit rather than truthy because "" is a legal user id, and treating it
    // as global would delete the wrong row.
    mutationFn: ({
      name,
      userId,
    }: {
      name: string
      userId?: string | null
    }) => {
      const scope =
        userId == null ? "" : `?user_id=${encodeURIComponent(userId)}`
      return apiFetch<void>(`/v1/aliases/${encodeURIComponent(name)}${scope}`, {
        method: "DELETE",
      })
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ALIASES] })
      void queryClient.invalidateQueries({ queryKey: [MODELS] })
    },
  })
}

export function useSettings() {
  return useQuery({
    queryKey: [SETTINGS],
    queryFn: () => apiFetch<GatewaySettings>("/v1/settings"),
    staleTime: 60_000,
  })
}

export function useUpdateSettings() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: UpdateSettingsRequest) =>
      apiFetch<GatewaySettings>("/v1/settings", {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: (data) => {
      queryClient.setQueryData([SETTINGS], data)
      // Toggling discovery changes which models the catalog and picker report.
      void queryClient.invalidateQueries({ queryKey: [MODELS] })
      void queryClient.invalidateQueries({ queryKey: [DISCOVERABLE] })
    },
  })
}

/**
 * Whether this deployment is refusing new dashboard sign-ins.
 *
 * Not read from the bootstrap, which carries the same flag: that one is fetched
 * once per page load and cached for the life of the tab, which is right for the
 * sign-in screen (it renders before there is anything to poll with) and wrong
 * for the switch that changes it. This is the live value the card renders.
 */
export function useMaintenanceMode() {
  return useQuery({
    queryKey: [MAINTENANCE_MODE],
    queryFn: () => apiFetch<MaintenanceMode>("/v1/settings/maintenance-mode"),
    // Polled and refreshed on focus, unlike every other settings read here.
    // A `staleTime` alone schedules nothing, and this app turns
    // `refetchOnWindowFocus` off globally, so a card left open would keep
    // showing whatever it fetched on mount. That is the one wrong answer this
    // card can give: another operator or an API client can flip the freeze, and
    // reporting a deployment open when it is frozen (or frozen when it is back)
    // is worse than a moment's blank. Same treatment as `useDashboardBuild`,
    // for the same reason: the value changes underneath the tab.
    refetchInterval: MAINTENANCE_MODE_POLL_MS,
    refetchOnWindowFocus: true,
    staleTime: 0,
  })
}

/**
 * Freeze or unfreeze dashboard sign-ins.
 *
 * Nothing else is invalidated: the freeze changes no data any other page shows,
 * and it deliberately does not touch the caller's own session, so the tab that
 * flipped it keeps working either way.
 */
export function useSetMaintenanceMode() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (enabled: boolean) =>
      apiFetch<MaintenanceMode>("/v1/settings/maintenance-mode", {
        method: "PATCH",
        body: JSON.stringify({ enabled }),
      }),
    onSuccess: (data) => {
      queryClient.setQueryData([MAINTENANCE_MODE], data)
    },
  })
}

export function useRotateMasterKey() {
  return useMutation({
    mutationFn: () =>
      apiFetch<RotateMasterKeyResponse>("/v1/settings/master-key/rotate", {
        method: "POST",
      }),
  })
}

/**
 * Set or change the password the signed-in identity uses to reach this
 * dashboard (`PUT /v1/auth/password`).
 *
 * Always the caller's own identity: the endpoint takes no id, and there is
 * deliberately no way for an operator to set somebody else's password. The
 * first call on a deployment supplies an address as well, which is the act that
 * claims it and retires master-key sign-in (otari-ai#1716).
 *
 * Two things this changes are cached elsewhere, and they are cached
 * differently. The bootstrap's `sign_in_methods` is a context read once per
 * load rather than a query, so no invalidation could reach it: the caller
 * reports the claim through `useRetireMasterKeySignIn` instead. The roster is
 * an ordinary query, and a claim writes `user.email` from null to the address,
 * so the Members page would otherwise show the row it fetched before the claim
 * for the rest of its `staleTime`. That one is invalidated here.
 *
 * Every *other* session this identity holds is revoked server-side; this one is
 * kept, so no 401 follows.
 */
export function useSetPassword() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: SetPasswordRequest) =>
      apiFetch<PasswordResponse>("/v1/auth/password", {
        method: "PUT",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATION_MEMBERS] })
    },
  })
}

/**
 * The signed-in identity's own passkeys, for the account page.
 *
 * Only ever the caller's own: the endpoint scopes to the session's identity, so
 * there is nothing to pass and nothing to filter here.
 *
 * `NO_RETRY` because the two ways this fails are both settled answers rather
 * than blips: a deployment with no relying party configured refuses with a 503
 * naming the setting, and that will refuse again on a retry.
 */
export function usePasskeys() {
  return useQuery({
    queryKey: [PASSKEYS],
    queryFn: () => apiFetch<PasskeysResponse>("/v1/auth/webauthn/credentials"),
    staleTime: 60_000,
    ...NO_RETRY,
  })
}

/**
 * Register a passkey: two calls with a browser ceremony between them.
 *
 * The whole ceremony is one mutation rather than two hooks and a component
 * holding the options in state. The options are useless on their own, they
 * expire, and the challenge they carry is spent by the second call, so exposing
 * the halves separately would let a component keep something that is already
 * void.
 *
 * A dismissed prompt throws `PasskeyCancelledError` out of `createPasskey`, and
 * is deliberately left to reach the caller: it is not a failed registration and
 * the card says nothing about it.
 *
 * Registering the first passkey is also what makes the gateway start publishing
 * `passkey` in `sign_in_methods`. That correction is not made here: the
 * deployment bootstrap is a context rather than a query, so it is reported by
 * the card through `useOfferPasskeySignIn`, exactly as claiming a deployment is
 * reported through `useRetireMasterKeySignIn` from `PasswordCard`.
 */
export function useRegisterPasskey() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: async (name: string | undefined) => {
      const options = await apiFetch<Record<string, unknown>>(
        "/v1/auth/webauthn/register/options",
        { method: "POST" },
      )
      const credential = await createPasskey(
        options as Parameters<typeof createPasskey>[0],
      )
      return apiFetch<Passkey>("/v1/auth/webauthn/register", {
        method: "POST",
        body: JSON.stringify({ credential, name }),
      })
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [PASSKEYS] })
    },
  })
}

/** Relabel one of the caller's passkeys, which is all that is editable. */
export function useRenamePasskey() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) =>
      apiFetch<Passkey>(`/v1/auth/webauthn/credentials/${id}`, {
        method: "PATCH",
        body: JSON.stringify({ name } satisfies RenamePasskeyRequest),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [PASSKEYS] })
    },
  })
}

/** Remove one of the caller's passkeys. */
export function useDeletePasskey() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/v1/auth/webauthn/credentials/${id}`, {
        method: "DELETE",
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [PASSKEYS] })
    },
  })
}

/**
 * The deployment's outgoing-mail configuration, for the Settings page.
 *
 * Distinct from `useDeployment().mail_ready`, which the shell already carries:
 * that one boolean gates a mail-dependent affordance anywhere in the app, while
 * this reports *why* mail is (un)available and is worth a request only on the
 * page that shows it.
 */
export function useMailSettings() {
  return useQuery({
    queryKey: [MAIL_SETTINGS],
    queryFn: () => apiFetch<MailSettings>("/v1/settings/mail"),
    staleTime: 60_000,
  })
}

/**
 * Sends a real message to prove the transport works.
 *
 * Nothing is invalidated on success: a test send changes no server state, and
 * the outcome lives in the mutation's own result. A failure comes back two
 * ways, and the page distinguishes them: a 200 with `ok: false` is a configured
 * transport that refused, while a 503 is a deployment with no transport at all.
 */
export function useSendTestMail() {
  return useMutation({
    mutationFn: (body: SendTestMailRequest) =>
      apiFetch<SendTestMailResponse>("/v1/settings/mail/test", {
        method: "POST",
        body: JSON.stringify(body),
      }),
  })
}

export function useToolSettings() {
  return useQuery({
    queryKey: [TOOL_SETTINGS],
    queryFn: () => apiFetch<ToolSettingsResponse>("/v1/tool-settings"),
    staleTime: 60_000,
  })
}

// The declaration forms this deployment honors. Depends on tool settings
// (interception, the backend URLs), so a settings save invalidates it.
export function useTools() {
  return useQuery({
    queryKey: [TOOLS],
    queryFn: () => apiFetch<ToolsResponse>("/v1/tools"),
    staleTime: 60_000,
  })
}

export function useUpdateToolSettings() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: UpdateToolSettingsRequest) =>
      apiFetch<ToolSettingsResponse>("/v1/tool-settings", {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: (data) => {
      queryClient.setQueryData([TOOL_SETTINGS], data)
      // Toggling interception or clearing a backend URL changes which
      // declarations the gateway accepts, so the "how to call" card must refetch.
      void queryClient.invalidateQueries({ queryKey: [TOOLS] })
      // A searxng search tool with no api_base of its own inherits web_search_url,
      // which this PATCH may have just changed, so the endpoint a blank box
      // resolves to (and whether one is required at all) has to be re-read.
      void queryClient.invalidateQueries({ queryKey: [SEARCH_PROVIDERS] })
    },
  })
}

// Search tools served by POST /v1/search: the editable rows plus the read-only
// config-file entries, so the page can show every tool a caller could name.
export function useSearchTools() {
  return useQuery({
    queryKey: [SEARCH_TOOLS],
    queryFn: () => apiFetch<SearchToolsResponse>("/v1/search-tools"),
    staleTime: 60_000,
  })
}

// Which search providers this build supports, and what each one needs, so the
// add form asks for a key or a backend URL only when the provider requires it.
export function useSearchProviders() {
  return useQuery({
    queryKey: [SEARCH_PROVIDERS],
    queryFn: () => apiFetch<SearchProviderInfo[]>("/v1/search-tools/providers"),
    staleTime: 300_000,
  })
}

export function useCreateSearchTool() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateSearchToolRequest) =>
      apiFetch<StoredSearchTool>("/v1/search-tools", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [SEARCH_TOOLS] }),
  })
}

export function useUpdateSearchTool() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      name,
      body,
    }: {
      name: string
      body: UpdateSearchToolRequest
    }) =>
      apiFetch<StoredSearchTool>(
        `/v1/search-tools/${encodeURIComponent(name)}`,
        {
          method: "PATCH",
          body: JSON.stringify(body),
        },
      ),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [SEARCH_TOOLS] }),
  })
}

export function useDeleteSearchTool() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (name: string) =>
      apiFetch<void>(`/v1/search-tools/${encodeURIComponent(name)}`, {
        method: "DELETE",
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [SEARCH_TOOLS] }),
  })
}

// Probe a (typically unsaved) service URL for reachability. Read-only, so it
// invalidates nothing.
export function useTestService() {
  return useMutation({
    mutationFn: ({ service, url }: { service: string; url: string }) =>
      apiFetch<TestServiceResponse>(
        `/v1/tool-settings/${encodeURIComponent(service)}/test`,
        {
          method: "POST",
          body: JSON.stringify({ url }),
        },
      ),
  })
}

// The pricing endpoint caps `limit` at 1000 server-side, so page through it
// rather than truncating: a gateway with a long price history could otherwise
// have older rows silently vanish from the models table.
const PRICING_PAGE_SIZE = 1000

// Cap the walk so a backend or proxy that ignores `skip` (returning a full page
// every time) can't spin this into an unbounded request loop. 100 pages is 100k
// rows, far beyond any realistic price history.
const PRICING_MAX_PAGES = 100

async function fetchAllPricing(): Promise<PricingResponse[]> {
  const all: PricingResponse[] = []
  for (let page = 0; page < PRICING_MAX_PAGES; page += 1) {
    const rows = await apiFetch<PricingResponse[]>(
      `/v1/pricing?skip=${page * PRICING_PAGE_SIZE}&limit=${PRICING_PAGE_SIZE}`,
    )
    all.push(...rows)
    if (rows.length < PRICING_PAGE_SIZE) {
      break
    }
  }
  return all
}

export function usePricing() {
  return useQuery({
    queryKey: [PRICING],
    queryFn: fetchAllPricing,
  })
}

export function useSetPricing() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: SetPricingRequest) =>
      apiFetch<PricingResponse>("/v1/pricing", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [PRICING] })
      void queryClient.invalidateQueries({ queryKey: [MODELS] })
    },
  })
}

export function useDeletePricing() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (modelKey: string) =>
      apiFetch<void>(`/v1/pricing/${encodeURIComponent(modelKey)}`, {
        method: "DELETE",
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [PRICING] })
      void queryClient.invalidateQueries({ queryKey: [MODELS] })
    },
  })
}

// Long deadline: this fetches the upstream snapshot and diffs it against every
// priced model, so it scales with the pricing table rather than with one hop.
export function usePreviewPricingRefresh() {
  return useMutation({
    mutationFn: () =>
      apiFetch<PricingRefreshPreview>("/v1/pricing/refresh", {
        method: "POST",
        signal: longRequestSignal(),
      }),
  })
}

export function useConfirmPricingRefresh() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () =>
      apiFetch("/v1/pricing/refresh/confirm", { method: "POST" }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [PRICING] })
      void queryClient.invalidateQueries({ queryKey: [MODELS] })
      void queryClient.invalidateQueries({ queryKey: [PROVIDERS] })
    },
  })
}

export function useRejectPricingRefresh() {
  return useMutation({
    mutationFn: () =>
      apiFetch<void>("/v1/pricing/refresh/reject", { method: "POST" }),
  })
}

// The keys endpoint caps `limit` at 1000 server-side; page through it (capped like
// pricing) so a gateway with many keys can't have rows silently vanish from the
// table, and a backend that ignores `skip` can't spin an unbounded loop.
const KEYS_PAGE_SIZE = 1000
const KEYS_MAX_PAGES = 100

async function fetchAllKeys(workspaceId?: string): Promise<ApiKey[]> {
  const all: ApiKey[] = []
  const scope = workspaceId ? `&workspace_id=${workspaceId}` : ""
  for (let page = 0; page < KEYS_MAX_PAGES; page += 1) {
    const rows = await apiFetch<ApiKey[]>(
      `/v1/keys?skip=${page * KEYS_PAGE_SIZE}&limit=${KEYS_PAGE_SIZE}${scope}`,
    )
    all.push(...rows)
    if (rows.length < KEYS_PAGE_SIZE) {
      break
    }
  }
  return all
}

// The workspace is part of the key, not just the request: switching workspaces
// has to refetch rather than serve the previous one's keys from cache. Same for
// the two below. An unset id keeps the deployment-wide view, which is what the
// organization context and a deployment with no workspace selected still want.
export function useKeys(workspaceId?: string) {
  return useQuery({
    queryKey: [KEYS, workspaceId ?? null],
    queryFn: () => fetchAllKeys(workspaceId),
    staleTime: 60_000,
  })
}

// Create returns the plaintext key exactly once (in `key`); the caller reveals it
// and must never write the response into the query cache.
export function useCreateKey() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateKeyRequest) =>
      apiFetch<CreateKeyResponse>("/v1/keys", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: [KEYS] }),
  })
}

export function useUpdateKey() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: UpdateKeyRequest }) =>
      apiFetch<ApiKey>(`/v1/keys/${encodeURIComponent(id)}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: [KEYS] }),
  })
}

// Regenerate: a new secret for the same key row. The old secret stops working
// immediately. Returns the new plaintext once, like create.
export function useRotateKey() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<CreateKeyResponse>(`/v1/keys/${encodeURIComponent(id)}/rotate`, {
        method: "POST",
      }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: [KEYS] }),
  })
}

export function useDeleteKey() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/v1/keys/${encodeURIComponent(id)}`, {
        method: "DELETE",
      }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: [KEYS] }),
  })
}

// The budgets endpoint caps `limit` at 1000 server-side; page through it (capped
// like keys/pricing) so a gateway with many budgets can't have rows silently
// vanish, and a backend that ignores `skip` can't spin an unbounded loop.
const BUDGETS_PAGE_SIZE = 1000
const BUDGETS_MAX_PAGES = 100

async function fetchAllBudgets(): Promise<Budget[]> {
  const all: Budget[] = []
  for (let page = 0; page < BUDGETS_MAX_PAGES; page += 1) {
    const rows = await apiFetch<Budget[]>(
      `/v1/budgets?skip=${page * BUDGETS_PAGE_SIZE}&limit=${BUDGETS_PAGE_SIZE}`,
    )
    all.push(...rows)
    if (rows.length < BUDGETS_PAGE_SIZE) {
      break
    }
  }
  return all
}

export function useBudgets() {
  return useQuery({
    queryKey: [BUDGETS],
    queryFn: fetchAllBudgets,
    staleTime: 60_000,
  })
}

// Per-user reset history for one budget. Enabled only once a budget id is set
// (the drill-down is opened), so the query does not fire for the whole list.
export function useBudgetResetLogs(budgetId: string | null) {
  return useQuery({
    queryKey: [BUDGETS, budgetId, "reset-logs"],
    queryFn: () =>
      apiFetch<BudgetResetLog[]>(
        `/v1/budgets/${encodeURIComponent(budgetId as string)}/reset-logs`,
      ),
    enabled: budgetId !== null,
    staleTime: 60_000,
  })
}

export function useCreateBudget() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateBudgetRequest) =>
      apiFetch<Budget>("/v1/budgets", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [BUDGETS] }),
  })
}

export function useUpdateBudget() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: UpdateBudgetRequest }) =>
      apiFetch<Budget>(`/v1/budgets/${encodeURIComponent(id)}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [BUDGETS] }),
  })
}

export function useDeleteBudget() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/v1/budgets/${encodeURIComponent(id)}`, {
        method: "DELETE",
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [BUDGETS] }),
  })
}

// The tenancy-scoped ceilings, which are a different mechanism from the budgets
// above rather than a view over them: each row carries its own counters, so one
// row is a pooled cap over whatever its scope names. See `client/index.ts`.
//
// The list route returns a bare array (not the `Paged` envelope the tenancy
// routes use) and caps `limit` at 1000 server-side, so it pages like budgets and
// keys do, with the same guard against a backend that ignores `skip`.
const SCOPED_BUDGETS_PAGE_SIZE = 1000
const SCOPED_BUDGETS_MAX_PAGES = 100

async function fetchAllScopedBudgets(): Promise<ScopedBudget[]> {
  const all: ScopedBudget[] = []
  for (let page = 0; page < SCOPED_BUDGETS_MAX_PAGES; page += 1) {
    const rows = await apiFetch<ScopedBudget[]>(
      `/v1/scoped-budgets?skip=${page * SCOPED_BUDGETS_PAGE_SIZE}&limit=${SCOPED_BUDGETS_PAGE_SIZE}`,
    )
    all.push(...rows)
    if (rows.length < SCOPED_BUDGETS_PAGE_SIZE) {
      break
    }
  }
  return all
}

export function useScopedBudgets() {
  return useQuery({
    queryKey: [SCOPED_BUDGETS],
    queryFn: fetchAllScopedBudgets,
    staleTime: 60_000,
  })
}

export function useCreateScopedBudget() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateScopedBudgetRequest) =>
      apiFetch<ScopedBudget>("/v1/scoped-budgets", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [SCOPED_BUDGETS] }),
  })
}

export function useUpdateScopedBudget() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      id,
      body,
    }: {
      id: string
      body: UpdateScopedBudgetRequest
    }) =>
      apiFetch<ScopedBudget>(`/v1/scoped-budgets/${encodeURIComponent(id)}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [SCOPED_BUDGETS] }),
  })
}

export function useDeleteScopedBudget() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/v1/scoped-budgets/${encodeURIComponent(id)}`, {
        method: "DELETE",
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [SCOPED_BUDGETS] }),
  })
}

// The users endpoint caps `limit` at 1000 server-side; page through it (capped
// like keys/budgets) so a gateway with many users can't have rows silently
// vanish, and a backend that ignores `skip` can't spin an unbounded loop.
const USERS_PAGE_SIZE = 1000
const USERS_MAX_PAGES = 100

async function fetchAllUsers(): Promise<User[]> {
  const all: User[] = []
  for (let page = 0; page < USERS_MAX_PAGES; page += 1) {
    const rows = await apiFetch<User[]>(
      `/v1/users?skip=${page * USERS_PAGE_SIZE}&limit=${USERS_PAGE_SIZE}`,
    )
    all.push(...rows)
    if (rows.length < USERS_PAGE_SIZE) {
      break
    }
  }
  return all
}

export function useUsers() {
  return useQuery({
    queryKey: [USERS],
    queryFn: fetchAllUsers,
    staleTime: 60_000,
  })
}

// Assigning a budget to a user changes that budget's usage rollup, so a user
// write invalidates the budgets list too.
function invalidateUserViews(
  queryClient: ReturnType<typeof useQueryClient>,
): void {
  void queryClient.invalidateQueries({ queryKey: [USERS] })
  void queryClient.invalidateQueries({ queryKey: [BUDGETS] })
}

export function useCreateUser() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateUserRequest) =>
      apiFetch<User>("/v1/users", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => invalidateUserViews(queryClient),
  })
}

export function useUpdateUser() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: UpdateUserRequest }) =>
      apiFetch<User>(`/v1/users/${encodeURIComponent(id)}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: () => invalidateUserViews(queryClient),
  })
}

export function useDeleteUser() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/v1/users/${encodeURIComponent(id)}`, {
        method: "DELETE",
      }),
    onSuccess: () => {
      invalidateUserViews(queryClient)
      // Deleting a user deactivates its keys server-side.
      void queryClient.invalidateQueries({ queryKey: [KEYS] })
    },
  })
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
  const params = new URLSearchParams()
  // A multi-value filter goes on the wire as a repeated param (the analytics
  // endpoints match any of them); an empty array is no filter at all, not a
  // filter matching nothing.
  const appendAll = (key: string, value: string | string[] | undefined) => {
    for (const one of typeof value === "string" ? [value] : (value ?? [])) {
      if (one) params.append(key, one)
    }
  }
  if (filters.workspace_id) params.set("workspace_id", filters.workspace_id)
  if (filters.start_date) params.set("start_date", filters.start_date)
  if (filters.end_date) params.set("end_date", filters.end_date)
  if (filters.status) params.set("status", filters.status)
  appendAll("model", filters.model)
  if (filters.endpoint) params.set("endpoint", filters.endpoint)
  if (filters.provider) params.set("provider", filters.provider)
  appendAll("user_id", filters.user_id)
  appendAll("api_key_id", filters.api_key_id)
  if (filters.source) params.set("source", filters.source)
  if (filters.source_label) params.set("source_label", filters.source_label)
  if (filters.tool) params.set("tool", filters.tool)
  if (filters.priced !== undefined) params.set("priced", String(filters.priced))
  if (filters.counts_toward_budget !== undefined) {
    params.set("counts_toward_budget", String(filters.counts_toward_budget))
  }
  return params
}

// One page of usage-log rows for the Activity viewer, newest first.
// `placeholderData: keepPreviousData` keeps the current page on screen while the
// next loads, so paging does not flash empty.
export function useUsageLogs(
  filters: UsageFilters,
  page: number,
  pageSize: number,
) {
  return useQuery({
    queryKey: [USAGE, "list", filters, page, pageSize],
    queryFn: () => {
      const params = usageParams(filters)
      params.set("skip", String(page * pageSize))
      params.set("limit", String(pageSize))
      return apiFetch<UsageEntry[]>(`/v1/usage?${params.toString()}`)
    },
    placeholderData: keepPreviousData,
    // The log is a snapshot an operator reads, not a feed. On a busy gateway rows
    // arrive faster than anyone can inspect them, so a page that refetched on its
    // own reshuffled the table out from under whoever was reading it. It refetches
    // only when asked: a mount, the refresh button, or a change of filters, window,
    // or page (all of which are in the key). Nothing here opts back into the
    // provider's refetch-on-focus default, which is already off (`provider.tsx`).
    // `useLiveUsageCount` is how the page still says that newer rows exist.
    staleTime: 10_000,
  })
}

// Total rows matching the same filters, for the paginator's "N of M". A separate
// request so /v1/usage stays a bare array; run alongside the list.
//
// Deliberately as frozen as the log it counts (see `useUsageLogs`): the total
// describes the page on screen, so a total that moved on its own would disagree
// with the rows the operator can actually page through.
export function useUsageCount(filters: UsageFilters, enabled = true) {
  return useQuery({
    queryKey: [USAGE, "count", filters],
    queryFn: () =>
      apiFetch<UsageCount>(
        `/v1/usage/count?${usageParams(filters).toString()}`,
      ),
    enabled,
    placeholderData: keepPreviousData,
    staleTime: 10_000,
  })
}

// How often the live row count re-reads. Slow, because nothing on screen moves
// when it changes: it only sizes the "N new" badge, which an operator glances at
// rather than watches. TanStack does not poll a backgrounded tab, so an idle
// dashboard costs nothing.
const NEW_ROW_POLL_MS = 15_000

// The same count as `useUsageCount`, polled, so a frozen page can say how far
// behind it has fallen without moving a single row.
//
// A separate cache entry rather than a `refetchInterval` on `useUsageCount`: the
// two readings serve opposite purposes (one is pinned to the rendered page, the
// other is deliberately ahead of it), and two observers of one query key cannot
// disagree about how fresh their data is. The duplicate `COUNT(*)` at mount is
// one indexed count, which is what makes polling it affordable in the first place.
export function useLiveUsageCount(filters: UsageFilters, enabled = true) {
  return useQuery({
    queryKey: [USAGE, "count", "live", filters],
    queryFn: () =>
      apiFetch<UsageCount>(
        `/v1/usage/count?${usageParams(filters).toString()}`,
      ),
    enabled,
    refetchInterval: NEW_ROW_POLL_MS,
    staleTime: 0,
    // A failed count is not worth surfacing: it sits beside a refresh button that
    // fetches the real thing, and the next poll retries anyway.
    retry: false,
  })
}

// How often the in-flight list re-reads. Tight, because it is the only view of a
// request that has not settled yet and the reason to watch it is that something is
// taking a while. TanStack does not poll a backgrounded tab
// (`refetchIntervalInBackground` defaults to false), so an idle dashboard left open
// costs nothing.
const IN_FLIGHT_POLL_MS = 2_000

// Requests the gateway is serving right now, rendered as a live count beside the
// activity log's refresh control rather than as rows in it: the log is a frozen
// snapshot, and rows that reordered themselves every two seconds were the reason
// a busy gateway's activity page could not be read at all. The read takes no
// filters: a request in progress has no outcome, cost, or token count for the
// log's filters to match on, so it is reported gateway-wide.
//
// Never cached across mounts (`staleTime: 0`) and never kept as placeholder data:
// a stale in-flight list is worse than none, since it claims work is running that
// finished a minute ago.
export function useInFlightRequests() {
  return useQuery({
    queryKey: [USAGE, "in-flight"],
    queryFn: () => apiFetch<InFlightResponse>("/v1/usage/in-flight"),
    refetchInterval: IN_FLIGHT_POLL_MS,
    staleTime: 0,
    // Retrying a 404 cannot help: a gateway that does not serve this endpoint
    // never will. Fail fast on it and add no rows rather than re-asking.
    retry: (failureCount, error) =>
      !(error instanceof ApiError && error.status === 404) && failureCount < 3,
  })
}

// How often the rolling failure count re-reads. A dropped-traffic signal is only
// useful if it moves while the operator watches it.
const FAILURE_COUNT_POLL_MS = 60_000

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
      }
      return apiFetch<UsageCount>(
        `/v1/usage/count?${usageParams(filters).toString()}`,
      )
    },
    enabled,
    refetchInterval: FAILURE_COUNT_POLL_MS,
    refetchOnWindowFocus: true,
    staleTime: 0,
    // A failed count is not worth surfacing: it sits beside its own alarm, and
    // the next poll retries anyway.
    retry: false,
  })
}

// The rows of one or more request groups: every attempt a routed request made,
// which is what turns "attempt 1 of 2, failed" into "and here is what served it".
// A plan is capped at a handful of candidates and the activity table pages at a
// hundred rows, so this leaves an order of magnitude of headroom over the largest
// batch either caller can ask for. It is deliberately not a tight bound: nothing
// downstream detects truncation, so the limit has to be one no real page reaches.
const REQUEST_GROUP_PAGE_LIMIT = 1000

// Fetched as a batch (the endpoint takes a repeatable `request_group_id`) so a
// page of the activity log costs one lookup rather than one per row. The key
// sorts its ids so two callers asking for the same set share a cache entry.
export function useRequestGroups(groupIds: readonly string[]) {
  const ids = [...new Set(groupIds)].sort()
  return useQuery({
    queryKey: [USAGE, "groups", ids],
    queryFn: () => {
      const params = new URLSearchParams()
      for (const id of ids) params.append("request_group_id", id)
      params.set("limit", String(REQUEST_GROUP_PAGE_LIMIT))
      return apiFetch<UsageEntry[]>(`/v1/usage?${params.toString()}`)
    },
    enabled: ids.length > 0,
    placeholderData: keepPreviousData,
    // A group is immutable once its request finished, so the only reason to
    // refetch is a group that was still in flight when it was first read.
    staleTime: 30_000,
  })
}

// Delete imported usage rows by selection (ids or by_filter). Only rows the
// server treats as imported (counts_toward_budget = false) are removed; every
// usage view is invalidated so the list, count, and analytics refresh.
//
// Given the long deadline, not apiFetch's default: a by_filter delete is one
// unbounded DELETE server-side, so its duration tracks the number of matched
// rows. Timing out here would report failure for a delete that committed anyway.
export function useDeleteUsage() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: UsageMutationSelection) =>
      apiFetch<UsageDeleteResult>("/v1/usage", {
        method: "DELETE",
        body: JSON.stringify(body),
        signal: longRequestSignal(),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [USAGE] })
    },
  })
}

// Set the cost of imported usage rows from manual per-1M rates. Long deadline
// for the same reason as the delete: the server reprices every matched row.
export function useSetUsagePrice() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: UsageSetPriceRequest) =>
      apiFetch<UsageSetPriceResult>("/v1/usage/set-price", {
        method: "POST",
        body: JSON.stringify(body),
        signal: longRequestSignal(),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [USAGE] })
    },
  })
}

// ---------- usage analytics summary ----------

// Pass this as `dimensions` to read only `totals` / `series`. Named rather than a
// bare `[]` at each call site so it is clear the omission is deliberate.
export const NO_BREAKDOWNS: SummaryDimension[] = []

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
      const params = usageParams(filters)
      params.set("bucket", bucket)
      // A repeated query param has no empty-list form, so an empty selection goes
      // on the wire as the server's `none` sentinel.
      if (dimensions) {
        for (const dimension of dimensions.length > 0 ? dimensions : ["none"]) {
          params.append("dimensions", dimension)
        }
      }
      return apiFetch<UsageSummary>(`/v1/usage/summary?${params.toString()}`)
    },
    enabled,
    placeholderData: keepPreviousData,
    staleTime: 30_000,
  })
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
      const params = usageParams(filters)
      params.set("bucket", bucket)
      params.set("group_by", groupBy as string)
      return apiFetch<UsageGroupedSeries>(
        `/v1/usage/series?${params.toString()}`,
      )
    },
    enabled: enabled && groupBy !== null,
    placeholderData: keepPreviousData,
    staleTime: 30_000,
    // A 404 is version skew (a gateway older than this dashboard, e.g. not yet
    // restarted onto the build that ships it); retrying cannot fix that, and
    // the page falls back to the ungrouped view with a notice instead.
    retry: (failureCount, error) =>
      !(error instanceof ApiError && error.status === 404) && failureCount < 3,
  })
}

// ---------- tenancy: organizations, workspaces, memberships ----------

// The tenancy endpoints answer `{ data, count }` and page with skip/limit, all
// capped at 1000 server-side. The pages want one list, so the walk is bounded
// the same way fetchAllPricing is: a backend or proxy that ignores `skip` must
// not turn "fetch everything" into an unbounded loop.
const TENANCY_PAGE_SIZE = 1000
const TENANCY_MAX_PAGES = 100

interface Paged<T> {
  data: T[]
  count: number
}

async function fetchAllPaged<T>(path: string): Promise<T[]> {
  const all: T[] = []
  for (let page = 0; page < TENANCY_MAX_PAGES; page += 1) {
    const body = await apiFetch<Paged<T>>(
      `${path}?skip=${page * TENANCY_PAGE_SIZE}&limit=${TENANCY_PAGE_SIZE}`,
    )
    all.push(...body.data)
    if (body.data.length < TENANCY_PAGE_SIZE) break
  }
  return all
}

// The organization the caller's identity is pointed at, and their standing in
// it. Every tenancy page reads it first: it names the tenant on screen and
// decides whether the management controls are offered at all. Read often and
// changed rarely, so it is cached for a minute like the other management lists.
export function useOrganizationContext() {
  return useQuery({
    queryKey: [ORGANIZATIONS, "context"],
    queryFn: () => apiFetch<OrganizationContext>("/v1/organizations/me"),
    staleTime: 60_000,
  })
}

// The organizations the caller is an active member of, which is what the
// organization half of the scope switcher renders. Its own read rather than a
// field on the context: the context is one organization, and a switcher needs
// the list. Cached for the same minute, because they move at the same rate.
export function useOrganizationMemberships() {
  return useQuery({
    queryKey: [ORGANIZATIONS, "memberships"],
    queryFn: () =>
      fetchAllPaged<CallerOrganizationMembership>(
        "/v1/organizations/me/memberships",
      ),
    staleTime: 60_000,
    // Same guard as `useUsageGroupedSeries` and `useInFlightRequests`, and for
    // both of their reasons: a gateway older than this bundle does not serve
    // this route (the process may not have restarted onto the build that ships
    // it), and a hybrid gateway answers 404 for every `/v1/organizations` path
    // by design. Neither is something a retry fixes; the switcher falls back to
    // stating the one organization the context names.
    retry: (failureCount, error) =>
      !(error instanceof ApiError && error.status === 404) && failureCount < 3,
  })
}

// Creating one makes the caller its owner and provisions a default workspace,
// and deliberately does not switch into it: the switcher chains this with
// `useSwitchOrganization` so the two steps stay separately reportable.
export function useCreateOrganization() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateOrganizationRequest) =>
      apiFetch<Organization>("/v1/organizations", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      // The membership list has a new row; nothing else has changed, because
      // the caller is still acting in the organization they were in.
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATIONS] })
    },
  })
}

// Switching moves `users.active_organization_id`, which is what every scoped
// read on the server resolves through, so *everything* cached here is about
// the organization just left. Hence `invalidateQueries()` with no key rather
// than a list of them: enumerating the affected keys would mean keeping that
// list in step with every future query, and the one it missed would render
// another organization's rows under this one's name.
export function useSwitchOrganization() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (organizationId: string) => {
      // Typed against the generated request rather than written inline, so a
      // field the gateway renames fails here instead of on the wire. The
      // parameter stays a bare id: both call sites have one, not a body.
      const body: SwitchOrganizationRequest = {
        organization_id: organizationId,
      }
      return apiFetch<OrganizationContext>("/v1/organizations/me/switch", {
        method: "POST",
        body: JSON.stringify(body),
      })
    },
    onSuccess: () => {
      void queryClient.invalidateQueries()
    },
  })
}

// `enabled` because the roster is now read from outside the Organization pages
// too, to name the owner of an API key. A deployment that does not host the
// `organizations` surface has no such route to call, so the caller gates on it
// rather than letting the page 404 on a request it only wanted for a label.
export function useOrganizationMembers(enabled = true) {
  return useQuery({
    queryKey: [ORGANIZATION_MEMBERS],
    queryFn: () =>
      fetchAllPaged<OrganizationMember>("/v1/organizations/me/members"),
    staleTime: 60_000,
    enabled,
  })
}

export function useUpdateOrganization() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: UpdateOrganizationRequest) =>
      apiFetch<OrganizationContext>("/v1/organizations/me", {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATIONS] })
    },
  })
}
// One of two write paths that put a second row on the roster: this one lands
// the membership `active` immediately, with nothing emailed.
// `useInviteOrganizationMember` below is the other, which lands `invited` and
// emails an accept link.
export function useAddOrganizationMember() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateOrganizationMemberRequest) =>
      apiFetch<CreateOrganizationMemberResult>("/v1/organizations/me/members", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATION_MEMBERS] })
      // A request may place the new member into workspaces in the same
      // transaction, so their rosters move with it.
      void queryClient.invalidateQueries({ queryKey: [WORKSPACES] })
      // The switcher reads its list from `workspace_memberships` on the
      // organization context, not from this key, so a roster change that moves
      // the caller in or out of a workspace has to refresh it too.
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATIONS] })
    },
  })
}

export function useUpdateOrganizationMember() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      id,
      body,
    }: {
      id: string
      body: UpdateOrganizationMemberRequest
    }) =>
      apiFetch<OrganizationMember>(
        `/v1/organizations/me/members/${encodeURIComponent(id)}`,
        { method: "PATCH", body: JSON.stringify(body) },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATION_MEMBERS] })
      // The caller may have changed their own role, which is what the page
      // gates its controls on.
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATIONS] })
    },
  })
}

export function useRemoveOrganizationMember() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/v1/organizations/me/members/${encodeURIComponent(id)}`, {
        method: "DELETE",
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATION_MEMBERS] })
      // A suspended member keeps their workspace rows, so every roster that
      // resolves a name through the organization list is now stale.
      void queryClient.invalidateQueries({ queryKey: [WORKSPACES] })
      // The switcher reads its list from `workspace_memberships` on the
      // organization context, not from this key, so a roster change that moves
      // the caller in or out of a workspace has to refresh it too.
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATIONS] })
    },
  })
}

// The other write path onto the roster: lands `invited` rather than `active`,
// and the response always carries `accept_link` (whether or not `mail_sent`
// is true), so the caller can offer "share this link yourself" when it isn't.
export function useInviteOrganizationMember() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: InviteOrganizationMemberRequest) =>
      apiFetch<InviteOrganizationMemberResult>(
        "/v1/organizations/me/member-invitations",
        { method: "POST", body: JSON.stringify(body) },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATION_MEMBERS] })
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATIONS] })
    },
  })
}

export function useRevokeOrganizationMemberInvitation() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (invitationId: string) =>
      apiFetch<void>(
        `/v1/organizations/me/member-invitations/${encodeURIComponent(invitationId)}`,
        { method: "DELETE" },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATION_MEMBERS] })
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATIONS] })
    },
  })
}

// The accept-invitation page's two calls. Both hit routes the server never
// gates on a session or the master key: the token in the emailed link is the
// caller's whole credential, and the gateway answers 404/400 for a bad one,
// never 401, so apiFetch's session-bounce on 401/403 never triggers here.
export function useValidateInvitation(token: string) {
  return useQuery({
    queryKey: ["invitation-preview", token],
    // POST with the token in the body, not a GET with it in the URL: the
    // token is a bearer credential, and a URL is what an access log or an
    // intermediate proxy routinely retains.
    queryFn: () =>
      apiFetch<InvitationPreview>("/v1/invitations/validate", {
        method: "POST",
        body: JSON.stringify({ token }),
      }),
    // An empty token (a malformed link) is never worth a round trip: the
    // server would only answer "not found" for what the client can already
    // see is missing.
    enabled: token.length > 0,
    retry: false,
  })
}

export function useAcceptInvitation() {
  return useMutation({
    mutationFn: (token: string) =>
      apiFetch<AcceptInvitationResult>("/v1/invitations/accept", {
        method: "POST",
        body: JSON.stringify({ token }),
      }),
  })
}

// The public auth flows (otari#650). Same shape as the two invitation calls
// above and for the same reason: nothing here is gated on a session or the
// master key, because a caller completing a signup or opening an emailed link
// holds neither. The gateway answers 400 for a bad token, 429 when the shared
// sign-in limiter fires, and 503 when this deployment cannot send mail, so
// apiFetch's session-bounce on 401/403 never triggers on any of them.
//
// None of them invalidates anything. They write to an identity this
// unauthenticated caller cannot read back, and the cache they would touch
// belongs to a session that does not exist yet.

// Claims a roster identity by setting its password, then mails a verification
// link. The response is the same sentence whether the address was unknown,
// already claimed, or genuinely just claimed, so nothing here may branch on it.
export function useSignup() {
  return useMutation({
    mutationFn: (body: SignupRequest) =>
      apiFetch<SignupResponse>("/v1/auth/signup", {
        method: "POST",
        body: JSON.stringify(body),
      }),
  })
}

// A query rather than a mutation, the same shape `useValidateInvitation` takes
// and for a reason that outranks the fact that this one does write: the page
// verifies on arrival rather than behind a button, so *whatever* fires it has
// to fire exactly once per token, and a query keyed on the token is the only
// one of the two that the cache makes idempotent for free. A mutation fired
// from an effect is not: `main.tsx` runs under StrictMode, whose
// mount/unmount/mount would spend a single-use token twice and land the second
// call's `400` over the first call's success.
//
// The knobs are what keep it a one-shot, and they are spelled out here rather
// than leaning on the provider's defaults, because "fires once" is this hook's
// contract and not a coincidence of how the app is configured. Never stale and
// never collected, so a remount reads the answer back instead of asking again.
// No retry, because a spent token's `400` is the final answer and not a blip.
// And the three automatic refetches are off by name: staleness alone does not
// hold them back once a query has failed, since a failure leaves no data for
// `staleTime` to keep fresh, so without these a reconnect or a remount would
// re-POST a token that is already gone.
//
// POST with the token in the body rather than a GET with it in the URL, the
// same reasoning `useValidateInvitation` gives: the token is a bearer
// credential and a URL is what an access log or an intermediate proxy
// routinely retains.
export function useVerifyEmail(token: string) {
  return useQuery({
    queryKey: ["verify-email", token],
    queryFn: () =>
      apiFetch<VerifyEmailResponse>("/v1/auth/verify-email", {
        method: "POST",
        body: JSON.stringify({ token }),
      }),
    // A malformed link is never worth a round trip; the page says so itself.
    enabled: token.length > 0,
    retry: false,
    retryOnMount: false,
    refetchOnMount: false,
    refetchOnReconnect: false,
    refetchOnWindowFocus: false,
    staleTime: Number.POSITIVE_INFINITY,
    gcTime: Number.POSITIVE_INFINITY,
  })
}

export function useResendVerification() {
  return useMutation({
    mutationFn: (email: string) =>
      apiFetch<ResendVerificationResponse>("/v1/auth/resend-verification", {
        method: "POST",
        body: JSON.stringify({ email }),
      }),
  })
}

export function useRequestPasswordReset() {
  return useMutation({
    mutationFn: (email: string) =>
      apiFetch<RequestPasswordResetResponse>("/v1/auth/password/reset", {
        method: "POST",
        body: JSON.stringify({ email }),
      }),
  })
}

// 204, so there is nothing to read back: the caller learns it worked by the
// call not raising, and signs in with the new password from the sign-in screen.
export function useResetPassword() {
  return useMutation({
    mutationFn: (body: ResetPasswordRequest) =>
      apiFetch<void>("/v1/auth/password/reset/confirm", {
        method: "POST",
        body: JSON.stringify(body),
      }),
  })
}

export function useWorkspaces() {
  return useQuery({
    queryKey: [WORKSPACES],
    queryFn: () => fetchAllPaged<Workspace>("/v1/workspaces"),
    staleTime: 60_000,
  })
}

// One workspace's roster. Nested under the workspaces key so deleting a
// workspace drops its roster with it.
export function useWorkspaceMembers(workspaceId: string | null) {
  return useQuery({
    queryKey: [WORKSPACES, workspaceId, "members"],
    queryFn: () =>
      fetchAllPaged<WorkspaceMember>(
        `/v1/workspaces/${encodeURIComponent(workspaceId as string)}/members`,
      ),
    enabled: workspaceId !== null,
    staleTime: 60_000,
  })
}

export function useCreateWorkspace() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateWorkspaceRequest) =>
      apiFetch<Workspace>("/v1/workspaces", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [WORKSPACES] })
      // The switcher reads its list from `workspace_memberships` on the
      // organization context, not from this key, so a workspace created
      // here would not be offered and a deleted one would stay selected.
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATIONS] })
    },
  })
}

export function useUpdateWorkspace() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: UpdateWorkspaceRequest }) =>
      apiFetch<Workspace>(`/v1/workspaces/${encodeURIComponent(id)}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [WORKSPACES] })
      // The switcher reads its list from `workspace_memberships` on the
      // organization context, not from this key, so a workspace created
      // here would not be offered and a deleted one would stay selected.
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATIONS] })
    },
  })
}

export function useDeleteWorkspace() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/v1/workspaces/${encodeURIComponent(id)}`, {
        method: "DELETE",
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [WORKSPACES] })
      // The switcher reads its list from `workspace_memberships` on the
      // organization context, not from this key, so a workspace created
      // here would not be offered and a deleted one would stay selected.
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATIONS] })
    },
  })
}

export function useAddWorkspaceMember() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      userId,
      role,
    }: {
      workspaceId: string
      userId: string
      role: WorkspaceMemberRole
    }) =>
      // The role travels as a query parameter, not a body: that is the wire
      // contract these endpoints were rehomed with.
      apiFetch<WorkspaceMember>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/members/${encodeURIComponent(userId)}?role=${encodeURIComponent(role)}`,
        { method: "POST" },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [WORKSPACES] })
      // The switcher reads its list from `workspace_memberships` on the
      // organization context, not from this key, so a roster change that moves
      // the caller in or out of a workspace has to refresh it too.
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATIONS] })
    },
  })
}

export function useUpdateWorkspaceMemberRole() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      userId,
      role,
    }: {
      workspaceId: string
      userId: string
      role: WorkspaceMemberRole
    }) =>
      apiFetch<WorkspaceMember>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/members/${encodeURIComponent(userId)}?role=${encodeURIComponent(role)}`,
        { method: "PATCH" },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [WORKSPACES] })
      // The switcher reads its list from `workspace_memberships` on the
      // organization context, not from this key, so a roster change that moves
      // the caller in or out of a workspace has to refresh it too.
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATIONS] })
    },
  })
}

// A workspace's budget-default templates. Nested under the workspaces key so
// deleting a workspace drops them with it, same as `useWorkspaceMembers`.
export function useWorkspaceBudgetDefaults(workspaceId: string | null) {
  return useQuery({
    queryKey: [WORKSPACES, workspaceId, "budget-defaults"],
    queryFn: () =>
      fetchAllPaged<WorkspaceBudgetDefault>(
        `/v1/workspaces/${encodeURIComponent(workspaceId as string)}/member-budget-policies`,
      ),
    enabled: workspaceId !== null,
    staleTime: 60_000,
  })
}

/**
 * Every workspace's roster, as one list, each row paired with its workspace.
 *
 * Same fan-out as `useAllWorkspaceBudgetDefaults` and for the same reason: a
 * roster is only served per workspace, and a standalone deployment has few. It
 * is what lets the organization roster answer "which workspaces is this person
 * in", which is otherwise only answerable one workspace at a time.
 */
export function useAllWorkspaceMembers(workspaceIds: string[]) {
  return useQueries({
    queries: workspaceIds.map((workspaceId) => ({
      queryKey: [WORKSPACES, workspaceId, "members"],
      queryFn: () =>
        fetchAllPaged<WorkspaceMember>(
          `/v1/workspaces/${encodeURIComponent(workspaceId)}/members`,
        ),
      staleTime: 60_000,
    })),
    combine: (results) => ({
      data: results.flatMap((result, index) =>
        (result.data ?? []).map((row) => ({
          workspaceId: workspaceIds[index],
          member: row,
        })),
      ),
      isLoading: results.some((result) => result.isLoading),
      // The first failure, surfaced rather than swallowed: a rejected read
      // contributes nothing to `data`, so without this the caller cannot tell a
      // workspace with no rows from one whose read failed, and a lost membership
      // or a lost ceiling looks exactly like a deliberate absence.
      error: results.find((result) => result.error)?.error ?? null,
      isSuccess: results.every((result) => result.isSuccess),
    }),
  })
}

/**
 * Every workspace's budget defaults, as one list.
 *
 * A fan-out rather than one call: defaults are only served per workspace
 * (`/v1/workspaces/{id}/member-budget-policies`), and a standalone deployment
 * has few workspaces, so N small cached reads beat adding a route. Each shares
 * the cache entry `useWorkspaceBudgetDefaults` uses, so opening a workspace
 * afterwards costs nothing.
 *
 * This is what lets the budgets list say a budget is a workspace's default:
 * without it the page would know the budget and not the assignment.
 */
export function useAllWorkspaceBudgetDefaults(workspaceIds: string[]) {
  return useQueries({
    queries: workspaceIds.map((workspaceId) => ({
      queryKey: [WORKSPACES, workspaceId, "budget-defaults"],
      queryFn: () =>
        fetchAllPaged<WorkspaceBudgetDefault>(
          `/v1/workspaces/${encodeURIComponent(workspaceId)}/member-budget-policies`,
        ),
      staleTime: 60_000,
    })),
    combine: (results) => ({
      // Paired with its workspace on the way out: a default names a workspace by
      // id, and the caller wants the name.
      data: results.flatMap((result, index) =>
        (result.data ?? []).map((row) => ({
          workspaceId: workspaceIds[index],
          default: row,
        })),
      ),
      isLoading: results.some((result) => result.isLoading),
      // The first failure, surfaced rather than swallowed: a rejected read
      // contributes nothing to `data`, so without this the caller cannot tell a
      // workspace with no rows from one whose read failed, and a lost membership
      // or a lost ceiling looks exactly like a deliberate absence.
      error: results.find((result) => result.error)?.error ?? null,
      isSuccess: results.every((result) => result.isSuccess),
    }),
  })
}

export function useCreateWorkspaceBudgetDefault() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      body,
    }: {
      workspaceId: string
      body: CreateWorkspaceBudgetDefaultRequest
    }) =>
      apiFetch<WorkspaceBudgetDefault>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/member-budget-policies`,
        { method: "POST", body: JSON.stringify(body) },
      ),
    onSuccess: (_data, { workspaceId }) => {
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "budget-defaults"],
      })
    },
  })
}

export function useUpdateWorkspaceBudgetDefault() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      defaultId,
      body,
    }: {
      workspaceId: string
      defaultId: string
      body: UpdateWorkspaceBudgetDefaultRequest
    }) =>
      apiFetch<WorkspaceBudgetDefault>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/member-budget-policies/${encodeURIComponent(defaultId)}`,
        { method: "PATCH", body: JSON.stringify(body) },
      ),
    onSuccess: (_data, { workspaceId }) => {
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "budget-defaults"],
      })
    },
  })
}

export function useDeleteWorkspaceBudgetDefault() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      defaultId,
    }: {
      workspaceId: string
      defaultId: string
    }) =>
      apiFetch<void>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/member-budget-policies/${encodeURIComponent(defaultId)}`,
        { method: "DELETE" },
      ),
    onSuccess: (_data, { workspaceId }) => {
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "budget-defaults"],
      })
    },
  })
}

// The guardrails the caller's organization mandates over its workspaces. A
// small hand-edited list rather than a growing table, but paged through like
// the rest of the tenancy surface so a backend that ignored `skip` cannot spin
// this either.
export function useOrganizationGuardrails(enabled = true) {
  return useQuery({
    queryKey: [ORGANIZATION_GUARDRAILS],
    queryFn: () =>
      fetchAllPaged<OrganizationGuardrail>("/v1/organizations/me/guardrails"),
    staleTime: 60_000,
    enabled,
  })
}

export function useCreateOrganizationGuardrail() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateOrganizationGuardrailRequest) =>
      apiFetch<OrganizationGuardrail>("/v1/organizations/me/guardrails", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({
        queryKey: [ORGANIZATION_GUARDRAILS],
      })
    },
  })
}

export function useUpdateOrganizationGuardrail() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      guardrailId,
      body,
    }: {
      guardrailId: string
      body: UpdateOrganizationGuardrailRequest
    }) =>
      apiFetch<OrganizationGuardrail>(
        `/v1/organizations/me/guardrails/${encodeURIComponent(guardrailId)}`,
        { method: "PATCH", body: JSON.stringify(body) },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({
        queryKey: [ORGANIZATION_GUARDRAILS],
      })
    },
  })
}

export function useDeleteOrganizationGuardrail() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (guardrailId: string) =>
      apiFetch<{ message: string }>(
        `/v1/organizations/me/guardrails/${encodeURIComponent(guardrailId)}`,
        { method: "DELETE" },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({
        queryKey: [ORGANIZATION_GUARDRAILS],
      })
    },
  })
}

// A workspace's code-execution policy over the deployment-wide sandbox. One
// object or none, so it is a plain read rather than a paged list, and it is
// nested under the workspaces key for the same reason the budget defaults are.
export function useWorkspaceCodeExecutionPolicy(workspaceId: string | null) {
  return useQuery({
    queryKey: [WORKSPACES, workspaceId, "code-execution-policy"],
    queryFn: () =>
      apiFetch<WorkspaceCodeExecutionPolicy>(
        `/v1/workspaces/${encodeURIComponent(workspaceId as string)}/code-execution-policy`,
      ),
    enabled: workspaceId !== null,
    staleTime: 60_000,
  })
}

export function useSetWorkspaceCodeExecutionPolicy() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      body,
    }: {
      workspaceId: string
      body: UpdateWorkspaceCodeExecutionPolicyRequest
    }) =>
      apiFetch<WorkspaceCodeExecutionPolicy>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/code-execution-policy`,
        { method: "PUT", body: JSON.stringify(body) },
      ),
    onSuccess: (_data, { workspaceId }) => {
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "code-execution-policy"],
      })
    },
  })
}

// Drops the row, which returns the workspace to the deployment's own behavior.
// Not the same as saving `enabled: true`: that is a stored decision not to
// narrow, while this is no decision at all.
export function useClearWorkspaceCodeExecutionPolicy() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ workspaceId }: { workspaceId: string }) =>
      apiFetch<WorkspaceCodeExecutionPolicy>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/code-execution-policy`,
        { method: "DELETE" },
      ),
    onSuccess: (_data, { workspaceId }) => {
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "code-execution-policy"],
      })
    },
  })
}

// A workspace's web-search configuration over the deployment-wide backend. One
// object or none, so it is a plain read rather than a paged list, and it is
// nested under the workspaces key for the same reason the code-execution policy
// next door is.
export function useWorkspaceWebSearchConfig(workspaceId: string | null) {
  return useQuery({
    queryKey: [WORKSPACES, workspaceId, "web-search"],
    queryFn: () =>
      apiFetch<WorkspaceWebSearchConfig>(
        `/v1/workspaces/${encodeURIComponent(workspaceId as string)}/web-search`,
      ),
    enabled: workspaceId !== null,
    staleTime: 60_000,
  })
}

export function useSetWorkspaceWebSearchConfig() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      body,
    }: {
      workspaceId: string
      body: UpdateWorkspaceWebSearchConfigRequest
    }) =>
      apiFetch<WorkspaceWebSearchConfig>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/web-search`,
        { method: "PUT", body: JSON.stringify(body) },
      ),
    onSuccess: (_data, { workspaceId }) => {
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "web-search"],
      })
    },
  })
}

// Drops the row, which returns the workspace to the deployment's own behavior.
// Not the same as saving `enabled: true`: that is a stored decision not to
// narrow, while this is no decision at all.
export function useClearWorkspaceWebSearchConfig() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ workspaceId }: { workspaceId: string }) =>
      apiFetch<WorkspaceWebSearchConfig>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/web-search`,
        { method: "DELETE" },
      ),
    onSuccess: (_data, { workspaceId }) => {
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "web-search"],
      })
    },
  })
}

// A workspace's MCP servers. A list rather than the single row the two config
// planes beside it hold, and nested under the workspaces key for the same
// reason they are.
//
// One request, at the endpoint's own documented ceiling, rather than a paged
// walk: the service caps how many servers a workspace may hold well below this,
// so a second page cannot exist. The ceiling is the route's (`le=1000`), not a
// copy of that cap, so this stays right if the cap moves.
const MCP_SERVERS_PAGE_SIZE = 1000

export function useWorkspaceMcpServers(workspaceId: string | null) {
  return useQuery({
    queryKey: [WORKSPACES, workspaceId, "mcp-servers"],
    queryFn: () =>
      apiFetch<WorkspaceMcpServers>(
        `/v1/workspaces/${encodeURIComponent(workspaceId as string)}/mcp-servers?limit=${MCP_SERVERS_PAGE_SIZE}`,
      ),
    enabled: workspaceId !== null,
    staleTime: 60_000,
  })
}

export function useCreateWorkspaceMcpServer() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      body,
    }: {
      workspaceId: string
      body: CreateWorkspaceMcpServerRequest
    }) =>
      apiFetch<WorkspaceMcpServer>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/mcp-servers`,
        { method: "POST", body: JSON.stringify(body) },
      ),
    onSuccess: (_data, { workspaceId }) => {
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "mcp-servers"],
      })
    },
  })
}

// A partial update, which is what keeps the write-only token's three states
// expressible. See `McpServerDialog` for the rule and how the form maps onto it.
export function useUpdateWorkspaceMcpServer() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      serverId,
      body,
    }: {
      workspaceId: string
      serverId: string
      body: UpdateWorkspaceMcpServerRequest
    }) =>
      apiFetch<WorkspaceMcpServer>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/mcp-servers/${encodeURIComponent(serverId)}`,
        { method: "PATCH", body: JSON.stringify(body) },
      ),
    onSuccess: (_data, { workspaceId }) => {
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "mcp-servers"],
      })
    },
  })
}

export function useDeleteWorkspaceMcpServer() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      serverId,
    }: {
      workspaceId: string
      serverId: string
    }) =>
      apiFetch<void>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/mcp-servers/${encodeURIComponent(serverId)}`,
        { method: "DELETE" },
      ),
    onSuccess: (_data, { workspaceId }) => {
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "mcp-servers"],
      })
    },
  })
}

export function useRemoveWorkspaceMember() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      userId,
    }: {
      workspaceId: string
      userId: string
    }) =>
      apiFetch<void>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/members/${encodeURIComponent(userId)}`,
        { method: "DELETE" },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [WORKSPACES] })
      // The switcher reads its list from `workspace_memberships` on the
      // organization context, not from this key, so a roster change that moves
      // the caller in or out of a workspace has to refresh it too.
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATIONS] })
    },
  })
}

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
      fetchAllPaged<OrganizationPricingOverride>(
        "/v1/organizations/me/pricing",
      ),
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
}

export function useCreateOrganizationPricing() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateOrganizationPricingOverride) =>
      apiFetch<OrganizationPricingOverride>("/v1/organizations/me/pricing", {
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
        `/v1/organizations/me/pricing/${encodeURIComponent(id)}`,
        { method: "PUT", body: JSON.stringify(body) },
      ),
    onSuccess: () => invalidateOrganizationPricing(queryClient),
  })
}

export function useDeleteOrganizationPricing() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/v1/organizations/me/pricing/${encodeURIComponent(id)}`, {
        method: "DELETE",
      }),
    onSuccess: () => invalidateOrganizationPricing(queryClient),
  })
}

// ---------------------------------------------------------------------------
// The first-request setup guide
// ---------------------------------------------------------------------------

// How often the guide asks whether the workspace's first request has landed.
// Only while it is on screen and still waiting (the card passes `enabled`), and
// the answer is one or two indexed reads server-side, so this is the interval
// that makes "send the request, watch it arrive" feel live without polling for a
// dashboard nobody is looking at.
const ACTIVATION_POLL_MS = 4_000

/**
 * Where the selected workspace stands on its first successful request.
 *
 * Polls only while the guide is actually being offered, which is also the only
 * state whose answer can still change on its own. A workspace that activated
 * cannot go back, and one whose guide was dismissed (or turned off for the
 * deployment) has nothing to wait for, so both stop the interval rather than
 * asking every few seconds for the life of the page.
 */
export function useWorkspaceActivation(
  workspaceId: string | null,
  enabled = true,
) {
  return useQuery({
    queryKey: [ACTIVATION, workspaceId],
    queryFn: () =>
      apiFetch<WorkspaceActivation>(
        `/v1/workspaces/${encodeURIComponent(workspaceId as string)}/activation`,
      ),
    enabled: enabled && workspaceId !== null,
    refetchInterval: (query) =>
      query.state.data?.experience_eligible ? ACTIVATION_POLL_MS : false,
    // A failed check is reported on the card, which offers "Check now": retrying
    // twice behind the operator's back would only delay that by a poll interval.
    ...NO_RETRY,
  })
}

// Issues the workspace's setup key and returns its plaintext exactly once, like
// `useCreateKey`: the caller shows it and must never write the response into the
// query cache. KEYS is invalidated because the key it rotates is an ordinary row
// on the Keys page, and ACTIVATION because the guide's state now records it.
export function useCreateActivationKey() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (workspaceId: string) =>
      apiFetch<ActivationApiKey>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/activation/key`,
        { method: "POST" },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [KEYS] })
      void queryClient.invalidateQueries({ queryKey: [ACTIVATION] })
    },
  })
}

// Permanent, and idempotent server-side. Only ACTIVATION is invalidated:
// dismissing retires the card and leaves the key it issued alone, so nothing on
// the Keys page changed.
export function useDismissActivation() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (workspaceId: string) =>
      apiFetch<{ message: string }>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/activation/dismiss`,
        { method: "POST" },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ACTIVATION] })
    },
  })
}
