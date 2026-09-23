import {
  hashKey,
  keepPreviousData,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query"
import type {
  AcceptInvitationRequest,
  AcceptInvitationResult,
  CallerOrganizationMembership,
  CreateOrganizationDomainRequest,
  CreateOrganizationRequest,
  CreateOrgProviderKeyRequest,
  InvitationPreview,
  InviteOrganizationMemberRequest,
  InviteOrganizationMemberResult,
  OfferOrgProviderModelRequest,
  Organization,
  OrganizationContext,
  OrganizationDomain,
  OrganizationMember,
  OrganizationMembers,
  OrgProviderAvailableModels,
  OrgProviderKey,
  OrgProviderModel,
  OrgProviderModels,
  OrgProviderModelsRefresh,
  PendingOrganizationInvitation,
  SwitchOrganizationRequest,
  UpdateOrganizationDomainRequest,
  UpdateOrganizationMemberRequest,
  UpdateOrganizationRequest,
  UpdateOrgProviderKeyRequest,
  UpdateOrgProviderModelRequest,
} from "@/client"
import { ApiError, apiFetch, longRequestSignal } from "@/shared/api/client"
import { fetchAllPaged } from "@/shared/api/paging"
import {
  CATALOG,
  MODELS,
  NO_RETRY,
  ORGANIZATION_CONTEXT,
  ORGANIZATION_DOMAINS,
  ORGANIZATION_MEMBERS,
  ORGANIZATION_PROVIDER_AVAILABLE_MODELS,
  ORGANIZATION_PROVIDER_KEYS,
  ORGANIZATION_PROVIDER_MODELS,
  ORGANIZATIONS,
  WORKSPACES,
} from "@/shared/api/queryKeys"

/** The context key as React Query addresses it, for a filter that excludes it. */
const ORGANIZATION_CONTEXT_HASH = hashKey(ORGANIZATION_CONTEXT)

// The organization the caller's identity is pointed at, and their standing in
// it. Every tenancy page reads it first: it names the tenant on screen and
// decides whether the management controls are offered at all. Read often and
// changed rarely, so it is cached for a minute like the other management lists.
//
// `enabled` is for the one page that renders ahead of a session: the public
// catalog has no organization to ask about, and asking would 401 into the
// sign-out handler.
export function useOrganizationContext(enabled = true) {
  return useQuery({
    queryKey: ORGANIZATION_CONTEXT,
    queryFn: () => apiFetch<OrganizationContext>("/organizations/me"),
    staleTime: 60_000,
    enabled,
  })
}

// Whether the deployment can encrypt a provider credential at rest, i.e. whether
// OTARI_SECRET_KEY is set on the server. Both provider-key pages gate their add
// control on it, so the rule lives here rather than twice.
//
// Read off the membership context rather than `/settings`, which reports the
// same fact as `secret_key_configured` but is operator-only: an organization
// owner is not one, so that query 403s for the whole tenant-facing audience and
// a refusal used to read as "the key is missing" (#839).
//
// Fails closed on an error and open while the context is still loading, so the
// control does not flicker to disabled on first paint. An older gateway omits
// the field, and a present-but-missing value reads as configured because those
// gateways never gated on it.
export function useProviderKeyEncryption() {
  const context = useOrganizationContext()
  return context.data
    ? context.data.provider_key_encryption_available !== false
    : !context.isError
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
        "/organizations/me/memberships",
      ),
    staleTime: 60_000,
    // Same guard as `useUsageGroupedSeries` and `useInFlightRequests`, and for
    // both of their reasons: a gateway older than this bundle does not serve
    // this route (the process may not have restarted onto the build that ships
    // it), and a hybrid gateway answers 404 for every `/organizations` path
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
      apiFetch<Organization>("/organizations", {
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
// the organization just left. Hence an invalidation with no key rather than a
// list of them: enumerating the affected keys would mean keeping that list in
// step with every future query, and the one it missed would render another
// organization's rows under this one's name.
//
// The context is the exception, and the only one: it is what the switch itself
// answers with, and what a role gate reads.
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
      return apiFetch<OrganizationContext>("/organizations/me/switch", {
        method: "POST",
        body: JSON.stringify(body),
      })
    },
    onSuccess: (context) => {
      // Written, not invalidated, and written first: the invalidation below is
      // where every query decides whether to refetch, and one gated on the role
      // in here would decide it against the organization just left, asking for
      // a read the new one refuses (otari#1300). Refetching the context instead
      // would hold the caller's new role a round trip behind the reads already
      // being made in it.
      queryClient.setQueryData(ORGANIZATION_CONTEXT, context)
      void queryClient.invalidateQueries({
        predicate: (query) => query.queryHash !== ORGANIZATION_CONTEXT_HASH,
      })
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
      fetchAllPaged<OrganizationMember>("/organizations/me/members"),
    staleTime: 60_000,
    enabled,
  })
}

/**
 * One page of the roster, with the total.
 *
 * The rows carry their own workspaces, ceilings and spend since otari#1381, so
 * the members table can ask for a page rather than reading the roster to join
 * six other collections against it.
 *
 * Separate from `useOrganizationMembers` rather than replacing it: that one
 * still answers the two readers who genuinely want every row, a label lookup
 * and a candidate list, and both are tracked elsewhere (otari#1380).
 */
export function useOrganizationMembersPage(page: number, pageSize: number) {
  return useQuery({
    queryKey: [ORGANIZATION_MEMBERS, "page", page, pageSize],
    queryFn: () =>
      apiFetch<OrganizationMembers>(
        `/organizations/me/members?skip=${page * pageSize}&limit=${pageSize}`,
      ),
    staleTime: 60_000,
    placeholderData: keepPreviousData,
  })
}

export function useUpdateOrganization() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: UpdateOrganizationRequest) =>
      apiFetch<OrganizationContext>("/organizations/me", {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
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
        `/organizations/me/members/${encodeURIComponent(id)}`,
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
      apiFetch<void>(`/organizations/me/members/${encodeURIComponent(id)}`, {
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

// The dashboard's one write path onto the roster: lands `invited`, and the
// response always carries `accept_link` (whether or not `mail_sent` is true),
// so the operator can share it themselves.
export function useInviteOrganizationMember() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: InviteOrganizationMemberRequest) =>
      apiFetch<InviteOrganizationMemberResult>(
        "/organizations/me/member-invitations",
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
        `/organizations/me/member-invitations/${encodeURIComponent(invitationId)}`,
        { method: "DELETE" },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATION_MEMBERS] })
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATIONS] })
    },
  })
}

// The invitee's own inbox: which organizations are waiting on the signed-in
// identity. Its own read rather than a field on the memberships list, which is
// filtered to `active` on the server precisely because an `invited` membership
// is not somewhere the caller may act yet.
//
// Keyed under ORGANIZATIONS so accepting, declining, or switching invalidates
// it along with the rest of the tenancy cache. Same 404 guard as
// `useOrganizationMemberships`, for both of its reasons: a gateway older than
// this bundle does not serve this route, and a hybrid gateway answers 404 for
// every `/organizations` path by design. Neither is something a retry
// fixes, and the entry point that reads the count treats a failure as "nothing
// waiting" rather than showing an error in the chrome.
export function usePendingOrganizationInvitations() {
  return useQuery({
    queryKey: [ORGANIZATIONS, "pending-memberships"],
    queryFn: () =>
      fetchAllPaged<PendingOrganizationInvitation>(
        "/organizations/me/pending-memberships",
      ),
    staleTime: 60_000,
    retry: (failureCount, error) =>
      !(error instanceof ApiError && error.status === 404) && failureCount < 3,
  })
}

// Accepting lands a second *active* membership, which is a new row in the
// switcher, so the whole tenancy prefix goes rather than only the inbox key.
// Not `invalidateQueries()` with no key, unlike `useSwitchOrganization`:
// accepting does not move `active_organization_id`, so everything cached for
// the organization the caller is still acting in stays valid.
export function useAcceptPendingMembership() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (organizationMemberId: string) =>
      apiFetch<AcceptInvitationResult>(
        `/organizations/me/pending-memberships/${encodeURIComponent(
          organizationMemberId,
        )}/accept`,
        { method: "POST" },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATIONS] })
    },
  })
}

// Declining suspends a membership in an organization the caller was never
// active in, so the only list that changes is the inbox itself.
export function useDeclinePendingMembership() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (organizationMemberId: string) =>
      apiFetch<{ message: string }>(
        `/organizations/me/pending-memberships/${encodeURIComponent(
          organizationMemberId,
        )}/decline`,
        { method: "POST" },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({
        queryKey: [ORGANIZATIONS, "pending-memberships"],
      })
    },
  })
}

// The accept-invitation page's two calls. Both hit routes the server never
// gates on a session or the master key: the token in the emailed link is the
// caller's whole credential, and the gateway answers 404/400 for a bad one,
// never 401, so apiFetch's session-bounce never triggers here.
export function useValidateInvitation(token: string) {
  return useQuery({
    queryKey: ["invitation-preview", token],
    // POST with the token in the body, not a GET with it in the URL: the
    // token is a bearer credential, and a URL is what an access log or an
    // intermediate proxy routinely retains.
    queryFn: () =>
      apiFetch<InvitationPreview>("/invitations/validate", {
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
    mutationFn: (body: AcceptInvitationRequest) =>
      apiFetch<AcceptInvitationResult>("/invitations/accept", {
        method: "POST",
        body: JSON.stringify(body),
      }),
  })
}

// The public auth flows (otari#650). Same shape as the two invitation calls
// above and for the same reason: nothing here is gated on a session or the
// master key, because a caller completing a signup or opening an emailed link
// holds neither. The gateway answers 400 for a bad token, 429 when the shared
// sign-in limiter fires, and 503 when this deployment cannot send mail, so
// apiFetch's session-bounce never triggers on any of them.
//
// None of them invalidates anything. They write to an identity this
// unauthenticated caller cannot read back, and the cache they would touch
// belongs to a session that does not exist yet.

// Claims a roster identity by setting its password, then mails a verification
// link. The response is the same sentence whether the address was unknown,
// already claimed, or genuinely just claimed, so nothing here may branch on it.

/**
 * Refresh what one write to a provider key actually moved.
 *
 * Both flags default off, because both are expensive in their own way.
 * Refetching every key's model list is what the separate root key exists to
 * avoid (see `queryKeys.ts`), and only a write that moves a model row earns it.
 * The catalog is the other: creating a key offers its whole model list,
 * archiving one withdraws what it served, restoring one brings it back,
 * deleting one takes its rows, and a re-entered credential makes an unusable key
 * usable again.
 */
function invalidateOrgProviderKeys(
  queryClient: ReturnType<typeof useQueryClient>,
  { offeredModels = false, catalog = false } = {},
): void {
  void queryClient.invalidateQueries({
    queryKey: [ORGANIZATION_PROVIDER_KEYS],
  })
  if (offeredModels) {
    void queryClient.invalidateQueries({
      queryKey: [ORGANIZATION_PROVIDER_MODELS],
    })
  }
  if (catalog) {
    void queryClient.invalidateQueries({ queryKey: [MODELS] })
    void queryClient.invalidateQueries({ queryKey: [CATALOG] })
  }
}

// Every model write moves three reads: this key's panel, and both catalog
// surfaces, because an offered model appears in the listing carrying its price
// and a withdrawn one leaves it. The same three `invalidateOrganizationPricing`
// moves, for the same reason.
function invalidateOrgProviderModels(
  queryClient: ReturnType<typeof useQueryClient>,
  keyId: string,
): void {
  void queryClient.invalidateQueries({
    queryKey: [ORGANIZATION_PROVIDER_MODELS, keyId],
  })
  void queryClient.invalidateQueries({ queryKey: [MODELS] })
  void queryClient.invalidateQueries({ queryKey: [CATALOG] })
}

// The organization's own upstream provider credentials (#670), which every
// workspace under it inherits. A different table from `/provider-credentials`
// above: that one is keyed on an instance name and belongs to the process, this
// one belongs to the tenant. Only a hosted deployment reports the surface these
// hooks serve (`organization_providers`).
//
// Archived keys are fetched too, and filtered in the page rather than by a
// second query: archiving is reversible, the list is small, and a toggle that
// refetched would make "show archived" a network round trip for a set the
// browser already holds.
//
// The list itself is organization owner/admin-gated, not member-readable
// (otari-ai#1944), so a caller reachable by a plain member passes `enabled`
// false for them rather than rendering the refusal.
export function useOrgProviderKeys(enabled = true) {
  return useQuery({
    queryKey: [ORGANIZATION_PROVIDER_KEYS],
    queryFn: () =>
      fetchAllPaged<OrgProviderKey>("/organizations/me/provider-keys", {
        include_archived: "true",
      }),
    staleTime: 60_000,
    enabled,
  })
}

export function useCreateOrgProviderKey() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateOrgProviderKeyRequest) =>
      apiFetch<OrgProviderKey>("/organizations/me/provider-keys", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () =>
      // The create offers everything the credential reaches, so the whole
      // catalog moves with it.
      invalidateOrgProviderKeys(queryClient, {
        offeredModels: true,
        catalog: true,
      }),
  })
}

export function useUpdateOrgProviderKey() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      keyId,
      body,
    }: {
      keyId: string
      body: UpdateOrgProviderKeyRequest
    }) =>
      apiFetch<OrgProviderKey>(
        `/organizations/me/provider-keys/${encodeURIComponent(keyId)}`,
        { method: "PATCH", body: JSON.stringify(body) },
      ),
    onSuccess: () =>
      // No offered row moves, but the catalog can: a key whose credential will
      // not decrypt is unusable, and an unusable key contributes nothing
      // (`organization_model_access.key_is_usable`), so re-entering a working
      // one puts its models back.
      invalidateOrgProviderKeys(queryClient, { catalog: true }),
  })
}

// Archiving also clears the organization default, and setting one clears it on
// whichever key held it, so every one of these four re-reads the whole list
// rather than patching the row it acted on.
export function useArchiveOrgProviderKey() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (keyId: string) =>
      apiFetch<OrgProviderKey>(
        `/organizations/me/provider-keys/${encodeURIComponent(keyId)}/archive`,
        { method: "POST" },
      ),
    onSuccess: () =>
      // Archiving withdraws what the key served without touching its rows.
      invalidateOrgProviderKeys(queryClient, { catalog: true }),
  })
}

export function useRestoreOrgProviderKey() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (keyId: string) =>
      apiFetch<OrgProviderKey>(
        `/organizations/me/provider-keys/${encodeURIComponent(keyId)}/restore`,
        { method: "POST" },
      ),
    onSuccess: () =>
      // Restoring serves them again.
      invalidateOrgProviderKeys(queryClient, { catalog: true }),
  })
}

export function useSetOrgProviderKeyDefault() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (keyId: string) =>
      apiFetch<OrgProviderKey>(
        `/organizations/me/provider-keys/${encodeURIComponent(keyId)}/default`,
        { method: "POST" },
      ),
    onSuccess: () =>
      // Which key dispatches, not which models exist.
      invalidateOrgProviderKeys(queryClient),
  })
}

// Permanent, and the API accepts it only for a key that is already archived.
export function useDeleteOrgProviderKey() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (keyId: string) =>
      apiFetch<{ message: string }>(
        `/organizations/me/provider-keys/${encodeURIComponent(keyId)}`,
        { method: "DELETE" },
      ),
    onSuccess: () =>
      // The rows cascade away with the key.
      invalidateOrgProviderKeys(queryClient, {
        offeredModels: true,
        catalog: true,
      }),
  })
}

// An organization's email-domain claims.
//
// Every mutation invalidates the whole list rather than patching its row.
// Verifying is the reason: it is the one call whose answer the server decides
// (the TXT lookup either finds the record or does not), so a cache patched from
// the request body would show a claim as verified that is not.

export function useOrganizationDomains(enabled = true) {
  return useQuery({
    queryKey: [ORGANIZATION_DOMAINS],
    queryFn: () =>
      apiFetch<{ data: OrganizationDomain[]; count: number }>(
        "/organizations/me/domains",
      ),
    staleTime: 60_000,
    enabled,
  })
}

function invalidateOrganizationDomains(
  queryClient: ReturnType<typeof useQueryClient>,
) {
  void queryClient.invalidateQueries({ queryKey: [ORGANIZATION_DOMAINS] })
}

export function useCreateOrganizationDomain() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateOrganizationDomainRequest) =>
      apiFetch<OrganizationDomain>("/organizations/me/domains", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => invalidateOrganizationDomains(queryClient),
  })
}

export function useUpdateOrganizationDomain() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      domainId,
      body,
    }: {
      domainId: string
      body: UpdateOrganizationDomainRequest
    }) =>
      apiFetch<OrganizationDomain>(
        `/organizations/me/domains/${encodeURIComponent(domainId)}`,
        { method: "PATCH", body: JSON.stringify(body) },
      ),
    onSuccess: () => invalidateOrganizationDomains(queryClient),
  })
}

export function useVerifyOrganizationDomain() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (domainId: string) =>
      apiFetch<OrganizationDomain>(
        `/organizations/me/domains/${encodeURIComponent(domainId)}/verify`,
        { method: "POST" },
      ),
    onSuccess: () => invalidateOrganizationDomains(queryClient),
  })
}

export function useDeleteOrganizationDomain() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (domainId: string) =>
      apiFetch<{ message: string }>(
        `/organizations/me/domains/${encodeURIComponent(domainId)}`,
        { method: "DELETE" },
      ),
    onSuccess: () => invalidateOrganizationDomains(queryClient),
  })
}

// The models an organization offers on one of its provider keys, and what each
// currently costs it. Paged on the server, because one provider can list several
// hundred models.
//
// `placeholderData` follows `useOrganizationPricing`'s shape rather than a bare
// `keepPreviousData`, and the difference is load-bearing here: the key id is
// part of the query key, so `keepPreviousData` would paint one provider's models
// under another provider's name for a frame every time a different row is
// expanded.
export function useOrgProviderModels(
  keyId: string,
  page: number,
  pageSize: number,
  enabled = true,
) {
  return useQuery({
    queryKey: [ORGANIZATION_PROVIDER_MODELS, keyId, page, pageSize],
    queryFn: () =>
      apiFetch<OrgProviderModels>(
        `/organizations/me/provider-keys/${encodeURIComponent(keyId)}/models?skip=${page * pageSize}&limit=${pageSize}`,
      ),
    staleTime: 60_000,
    placeholderData: (previous, previousQuery) =>
      previousQuery?.queryKey[1] === keyId ? previous : undefined,
    enabled,
  })
}

// What the provider says it serves on the stored credential. Answering means
// dialing the upstream, so it is fetched only while the add-model form is open
// and held for a minute: one dial per form visit, not one per re-render.
export function useOrgProviderAvailableModels(keyId: string, enabled: boolean) {
  return useQuery({
    ...NO_RETRY,
    queryKey: [ORGANIZATION_PROVIDER_AVAILABLE_MODELS, keyId],
    queryFn: () =>
      apiFetch<OrgProviderAvailableModels>(
        `/organizations/me/provider-keys/${encodeURIComponent(keyId)}/available-models`,
      ),
    staleTime: 60_000,
    enabled,
  })
}

export function useOfferOrgProviderModel(keyId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: OfferOrgProviderModelRequest) =>
      apiFetch<OrgProviderModel>(
        `/organizations/me/provider-keys/${encodeURIComponent(keyId)}/models`,
        { method: "POST", body: JSON.stringify(body) },
      ),
    onSuccess: () => invalidateOrgProviderModels(queryClient, keyId),
  })
}

export function useSetOrgProviderModelEnabled(keyId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      modelId,
      ...body
    }: { modelId: string } & UpdateOrgProviderModelRequest) =>
      apiFetch<OrgProviderModel>(
        `/organizations/me/provider-keys/${encodeURIComponent(keyId)}/models/${encodeURIComponent(modelId)}`,
        { method: "PATCH", body: JSON.stringify(body) },
      ),
    onSuccess: () => invalidateOrgProviderModels(queryClient, keyId),
  })
}

export function useWithdrawOrgProviderModel(keyId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (modelId: string) =>
      apiFetch<{ message: string }>(
        `/organizations/me/provider-keys/${encodeURIComponent(keyId)}/models/${encodeURIComponent(modelId)}`,
        { method: "DELETE" },
      ),
    onSuccess: () => invalidateOrgProviderModels(queryClient, keyId),
  })
}

// Both refreshes carry `longRequestSignal`, as the deployment pricing preview
// does: one re-dials the provider and the other walks the community dataset per
// model, and neither fits the default request budget.
export function useRefreshOrgProviderModels(keyId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () =>
      apiFetch<OrgProviderModelsRefresh>(
        `/organizations/me/provider-keys/${encodeURIComponent(keyId)}/models/refresh`,
        { method: "POST", signal: longRequestSignal() },
      ),
    onSuccess: () => {
      invalidateOrgProviderModels(queryClient, keyId)
      // The dial's answer is fresher than whatever the picker last cached.
      void queryClient.invalidateQueries({
        queryKey: [ORGANIZATION_PROVIDER_AVAILABLE_MODELS, keyId],
      })
    },
  })
}

export function useRefreshOrgProviderModelPricing(keyId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () =>
      apiFetch<OrgProviderModelsRefresh>(
        `/organizations/me/provider-keys/${encodeURIComponent(keyId)}/pricing/refresh`,
        { method: "POST", signal: longRequestSignal() },
      ),
    // Not the available-models key: this one never asks the provider anything.
    onSuccess: () => invalidateOrgProviderModels(queryClient, keyId),
  })
}

// A workspace's code-execution policy over the deployment-wide sandbox. One
// object or none, so it is a plain read rather than a paged list, and it is
// nested under the workspaces key for the same reason the budget defaults are.
