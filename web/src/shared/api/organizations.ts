import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  AcceptInvitationResult,
  CallerOrganizationMembership,
  CreateOrganizationDomainRequest,
  CreateOrganizationMemberRequest,
  CreateOrganizationMemberResult,
  CreateOrganizationRequest,
  CreateOrgProviderKeyRequest,
  InvitationPreview,
  InviteOrganizationMemberRequest,
  InviteOrganizationMemberResult,
  Organization,
  OrganizationContext,
  OrganizationDomain,
  OrganizationMember,
  OrgProviderKey,
  PendingOrganizationInvitation,
  SwitchOrganizationRequest,
  UpdateOrganizationDomainRequest,
  UpdateOrganizationMemberRequest,
  UpdateOrganizationRequest,
  UpdateOrgProviderKeyRequest,
} from "@/client"
import { ApiError, apiFetch } from "@/shared/api/client"
import { fetchAllPaged } from "@/shared/api/paging"
import {
  ORGANIZATION_DOMAINS,
  ORGANIZATION_MEMBERS,
  ORGANIZATION_PROVIDER_KEYS,
  ORGANIZATIONS,
  WORKSPACES,
} from "@/shared/api/queryKeys"

export function useOrganizationContext() {
  return useQuery({
    queryKey: [ORGANIZATIONS, "context"],
    queryFn: () => apiFetch<OrganizationContext>("/v1/organizations/me"),
    staleTime: 60_000,
  })
}

// Whether the deployment can encrypt a provider credential at rest, i.e. whether
// OTARI_SECRET_KEY is set on the server. Both provider-key pages gate their add
// control on it, so the rule lives here rather than twice.
//
// Read off the membership context rather than `/v1/settings`, which reports the
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

// The invitee's own inbox: which organizations are waiting on the signed-in
// identity. Its own read rather than a field on the memberships list, which is
// filtered to `active` on the server precisely because an `invited` membership
// is not somewhere the caller may act yet.
//
// Keyed under ORGANIZATIONS so accepting, declining, or switching invalidates
// it along with the rest of the tenancy cache. Same 404 guard as
// `useOrganizationMemberships`, for both of its reasons: a gateway older than
// this bundle does not serve this route, and a hybrid gateway answers 404 for
// every `/v1/organizations` path by design. Neither is something a retry
// fixes, and the entry point that reads the count treats a failure as "nothing
// waiting" rather than showing an error in the chrome.
export function usePendingOrganizationInvitations() {
  return useQuery({
    queryKey: [ORGANIZATIONS, "pending-memberships"],
    queryFn: () =>
      fetchAllPaged<PendingOrganizationInvitation>(
        "/v1/organizations/me/pending-memberships",
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
        `/v1/organizations/me/pending-memberships/${encodeURIComponent(
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
        `/v1/organizations/me/pending-memberships/${encodeURIComponent(
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
// apiFetch's session-bounce never triggers on any of them.
//
// None of them invalidates anything. They write to an identity this
// unauthenticated caller cannot read back, and the cache they would touch
// belongs to a session that does not exist yet.

// Claims a roster identity by setting its password, then mails a verification
// link. The response is the same sentence whether the address was unknown,
// already claimed, or genuinely just claimed, so nothing here may branch on it.

function invalidateOrgProviderKeys(
  queryClient: ReturnType<typeof useQueryClient>,
): void {
  void queryClient.invalidateQueries({
    queryKey: [ORGANIZATION_PROVIDER_KEYS],
  })
}

// The organization's own upstream provider credentials (#670), which every
// workspace under it inherits. A different table from `/v1/provider-credentials`
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
      fetchAllPaged<OrgProviderKey>("/v1/organizations/me/provider-keys", {
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
      apiFetch<OrgProviderKey>("/v1/organizations/me/provider-keys", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => invalidateOrgProviderKeys(queryClient),
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
        `/v1/organizations/me/provider-keys/${encodeURIComponent(keyId)}`,
        { method: "PATCH", body: JSON.stringify(body) },
      ),
    onSuccess: () => invalidateOrgProviderKeys(queryClient),
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
        `/v1/organizations/me/provider-keys/${encodeURIComponent(keyId)}/archive`,
        { method: "POST" },
      ),
    onSuccess: () => invalidateOrgProviderKeys(queryClient),
  })
}

export function useRestoreOrgProviderKey() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (keyId: string) =>
      apiFetch<OrgProviderKey>(
        `/v1/organizations/me/provider-keys/${encodeURIComponent(keyId)}/restore`,
        { method: "POST" },
      ),
    onSuccess: () => invalidateOrgProviderKeys(queryClient),
  })
}

export function useSetOrgProviderKeyDefault() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (keyId: string) =>
      apiFetch<OrgProviderKey>(
        `/v1/organizations/me/provider-keys/${encodeURIComponent(keyId)}/default`,
        { method: "POST" },
      ),
    onSuccess: () => invalidateOrgProviderKeys(queryClient),
  })
}

// Permanent, and the API accepts it only for a key that is already archived.
export function useDeleteOrgProviderKey() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (keyId: string) =>
      apiFetch<{ message: string }>(
        `/v1/organizations/me/provider-keys/${encodeURIComponent(keyId)}`,
        { method: "DELETE" },
      ),
    onSuccess: () => invalidateOrgProviderKeys(queryClient),
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
        "/v1/organizations/me/domains",
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
      apiFetch<OrganizationDomain>("/v1/organizations/me/domains", {
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
        `/v1/organizations/me/domains/${encodeURIComponent(domainId)}`,
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
        `/v1/organizations/me/domains/${encodeURIComponent(domainId)}/verify`,
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
        `/v1/organizations/me/domains/${encodeURIComponent(domainId)}`,
        { method: "DELETE" },
      ),
    onSuccess: () => invalidateOrganizationDomains(queryClient),
  })
}

// A workspace's code-execution policy over the deployment-wide sandbox. One
// object or none, so it is a plain read rather than a paged list, and it is
// nested under the workspaces key for the same reason the budget defaults are.
