import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  CallerIdentity,
  MailedLinkStatusResponse,
  OrganizationContext,
  Passkey,
  PasskeysResponse,
  PasswordResponse,
  RenamePasskeyRequest,
  RequestPasswordResetResponse,
  ResendVerificationResponse,
  ResetPasswordRequest,
  RotateMasterKeyResponse,
  SetPasswordRequest,
  SignupRequest,
  SignupResponse,
  UpdateProfileRequest,
  VerifyEmailRequest,
  VerifyEmailResponse,
} from "@/client"
import { apiFetch, retryUnlessNotFound } from "@/shared/api/client"
import {
  MAILED_LINK_STATUS,
  NO_RETRY,
  ORGANIZATION_MEMBERS,
  ORGANIZATIONS,
  PASSKEYS,
  PROFILE,
} from "@/shared/api/queryKeys"
import { createPasskey } from "@/shared/helpers/webauthn"

export function useRotateMasterKey() {
  return useMutation({
    mutationFn: () =>
      apiFetch<RotateMasterKeyResponse>("/settings/master-key/rotate", {
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
 * Three things this changes are cached elsewhere, and they are cached
 * differently. The bootstrap's `sign_in_methods` is a context read once per
 * load rather than a query, so no invalidation could reach it: the caller
 * reports the claim through `useRetireMasterKeySignIn` instead. The roster is
 * an ordinary query, and a claim writes `user.email` from null to the address,
 * so the Members page would otherwise show the row it fetched before the claim
 * for the rest of its `staleTime`. That one is invalidated here.
 *
 * The third is the membership context's `caller`, which carries the two facts
 * the account page builds its form from: the address, and `has_password`. Both
 * move on this call and the account page has to see them move, or the card that
 * just set a first password goes on offering to set one. Seeded from the
 * response before being invalidated, for the reason `useUpdateProfile` does the
 * same: the page settles on the same tick rather than a round trip later.
 *
 * Every *other* session this identity holds is revoked server-side; this one is
 * kept, so no 401 follows.
 */
export function useSetPassword() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: SetPasswordRequest) =>
      apiFetch<PasswordResponse>("/auth/password", {
        method: "PUT",
        body: JSON.stringify(body),
      }),
    onSuccess: (result) => {
      queryClient.setQueryData<OrganizationContext>(
        [ORGANIZATIONS, "context"],
        (previous) =>
          previous?.caller
            ? {
                ...previous,
                caller: {
                  ...previous.caller,
                  email: result.email,
                  has_password: true,
                  claims_deployment: false,
                },
              }
            : previous,
      )
      void queryClient.invalidateQueries({
        queryKey: [ORGANIZATIONS, "context"],
      })
      void queryClient.invalidateQueries({ queryKey: [PROFILE] })
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATION_MEMBERS] })
    },
  })
}

/**
 * Who the signed-in identity is (`GET /v1/auth/profile`), whether or not it
 * belongs to an organization.
 *
 * The organization context carries the same `caller` and is read on every page
 * anyway, so this is for what that context cannot answer: a caller with no
 * organization, whom it refuses with a 404. A 404 here is a gateway older than
 * the route, so it is not retried, and a caller reads it as "unknown".
 */
export function useProfile(enabled = true) {
  return useQuery({
    queryKey: [PROFILE],
    queryFn: () => apiFetch<CallerIdentity>("/auth/profile"),
    staleTime: 60_000,
    retry: retryUnlessNotFound(useQueryClient()),
    enabled,
  })
}

/**
 * Change the name the signed-in identity goes by (`PATCH /v1/auth/profile`).
 *
 * Always the caller's own identity: the endpoint takes no id. `null` clears the
 * name, which is the state a roster entry added by address starts in, and every
 * surface that draws a person falls back to the address from there.
 *
 * Three caches carry that name. The membership context is where the sidebar
 * reads it, and the response is exactly the `caller` it holds, so that one is
 * seeded from the answer before being invalidated: the account control renames
 * itself on the same tick rather than a round trip later. The profile read is
 * the same shape for a caller with no organization, and is seeded the same way.
 * The roster is the third, where
 * `userDisplay` resolves a user id for Usage, Activity and Budgets. The context
 * is touched at its own key rather than at `[ORGANIZATIONS]`, which would re-read
 * the memberships list as well and nothing here moves it.
 */
export function useUpdateProfile() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: UpdateProfileRequest) =>
      apiFetch<CallerIdentity>("/auth/profile", {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: (caller) => {
      queryClient.setQueryData<OrganizationContext>(
        [ORGANIZATIONS, "context"],
        (previous) => (previous ? { ...previous, caller } : previous),
      )
      queryClient.setQueryData<CallerIdentity>([PROFILE], caller)
      void queryClient.invalidateQueries({
        queryKey: [ORGANIZATIONS, "context"],
      })
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
    queryFn: () => apiFetch<PasskeysResponse>("/auth/webauthn/credentials"),
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
        "/auth/webauthn/register/options",
        { method: "POST" },
      )
      const credential = await createPasskey(
        options as Parameters<typeof createPasskey>[0],
      )
      return apiFetch<Passkey>("/auth/webauthn/register", {
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
      apiFetch<Passkey>(`/auth/webauthn/credentials/${id}`, {
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
      apiFetch<void>(`/auth/webauthn/credentials/${id}`, {
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

export function useSignup() {
  return useMutation({
    mutationFn: (body: SignupRequest) =>
      apiFetch<SignupResponse>("/auth/signup", {
        method: "POST",
        body: JSON.stringify(body),
      }),
  })
}

// The link verifies only together with the password chosen at signup, so this
// runs behind a button rather than on arrival. The token travels in the body,
// not in a URL an access log or a proxy routinely retains.
export function useVerifyEmail() {
  return useMutation({
    mutationFn: (body: VerifyEmailRequest) =>
      apiFetch<VerifyEmailResponse>("/auth/verify-email", {
        method: "POST",
        body: JSON.stringify(body),
      }),
  })
}

export function useResendVerification() {
  return useMutation({
    mutationFn: (email: string) =>
      apiFetch<ResendVerificationResponse>("/auth/resend-verification", {
        method: "POST",
        body: JSON.stringify({ email }),
      }),
  })
}

export function useRequestPasswordReset() {
  return useMutation({
    mutationFn: (email: string) =>
      apiFetch<RequestPasswordResetResponse>("/auth/password/reset", {
        method: "POST",
        body: JSON.stringify({ email }),
      }),
  })
}

/**
 * Whether a mailed reset or verification link can still be used, asked on
 * arrival so a spent or expired link says so before anyone types a password.
 *
 * Checking does not use the link. POST with the token in the body for the
 * reason `useValidateInvitation` gives. Not retried and not refetched: the
 * answer only moves when the page itself uses the link, and a gateway that
 * cannot answer (one older than the route, or a throttle) leaves the page on
 * its form, which still works.
 */
export function useMailedLinkStatus(kind: "reset" | "verify", token: string) {
  return useQuery({
    queryKey: [MAILED_LINK_STATUS, kind, token],
    queryFn: async () => {
      const result = await apiFetch<MailedLinkStatusResponse>(
        kind === "reset"
          ? "/auth/password/reset/validate"
          : "/auth/verify-email/validate",
        { method: "POST", body: JSON.stringify({ token }) },
      )
      return result.status
    },
    enabled: token.length > 0,
    staleTime: Number.POSITIVE_INFINITY,
    refetchOnWindowFocus: false,
    ...NO_RETRY,
  })
}

// 204, so there is nothing to read back: the caller learns it worked by the
// call not raising, and signs in with the new password from the sign-in screen.
export function useResetPassword() {
  return useMutation({
    mutationFn: (body: ResetPasswordRequest) =>
      apiFetch<void>("/auth/password/reset/confirm", {
        method: "POST",
        body: JSON.stringify(body),
      }),
  })
}

// `enabled` lets a page that only sometimes offers a workspace control (the
// organization-wide Usage page's filter) skip the read everywhere else.
