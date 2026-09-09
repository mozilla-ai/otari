import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
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
  VerifyEmailResponse,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import {
  NO_RETRY,
  ORGANIZATION_MEMBERS,
  PASSKEYS,
} from "@/shared/api/queryKeys"
import { createPasskey } from "@/shared/helpers/webauthn"

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

// `enabled` lets a page that only sometimes offers a workspace control (the
// organization-wide Usage page's filter) skip the read everywhere else.
