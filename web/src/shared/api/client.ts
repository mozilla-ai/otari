// Thin fetch wrapper for the gateway's management API. The dashboard is served
// from the same origin as the API, so paths are relative ("/v1/models") and the
// HttpOnly session cookie minted at sign-in rides along automatically (fetch
// defaults to credentials: "same-origin"). A credential is sent exactly once, to
// POST /v1/auth/session, and is never written to browser storage: it lives in
// the sign-in form's state until the request goes out and is gone on reload.

import { getPasskeyAssertion } from "@/shared/helpers/webauthn"

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = "ApiError"
    this.status = status
  }
}

// AuthProvider registers a callback so a 401 anywhere can drop the session
// and bounce the operator back to the login screen.
let unauthorizedHandler: (() => void) | null = null

export function setUnauthorizedHandler(handler: (() => void) | null): void {
  unauthorizedHandler = handler
}

// Reads the body once and reports both what to show and who wrote it. `detail`
// is non-null only when the body is JSON carrying a string `detail`, which is
// the shape every refusal this gateway writes has and the shape an intermediary
// answering for it does not. Callers that only need something to display take
// `message`; the one caller that has to tell a gateway refusal from a proxy's
// own status takes `detail`.
async function readRefusal(
  response: Response,
): Promise<{ detail: string | null; message: string }> {
  let detail: string | null = null
  let message: string | null = null
  try {
    const data = JSON.parse(await response.text()) as { detail?: unknown }
    if (typeof data.detail === "string") {
      detail = data.detail
      message = data.detail
    } else if (data.detail != null) {
      message = JSON.stringify(data.detail)
    }
  } catch {
    // Body was not JSON; fall through to the status text.
  }
  return {
    detail,
    message:
      message ?? (response.statusText || `Request failed (${response.status})`),
  }
}

async function extractErrorMessage(response: Response): Promise<string> {
  return (await readRefusal(response)).message
}

// The credentials POST /v1/auth/session accepts. Exactly one form per request:
// the gateway refuses a body carrying both. Which one a deployment currently
// takes is published in the bootstrap's `sign_in_methods`, so the sign-in screen
// renders the form that will work rather than discovering it from a refusal.
export type SignInCredential =
  | { masterKey: string }
  | { email: string; password: string }

// A refusal carries the gateway's own explanation rather than a bare false,
// because the refusals mean different things and only the server knows which
// applies: a 401 is a wrong credential, a 403 is the master key presented to a
// deployment that has retired it as a sign-in, and a 503 is maintenance mode
// freezing every credential while the gateway is redeployed. Rendering
// "Invalid master key." over any of the last two, as this did before there was
// a second credential, tells the operator to retry the thing that cannot work.
export interface SignInResult {
  ok: boolean
  message?: string
  /**
   * The refusal's status, on a refusal. Present so a caller can tell the two
   * apart without re-reading the message: the wording is the gateway's and is
   * the one part of a refusal that must not be recorded anywhere.
   */
  status?: number
}

// Exchange a credential for a server-issued session: the gateway verifies it and
// answers with an HttpOnly cookie holding an opaque session token, so the
// credential itself never needs to be stored (or even kept in memory)
// afterwards. Refusals come back as `ok: false` with the gateway's message:
// 401 and 403 always, and 503 when the gateway wrote the body (maintenance
// mode) rather than an intermediary answering for it. Network faults, an
// unreachable gateway, and other failures throw ApiError so the UI can explain
// them.
export async function createSession(
  credential: SignInCredential,
): Promise<SignInResult> {
  const body =
    "masterKey" in credential
      ? { master_key: credential.masterKey }
      : { email: credential.email, password: credential.password }
  let response: Response
  try {
    response = await fetch("/v1/auth/session", {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    })
  } catch (error) {
    if (isTimeout(error)) {
      throw new ApiError(0, TIMEOUT_MESSAGE)
    }
    throw new ApiError(0, "Network error: could not reach the gateway.")
  }
  if (response.status === 401 || response.status === 403) {
    return {
      ok: false,
      message: await extractErrorMessage(response),
      status: response.status,
    }
  }
  // 503 is maintenance mode, and a refusal the gateway wrote belongs with the
  // other two rather than on the throw path: it is a deliberate answer, in
  // wording meant for the person reading it, not a fault. The sign-in screen
  // normally renders a notice instead of the form on a frozen deployment (it
  // reads the same flag from the bootstrap); this is the tab that was already
  // open when the freeze started.
  //
  // Gated on the gateway having written it, not on the status alone. The
  // redeploy this feature exists for is exactly when a proxy with no healthy
  // upstream answers 503 itself, and that body carries no `detail`. Treating it
  // as a refusal would render "Service Unavailable" on a credential's label
  // row, which says the credential was rejected by a gateway that never saw it.
  // Those take the ApiError path and are explained as the fault they are.
  if (response.status === 503) {
    const refusal = await readRefusal(response)
    if (refusal.detail !== null) {
      return { ok: false, message: refusal.detail, status: response.status }
    }
    throw new ApiError(response.status, refusal.message)
  }
  if (!response.ok) {
    throw new ApiError(response.status, await extractErrorMessage(response))
  }
  return { ok: true }
}

// Sign in with a passkey: two calls with a browser ceremony between them.
//
// Hand-written here beside `createSession`, and for the same reason: `apiFetch`
// treats a 401 as an expired session and bounces to the sign-in screen, which
// is exactly wrong on the screen somebody is signing in *from*. A refused
// passkey comes back as `ok: false` carrying the gateway's own message.
//
// A dismissed prompt is not a refusal and is not reported as one: the ceremony
// throws `PasskeyCancelledError`, which the caller distinguishes.
export async function signInWithPasskey(): Promise<SignInResult> {
  const options = await publicPost("/v1/auth/webauthn/authenticate/options")
  if (!options.ok) {
    return { ok: false, message: options.message, status: options.status }
  }
  const assertion = await getPasskeyAssertion(
    options.body as Parameters<typeof getPasskeyAssertion>[0],
  )
  const verified = await publicPost("/v1/auth/webauthn/authenticate", {
    credential: assertion,
  })
  return verified.ok
    ? { ok: true }
    : { ok: false, message: verified.message, status: verified.status }
}

// One unauthenticated POST, with the sign-in screen's error handling: a 401 or
// 403 is the gateway's answer rather than an exception, and anything else is a
// failure the screen cannot explain away.
async function publicPost(
  path: string,
  body?: unknown,
): Promise<{
  ok: boolean
  message?: string
  status?: number
  body?: unknown
}> {
  let response: Response
  try {
    response = await fetch(path, {
      method: "POST",
      headers: {
        Accept: "application/json",
        ...(body === undefined ? {} : { "Content-Type": "application/json" }),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    })
  } catch (error) {
    if (isTimeout(error)) {
      throw new ApiError(0, TIMEOUT_MESSAGE)
    }
    throw new ApiError(0, "Network error: could not reach the gateway.")
  }
  if (response.status === 401 || response.status === 403) {
    // The status travels with the refusal for the reason `SignInResult.status`
    // documents: the caller records which refusal happened without touching the
    // message, which is the gateway's wording and the one part that must not be.
    return {
      ok: false,
      message: await extractErrorMessage(response),
      status: response.status,
    }
  }
  if (!response.ok) {
    throw new ApiError(response.status, await extractErrorMessage(response))
  }
  return { ok: true, body: await response.json() }
}

// Best-effort server-side sign-out: revokes the cookie's session and expires
// the cookie. Uses raw fetch (not apiFetch) and swallows failures so the
// 401-bounce path can call it without re-entering the unauthorized handler.
// Bounded like every other management call: an unbounded sign-out could
// otherwise stay in flight past a subsequent sign-in and clobber its fresh
// cookie with this call's expiring one (see #557).
export async function deleteSession(): Promise<void> {
  try {
    await fetch("/v1/auth/session", {
      method: "DELETE",
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    })
  } catch {
    // Signing out locally still proceeds; the session expires on its TTL.
  }
}

// Upper bound on any single management call. Nothing here should take this
// long: the gateway bounds its own provider fan-out well below it. The point is
// that a request which hangs anyway (dead socket, stalled proxy) gives its
// browser connection slot back on a deadline we control instead of holding it
// open. On HTTP/1.1 a browser allows only ~6 sockets per origin, so a handful of
// hung requests is enough to queue everything an operator clicks afterwards.
// Callers pass their own `signal` to override.
const REQUEST_TIMEOUT_MS = 30_000
const TIMEOUT_MESSAGE = `The gateway did not respond within ${REQUEST_TIMEOUT_MS / 1000}s.`

// For the handful of calls whose work scales with the data rather than with one
// upstream hop: the bulk usage delete and reprice, and the pricing-snapshot
// refresh. `DELETE /v1/usage` with `by_filter` issues one unbounded DELETE and
// the reprice loops over every matched row, so on a large imported-usage table
// either can outrun the 30s bound above. Aborting them is worse than waiting: the
// server transaction commits regardless of whether the browser is still
// listening, so the operator would be told the delete failed when it succeeded,
// and the obvious next move is to run it again. Still bounded, because a socket
// held forever is what the deadline exists to prevent.
export const LONG_REQUEST_TIMEOUT_MS = 5 * 60_000

/** Signal for a request whose duration scales with the data, not with one hop. */
export function longRequestSignal(): AbortSignal {
  return AbortSignal.timeout(LONG_REQUEST_TIMEOUT_MS)
}

// A TimeoutError from AbortSignal.timeout means we gave up, not that the gateway
// is unreachable; saying so points at the right thing to look at. It can surface
// from either await: fetch() resolves once headers arrive, so a body that then
// stalls trips the same deadline on the JSON read instead.
function isTimeout(error: unknown): boolean {
  return error instanceof DOMException && error.name === "TimeoutError"
}

export async function apiFetch<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const headers = new Headers(init.headers)
  headers.set("Accept", "application/json")
  if (init.body != null && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json")
  }
  const signal = init.signal ?? AbortSignal.timeout(REQUEST_TIMEOUT_MS)
  // Only name the deadline when it is ours; a caller-supplied signal has its own
  // budget, and quoting 30s at an operator who waited five minutes is worse than
  // saying nothing.
  const timeoutMessage = init.signal
    ? "The gateway did not respond in time."
    : TIMEOUT_MESSAGE

  let response: Response
  try {
    response = await fetch(path, { ...init, headers, signal })
  } catch (error) {
    if (isTimeout(error)) {
      throw new ApiError(0, timeoutMessage)
    }
    throw new ApiError(0, "Network error: could not reach the gateway.")
  }

  // 401 (expired/revoked session) or 403 both mean this session can't use the
  // management API anymore: drop it and bounce to sign-in.
  if (response.status === 401 || response.status === 403) {
    unauthorizedHandler?.()
    throw new ApiError(response.status, await extractErrorMessage(response))
  }

  if (!response.ok) {
    throw new ApiError(response.status, await extractErrorMessage(response))
  }

  if (response.status === 204) {
    return undefined as T
  }

  try {
    return (await response.json()) as T
  } catch (error) {
    // Every caller expects an ApiError; a raw DOMException here would reach the
    // UI as an unrecognized failure. A malformed body is still its own error.
    if (isTimeout(error)) {
      throw new ApiError(0, timeoutMessage)
    }
    throw error
  }
}
