// Thin fetch wrapper for the gateway's management API. The dashboard is served
// from the same origin as the API, so paths are relative ("/v1/models") and the
// HttpOnly session cookie minted at sign-in rides along automatically (fetch
// defaults to credentials: "same-origin"). The raw master key is sent exactly
// once, to POST /v1/auth/session, and never stored in the browser.

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

// AuthProvider registers a callback so a 401 anywhere can drop the session
// and bounce the operator back to the login screen.
let unauthorizedHandler: (() => void) | null = null;

export function setUnauthorizedHandler(handler: (() => void) | null): void {
  unauthorizedHandler = handler;
}

async function extractErrorMessage(response: Response): Promise<string> {
  try {
    const data = (await response.json()) as { detail?: unknown };
    if (typeof data.detail === "string") {
      return data.detail;
    }
    if (data.detail != null) {
      return JSON.stringify(data.detail);
    }
  } catch {
    // Body was not JSON; fall through to the status text.
  }
  return response.statusText || `Request failed (${response.status})`;
}

// Exchange the master key for a server-issued session: the gateway verifies the
// key and answers with an HttpOnly cookie holding an opaque session token, so
// the key itself never needs to be stored (or even kept in memory) afterwards.
// Returns false on 401/403 (wrong key) and throws ApiError for network/other
// failures so the UI can explain them.
export async function createSession(key: string): Promise<boolean> {
  let response: Response;
  try {
    response = await fetch("/v1/auth/session", {
      method: "POST",
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: JSON.stringify({ master_key: key }),
    });
  } catch {
    throw new ApiError(0, "Network error: could not reach the gateway.");
  }
  if (response.status === 401 || response.status === 403) {
    return false;
  }
  if (!response.ok) {
    throw new ApiError(response.status, await extractErrorMessage(response));
  }
  return true;
}

// Best-effort server-side sign-out: revokes the cookie's session and expires
// the cookie. Uses raw fetch (not apiFetch) and swallows failures so the
// 401-bounce path can call it without re-entering the unauthorized handler.
export async function deleteSession(): Promise<void> {
  try {
    await fetch("/v1/auth/session", { method: "DELETE" });
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
const REQUEST_TIMEOUT_MS = 30_000;
const TIMEOUT_MESSAGE = `The gateway did not respond within ${REQUEST_TIMEOUT_MS / 1000}s.`;

// For the handful of calls whose work scales with the data rather than with one
// upstream hop: the bulk usage delete and reprice, and the pricing-snapshot
// refresh. `DELETE /v1/usage` with `by_filter` issues one unbounded DELETE and
// the reprice loops over every matched row, so on a large imported-usage table
// either can outrun the 30s bound above. Aborting them is worse than waiting: the
// server transaction commits regardless of whether the browser is still
// listening, so the operator would be told the delete failed when it succeeded,
// and the obvious next move is to run it again. Still bounded, because a socket
// held forever is what the deadline exists to prevent.
export const LONG_REQUEST_TIMEOUT_MS = 5 * 60_000;

/** Signal for a request whose duration scales with the data, not with one hop. */
export function longRequestSignal(): AbortSignal {
  return AbortSignal.timeout(LONG_REQUEST_TIMEOUT_MS);
}

// A TimeoutError from AbortSignal.timeout means we gave up, not that the gateway
// is unreachable; saying so points at the right thing to look at. It can surface
// from either await: fetch() resolves once headers arrive, so a body that then
// stalls trips the same deadline on the JSON read instead.
function isTimeout(error: unknown): boolean {
  return error instanceof DOMException && error.name === "TimeoutError";
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  if (init.body != null && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const signal = init.signal ?? AbortSignal.timeout(REQUEST_TIMEOUT_MS);
  // Only name the deadline when it is ours; a caller-supplied signal has its own
  // budget, and quoting 30s at an operator who waited five minutes is worse than
  // saying nothing.
  const timeoutMessage = init.signal ? "The gateway did not respond in time." : TIMEOUT_MESSAGE;

  let response: Response;
  try {
    response = await fetch(path, { ...init, headers, signal });
  } catch (error) {
    if (isTimeout(error)) {
      throw new ApiError(0, timeoutMessage);
    }
    throw new ApiError(0, "Network error: could not reach the gateway.");
  }

  // 401 (expired/revoked session) or 403 both mean this session can't use the
  // management API anymore: drop it and bounce to sign-in.
  if (response.status === 401 || response.status === 403) {
    unauthorizedHandler?.();
    throw new ApiError(response.status, await extractErrorMessage(response));
  }

  if (!response.ok) {
    throw new ApiError(response.status, await extractErrorMessage(response));
  }

  if (response.status === 204) {
    return undefined as T;
  }

  try {
    return (await response.json()) as T;
  } catch (error) {
    // Every caller expects an ApiError; a raw DOMException here would reach the
    // UI as an unrecognized failure. A malformed body is still its own error.
    if (isTimeout(error)) {
      throw new ApiError(0, timeoutMessage);
    }
    throw error;
  }
}
