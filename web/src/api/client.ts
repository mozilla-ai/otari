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

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  if (init.body != null && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  let response: Response;
  try {
    response = await fetch(path, { ...init, headers });
  } catch {
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

  return (await response.json()) as T;
}
