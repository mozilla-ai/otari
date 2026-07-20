// Thin fetch wrapper for the gateway's management API. The dashboard is served
// from the same origin as the API, so paths are relative ("/v1/models"). Every
// management endpoint requires the master key, sent as a Bearer token.

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

// AuthProvider registers a callback so a 401 anywhere can drop the stored key
// and bounce the operator back to the login screen.
let unauthorizedHandler: (() => void) | null = null;

export function setUnauthorizedHandler(handler: (() => void) | null): void {
  unauthorizedHandler = handler;
}

let masterKey: string | null = null;

export function setMasterKey(key: string | null): void {
  masterKey = key;
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

// Validate a candidate master key against a cheap master-key-gated endpoint
// before we treat the operator as signed in. GET /v1/settings is a small,
// always-present management read, so it works regardless of which management
// pages the dashboard ships. Returns false on 401/403 (wrong key) and throws
// ApiError for network/other failures so the UI can explain them.
export async function validateMasterKey(key: string): Promise<boolean> {
  let response: Response;
  try {
    response = await fetch("/v1/settings", {
      headers: { Accept: "application/json", Authorization: `Bearer ${key}` },
    });
  } catch {
    throw new ApiError(0, "Network error: could not reach the gateway.");
  }
  // Both 401 and 403 mean "not an admin key" for the management endpoints.
  if (response.status === 401 || response.status === 403) {
    return false;
  }
  if (!response.ok) {
    throw new ApiError(response.status, await extractErrorMessage(response));
  }
  return true;
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  if (init.body != null && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (masterKey) {
    headers.set("Authorization", `Bearer ${masterKey}`);
  }

  let response: Response;
  try {
    response = await fetch(path, { ...init, headers });
  } catch {
    throw new ApiError(0, "Network error: could not reach the gateway.");
  }

  // 401 (bad/absent key) or 403 (a valid key that isn't the master key) both mean
  // this session can't use the management API: drop it and bounce to sign-in.
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
