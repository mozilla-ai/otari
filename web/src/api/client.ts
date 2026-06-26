// Thin fetch wrapper for the gateway's management API. The dashboard is served
// from the same origin as the API, so paths are relative ("/v1/keys"). Every
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

  if (response.status === 401) {
    unauthorizedHandler?.();
    throw new ApiError(401, await extractErrorMessage(response));
  }

  if (!response.ok) {
    throw new ApiError(response.status, await extractErrorMessage(response));
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}
