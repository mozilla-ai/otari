import { ApiError } from "@/shared/api/client"

/**
 * The code a failure is recorded under.
 *
 * `http_<status>`, which is `otari-ai/frontend/src/shared/helpers/apiError.ts`'s
 * `getAnalyticsErrorCode`. Deliberately the platform's *values* and not only its
 * property names: a value used as a breakdown is as much a shared vocabulary as
 * the event name over it, and `error_code: "credential_rejected"` beside a
 * historical `error_code: "http_401"` splits one funnel in two exactly the way a
 * renamed event would.
 *
 * Never the gateway's message. A refusal's wording is the one part of it that
 * can carry something the operator typed.
 */
export function analyticsErrorCode(error: unknown): string {
  if (error instanceof ApiError) {
    // `ApiError(0, …)` is this client's "never reached the gateway", which is
    // not an HTTP status and must not be recorded as one.
    return error.status === 0 ? "network_error" : `http_${error.status}`
  }
  return "unknown_error"
}

/** The same, for a status this app already holds rather than an error object. */
export function analyticsStatusCode(status: number | undefined): string {
  return status === undefined ? "unknown_error" : `http_${status}`
}
