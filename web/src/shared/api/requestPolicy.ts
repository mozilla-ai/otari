/**
 * The shape `overlayRequestPolicy` answers with.
 *
 * In its own module so a build that replaces the seam can import the type:
 * the seam's specifier is what the alias rewrites, so a type on the seam is
 * unreachable from the module standing in for it. Same arrangement as
 * `shared/telemetry/types.ts` beside `overlayTelemetry`.
 */

export interface RequestPolicy {
  /**
   * The origin every API path is prepended with, or `""` for the page's own.
   * A scheme and host with no path or trailing slash, `https://api.example.com`.
   */
  origin: string
  /**
   * `fetch`'s credentials mode: `same-origin` on the page's own origin, and
   * `include` when `origin` names another host, so the session cookie that host
   * set is sent back to it.
   */
  credentials: RequestCredentials
}

export const SAME_ORIGIN_POLICY: RequestPolicy = {
  origin: "",
  credentials: "same-origin",
}
