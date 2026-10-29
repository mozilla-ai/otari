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
  /**
   * Answers this browser keeps from one load to show on the next while the
   * current ones are read (`shared/api/keptAnswers.ts`). Off unless a build
   * that knows its API is far away says otherwise: on a gateway beside its
   * dashboard the round trip saved is a few milliseconds, and a gateway rebuilt
   * on another database at the same address would first show the old one's.
   * Absent reads as off, so a replacing module written before this field
   * keeps nothing.
   */
  keptAnswers?: KeptAnswers
}

/** One switch per answer, so each can be turned on and measured alone. */
export interface KeptAnswers {
  bootstrap: boolean
  organizationContext: boolean
}

export const SAME_ORIGIN_POLICY: RequestPolicy = {
  origin: "",
  credentials: "same-origin",
  keptAnswers: { bootstrap: false, organizationContext: false },
}
