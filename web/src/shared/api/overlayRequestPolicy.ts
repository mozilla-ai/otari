/**
 * Where the management API is, and what credential a request to it carries.
 *
 * This build answers "the page's own origin, with its own cookies", which is
 * every deployment that serves its dashboard beside its API. A build whose one
 * dashboard reaches several deployments replaces this module at build time and
 * answers with the origin the person chose; that version owns how the choice
 * is made and kept, so the base client holds no state for a topology it does
 * not have.
 *
 * `prepareRequests` runs once, before the bootstrap is read, so a replacing
 * module can settle its answer (a stored choice, a directory it has to fetch)
 * ahead of the first request that depends on it. It must resolve rather than
 * reject: `main.tsx` treats a rejection as "same origin" and carries on, since
 * a dashboard that cannot decide where its API is still has the origin it was
 * served from.
 *
 * **Reached by its `@/shared/api/overlayRequestPolicy` specifier and never
 * relatively**, which is the seam rule and not a style call; `overlaySeams.test.ts`
 * enforces it and web/AGENTS.md says why.
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

export async function prepareRequests(): Promise<void> {}

export function requestPolicy(): RequestPolicy {
  return SAME_ORIGIN_POLICY
}
