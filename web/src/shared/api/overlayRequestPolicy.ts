/**
 * Where the management API is, and what credential a request to it carries.
 *
 * This build answers "the page's own origin, with its own cookies", which is
 * every deployment that serves its dashboard beside its API. A build whose one
 * dashboard reaches several deployments replaces this module at build time and
 * answers with the origin the person chose; that version owns how the choice
 * is made and kept, so the base client holds no state for a topology it does
 * not have. The shape it answers with is `RequestPolicy` in `./requestPolicy`,
 * kept off this module so the replacing one can import it.
 *
 * `prepareRequests` runs once, before the bootstrap is read, so a replacing
 * module can settle its answer (a stored choice, a directory it has to fetch)
 * ahead of the first request that depends on it. Resolving means the answer is
 * settled; rejecting means no deployment could be chosen, and the dashboard
 * then shows the gateway as unreachable rather than reading a bootstrap from
 * wherever the policy happened to point (`app/boot.ts`). A replacing module
 * that wants the page's own origin as its fallback resolves with it.
 *
 * **Reached by its `@/shared/api/overlayRequestPolicy` specifier and never
 * relatively**, which is the seam rule and not a style call; `overlaySeams.test.ts`
 * enforces it and web/AGENTS.md says why.
 */

import { type RequestPolicy, SAME_ORIGIN_POLICY } from "./requestPolicy"

export async function prepareRequests(): Promise<void> {}

export function requestPolicy(): RequestPolicy {
  return SAME_ORIGIN_POLICY
}
